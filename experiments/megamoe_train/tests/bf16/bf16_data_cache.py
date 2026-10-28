"""Persistent input/golden cache; NPU outputs are never cached.

Cache hits check exact file lengths and recorded modification times, without
rehashing multi-GB weights. Materialized hard links pin the old files when a
cache entry is refreshed. Cache files must be treated as read-only inputs.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


CACHE_VERSION = 1


def _identity(case, seed, options, golden):
    # Kernel commits, timing settings, tolerances and SHMEM allocation sizes
    # do not change the stimulus. Generator changes invalidate the cache.
    source = Path(golden.__file__).read_text(encoding="utf-8")
    return {
        "version": CACHE_VERSION,
        "generator_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "numpy": golden.np.__version__,
        "case": {key: case[key] for key in ("M", "N", "K", "EP", "TOPK", "expertPerRank")},
        "capacity": {key: case.get(key) for key in ("maxRecvTokenNum", "validateRecvCapacity")},
        "seed": seed,
        **options,
    }


def _file_sizes(case, include_golden, golden):
    m, n, k, topk, experts = (case[key] for key in ("M", "N", "K", "TOPK", "expertPerRank"))
    sizes = {
        "input_x_bf16": m * k * 2,
        "input_expert_ids_bf16": m * topk * 4,
        "input_expert_weights_bf16": m * topk * 2,
        "weight1_bf16": experts * n * k * 2,
        "weight2_bf16": experts * k * (n // 2) * 2,
    }
    if include_golden:
        sizes.update(golden_y_bf16=m * k * 2, golden_expert_token_nums_bf16=experts * 4)
    return {
        golden.data_path(Path(), rank, name, case).name: size
        for rank in range(case["EP"])
        for name, size in sizes.items()
    }


@contextmanager
def _entry_lock(path):
    # OS locks are released on process exit, including an interrupted golden
    # generation. A leftover lock file is not a stale lock.
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        if os.name == "nt":
            import msvcrt

            def acquire():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(handle, fcntl.LOCK_UN)

        next_report = 0.0
        while True:
            try:
                acquire()
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= next_report:
                    print(f"[DATA_CACHE] WAIT another process is preparing {path.parent}", flush=True)
                    next_report = time.monotonic() + 60
                time.sleep(0.2)
        try:
            yield
        finally:
            release()


def _check(manifest_path, payload, identity, required, need_golden):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no readable completed manifest"
    if not isinstance(manifest, dict) or manifest.get("identity") != identity:
        return "cache identity changed"
    if need_golden and manifest.get("has_golden") is not True:
        return "input-only cache; precision requires golden"
    records = manifest.get("files")
    if not isinstance(records, dict):
        return "missing file records"
    for name, size in required.items():
        try:
            stat = (payload / name).stat()
        except OSError:
            return f"missing file: {name}"
        if not (payload / name).is_file() or stat.st_size != size:
            return f"file size mismatch: {name}"
        if records.get(name) != {"size": size, "mtime_ns": stat.st_mtime_ns}:
            return f"file changed: {name}"
    return None


def _materialize(payload, data_dir, names):
    copied = 0
    for name in names:
        source, target = payload / name, data_dir / name
        if target.is_file() and not target.is_symlink() and os.path.samefile(source, target):
            continue
        temporary = data_dir / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            try:
                os.link(source, temporary)
            except OSError as error:
                if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
                    raise
                if copied == 0:
                    print(
                        "[DATA_CACHE] Hard links unavailable; copying inputs. "
                        "Place --data-cache-dir on the results filesystem to avoid large copies.",
                        flush=True,
                    )
                shutil.copy2(source, temporary)
                copied += 1
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"[DATA_CACHE] materialized files={len(names)} copied={copied} data_dir={data_dir}", flush=True)


def prepare_data(
    case,
    data_dir,
    seed,
    *,
    cache_root,
    refresh=False,
    generate_golden=True,
    routing_mode="balanced",
    routing_skew=1.0,
    topk_weight_mode="random",
    early_topk_weight=True,
):
    import mega_moe_bf16_golden as golden

    options = dict(
        routing_mode=routing_mode,
        routing_skew=routing_skew,
        topk_weight_mode=topk_weight_mode,
        early_topk_weight=early_topk_weight,
    )
    identity = _identity(case, seed, options, golden)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    entry = Path(cache_root).resolve() / key
    payload = entry / "data"
    data_dir = Path(data_dir).resolve()
    if data_dir == entry or entry in data_dir.parents or data_dir in entry.parents:
        raise ValueError("Test result directory and cache entry must be separate")
    payload.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    required = _file_sizes(case, generate_golden, golden)
    manifest_path = entry / "manifest.json"
    with _entry_lock(entry / "prepare.lock"):
        reason = "forced refresh" if refresh else _check(manifest_path, payload, identity, required, generate_golden)
        if reason is None:
            print(f"[DATA_CACHE] HIT key={key[:12]} golden_required={generate_golden} cache={entry}", flush=True)
        else:
            print(f"[DATA_CACHE] MISS key={key[:12]} reason={reason} cache={entry}", flush=True)
            # Invalidate first. Unlink known cache files before generation so
            # existing result hard links retain their original data/weights.
            manifest_path.unlink(missing_ok=True)
            for name in _file_sizes(case, True, golden):
                (payload / name).unlink(missing_ok=True)
            golden.generate(case, payload, seed, generate_golden=generate_golden, **options)
            records = {}
            for name, size in required.items():
                stat = (payload / name).stat()
                if not (payload / name).is_file() or stat.st_size != size:
                    raise ValueError(
                        f"Generated file has incorrect size: {name}; expected={size} actual={stat.st_size}"
                    )
                records[name] = {"size": size, "mtime_ns": stat.st_mtime_ns}
            manifest = {"identity": identity, "has_golden": generate_golden, "files": records}
            temporary = entry / "manifest.tmp"
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temporary, manifest_path)
            print(f"[DATA_CACHE] READY key={key[:12]} files={len(records)} golden={generate_golden}", flush=True)
        _materialize(payload, data_dir, required)
        (data_dir / "input_cache.json").write_text(
            json.dumps({"cache": str(entry), "identity": identity}, indent=2), encoding="utf-8"
        )
    return reason is None
