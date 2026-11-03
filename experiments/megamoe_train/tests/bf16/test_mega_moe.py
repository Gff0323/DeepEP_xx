#!/usr/bin/env python3
"""MegaMoE BF16: one entry for full forward/backward precision and timing."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib
from importlib.machinery import EXTENSION_SUFFIXES, PathFinder
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import sysconfig
import traceback
from urllib.parse import urlsplit
from uuid import uuid4


def parse_devices(value):
    try:
        devices = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated physical device IDs") from error
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("Device IDs must be distinct and nonnegative")
    return devices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", type=parse_devices, nargs="?", default=None)
    parser.add_argument("--atk", action="store_true", help="Also require ATK private CV double-benchmark precision")
    parser.add_argument(
        "--atk-ratios",
        type=float,
        nargs=3,
        metavar=("MAX", "MEAN", "RMSE"),
        default=None,
        help="Explicit ATK ratio limits; default 5 / 1.5 / 1.5, number_count_radio stays 2",
    )
    parser.add_argument("--suite", choices=("smoke", "core", "extended", "model"), default=None)
    parser.add_argument("--case-id", action="append", default=None, help="Select an ID within --suite; repeatable")
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List the selected suite without importing Torch/ATK or running devices",
    )
    parser.add_argument(
        "--skip",
        choices=("precision", "performance"),
        default=None,
        help="Skip precision or performance; by default run precision, then performance",
    )
    parser.add_argument("--seq", type=int, default=129, help="Tokens per rank")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gmm-n", type=int, default=1024, help="GMM1 output dimension, equal to 2*F")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--expert-per-rank", type=int, default=2)
    parser.add_argument("--routing-mode", choices=("balanced", "random", "imbalanced"), default="balanced")
    parser.add_argument("--save-gmm1-output", choices=("none", "bf16", "mxfp8", "auto"), default="bf16")
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--chunk-rows", type=int, default=256, help="CPU FP64 expert-row chunk size")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--local-mem-size", type=int, default=None, help="SHMEM bytes; default auto, aligned to 2 MiB")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--shmem-ip-port", default="tcp://127.0.0.1:29862")
    parser.add_argument("--output", type=Path, default=None, help="New report directory")
    args = parser.parse_args(argv)
    if (args.list_cases or args.case_id) and not args.suite:
        parser.error("--list-cases and --case-id require --suite")
    if args.atk and args.skip == "precision":
        parser.error("--atk requires precision; use --skip performance for accuracy only")
    if args.atk_ratios is not None and (
        not args.atk or not all(math.isfinite(value) and value > 0 for value in args.atk_ratios)
    ):
        parser.error("--atk-ratios requires --atk and three finite positive limits")
    if args.suite:
        if "RANK" in os.environ:
            parser.error("Run --suite once with python, rather than under torchrun; single cases support torchrun")
        owned = {
            "--seq",
            "--hidden",
            "--gmm-n",
            "--topk",
            "--expert-per-rank",
            "--routing-mode",
            "--save-gmm1-output",
            "--swiglu-limit",
            "--seed",
        }
        if any(token.partition("=")[0] in owned for token in (sys.argv[1:] if argv is None else argv)):
            parser.error("--suite defines shape, routing, save mode, clipping and seed; omit individual case flags")
    inherited = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    args.devices = args.devices or (
        parse_devices(inherited) if inherited else list(range(int(os.environ.get("LOCAL_WORLD_SIZE", "2"))))
    )
    world = int(os.environ.get("WORLD_SIZE", str(len(args.devices))))
    if not 2 <= world <= 1024 or not 1 <= args.seq <= 16384 or not 1 <= args.expert_per_rank <= 32:
        parser.error("Requires EP in 2..1024, seq in 1..16384 and local experts in 1..32")
    if args.hidden <= 0 or args.hidden % 256 or args.gmm_n <= 0 or args.gmm_n % 512:
        parser.error("Requires hidden % 256 == 0 and gmm-n % 512 == 0, both positive")
    if world * args.expert_per_rank > 2048 or not 1 <= args.topk <= world * args.expert_per_rank:
        parser.error("Requires total experts <= 2048 and 1 <= topk <= total experts")
    if args.seed < 0 or args.chunk_rows <= 0 or args.warmup < 0 or args.loops <= 0 or not 1 <= args.timeout <= 3600:
        parser.error("Invalid seed, chunk size, iteration count or timeout")
    if not math.isfinite(args.swiglu_limit) or abs(args.swiglu_limit) > float.fromhex("0x1.fffffep127"):
        parser.error("swiglu-limit must be finite and representable as float32")
    args.swiglu_limit = struct.unpack("f", struct.pack("f", args.swiglu_limit))[0]
    endpoint = urlsplit(args.shmem_ip_port)
    if (
        endpoint.scheme != "tcp"
        or not endpoint.hostname
        or endpoint.path
        or not endpoint.port
        or not 1024 <= endpoint.port <= 65533
    ):
        parser.error("shmem-ip-port must be tcp://<IPv4>:<port>, with port in 1024..65533")
    import ipaddress

    address = ipaddress.IPv4Address(endpoint.hostname)
    if world > len(args.devices) and address.is_loopback:
        parser.error("Multi-node jobs require a rank-0 address reachable from every node")
    if args.local_mem_size is not None and (args.local_mem_size < 2 << 20 or args.local_mem_size % (2 << 20)):
        parser.error("local-mem-size must be positive and aligned to 2 MiB")
    if "LOCAL_WORLD_SIZE" in os.environ and int(os.environ["LOCAL_WORLD_SIZE"]) != len(args.devices):
        parser.error("The device list must match torchrun's local process count")
    return args


def suite_cases(args):
    manifest = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    defaults = manifest["defaults"]
    seen, selected = set(), []
    for item in manifest["cases"]:
        identity = item["id"]
        if not re.fullmatch(r"[a-z0-9_]+", identity) or identity in seen:
            raise ValueError(f"Invalid/duplicate case ID: {identity}")
        seen.add(identity)
        unknown = set(item) - set(defaults) - {"id", "suites", "covers"}
        if unknown:
            raise ValueError(f"Unknown case fields in {identity}: {sorted(unknown)}")
        if args.suite in item["suites"]:
            selected.append({**defaults, **{k: v for k, v in item.items() if k != "suites"}})
    if args.case_id:
        unknown = set(args.case_id) - {case["id"] for case in selected}
        if unknown:
            raise ValueError(f"Case IDs absent from suite {args.suite}: {sorted(unknown)}")
        selected = [case for case in selected if case["id"] in args.case_id]
    if not selected:
        raise ValueError(f"No cases in suite {args.suite}")
    return selected


def case_argv(args, case, index, output):
    endpoint = urlsplit(args.shmem_ip_port)
    argv = [
        ",".join(map(str, args.devices)),
        "--shmem-ip-port",
        f"tcp://{endpoint.hostname}:{endpoint.port + 4 * index}",
    ]
    for key, value in case.items():
        if key not in ("id", "covers"):
            argv += ["--" + key.replace("_", "-"), str(value)]
    for key in ("chunk_rows", "warmup", "loops", "timeout", "skip", "local_mem_size"):
        value = getattr(args, key)
        if value is not None:
            argv += ["--" + key.replace("_", "-"), str(value)]
    if args.atk:
        argv += ["--atk"]
    if args.atk_ratios is not None:
        argv += ["--atk-ratios", *map(str, args.atk_ratios)]
    return argv + ["--output", str(output)]


def run_suite(args):
    cases = suite_cases(args)
    output = args.output or Path(__file__).resolve().parent / "output" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.suite + "_" + uuid4().hex[:8]
    )
    commands = [case_argv(args, case, index, output / case["id"]) for index, case in enumerate(cases)]
    for command in commands:
        parse_args(command)  # Validate every case before creating directories or starting any job.
    if args.list_cases:
        print("id\tEP\tM/rank\tH\tF\tTopK\tE/rank\trouting\tsave\tclip")
        for case in cases:
            print(
                "\t".join(
                    map(
                        str,
                        (
                            case["id"],
                            len(args.devices),
                            case["seq"],
                            case["hidden"],
                            case["gmm_n"] // 2,
                            case["topk"],
                            case["expert_per_rank"],
                            case["routing_mode"],
                            case["save_gmm1_output"],
                            case["swiglu_limit"],
                        ),
                    )
                )
            )
        print(f"{len(cases)} cases; list only, no imports of Torch/ATK and no device execution")
        return 0
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.devices))
    installed_root()
    if args.atk:
        from atk_compare import require_atk
        from precision_metrics import OUTPUTS

        require_atk()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "suite": args.suite,
        "devices": args.devices,
        "world_size": len(args.devices),
        "atk_requested": args.atk,
        "skip": args.skip,
        "passed": False,
        "cases": [{"id": case["id"], "config": case, "status": "NOT_RUN"} for case in cases],
    }
    summary_file = output / "suite_summary.json"

    def save():
        summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save()
    runtime_error = False
    for index, (record, command) in enumerate(zip(summary["cases"], commands, strict=True)):
        print(f"[CASE] {index + 1}/{len(cases)} {record['id']}", flush=True)
        try:
            with (output / (record["id"] + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), *command], stdout=log, stderr=subprocess.STDOUT
                )
            record["exit_code"] = process.returncode
            report_file = output / record["id"] / "summary.json"
            if process.returncode in (0, 2) and report_file.is_file():
                result = json.loads(report_file.read_text(encoding="utf-8"))
                ranks = result["ranks"]
                if {rank["rank"] for rank in ranks} != set(range(len(args.devices))) or len(ranks) != len(args.devices):
                    raise RuntimeError(f"Incomplete rank coverage in {report_file}")
                if result["passed"] != all(rank["passed"] for rank in ranks) or result["passed"] != (
                    process.returncode == 0
                ):
                    raise RuntimeError(f"Inconsistent report/exit code in {report_file}")
                if args.atk:
                    if not result.get("atk_requested") or not result["precision_run"]:
                        raise RuntimeError(f"ATK precision did not run in {report_file}")
                    for rank in ranks:
                        check = rank.get("precision", {}).get("atk", {})
                        outputs = check.get("outputs", {})
                        if (
                            check.get("provider") != "private"
                            or check.get("standard") != "cv_fused_double_benchmark"
                            or set(outputs) != set(OUTPUTS)
                            or check.get("passed") != all(value.get("passed", False) for value in outputs.values())
                            or (rank["passed"] and not check.get("passed"))
                        ):
                            raise RuntimeError(
                                f"Incomplete/inconsistent ATK results in {report_file}, rank={rank['rank']}"
                            )
                record.update(
                    status="PASS" if result["passed"] else "FAIL",
                    report=str(report_file),
                    precision_run=result["precision_run"],
                    performance_run=result["performance_run"],
                )
            else:
                record["status"] = "ERROR"
                runtime_error = True
        except BaseException as error:
            record.update(status="ERROR", error=f"{type(error).__name__}: {error}")
            save()
            raise
        save()
        print(f"[CASE] {record['id']} {record['status']} log={output / (record['id'] + '.log')}", flush=True)
        if runtime_error:
            break  # Device/communication/setup errors stop the suite; remaining cases stay NOT_RUN.
    summary["passed"] = all(record["status"] == "PASS" for record in summary["cases"])
    save()
    print(f"[SUITE] {'PASS' if summary['passed'] else 'FAIL'} report={summary_file}", flush=True)
    return 1 if runtime_error else 0 if summary["passed"] else 2


def installed_root():
    locations = list(dict.fromkeys(sysconfig.get_path(name) for name in ("purelib", "platlib")))
    spec = PathFinder.find_spec("deep_ep", locations)
    hint = "Build with DEEPEP_BUILD_MEGAMOE_TRAIN=ON and install its wheel in this Python environment."
    if spec is None or spec.origin is None:
        raise RuntimeError(f"No installed deep_ep package for {sys.executable}. {hint}")
    root = Path(spec.origin).resolve().parent.parent
    for name in ("deep_ep", "deep_ep_experimental", "deep_ep_experimental.megamoe_train"):
        directory = root.joinpath(*name.split("."))
        if not (directory / "__init__.py").is_file():
            raise RuntimeError(f"Missing installed package {directory}. {hint}")
        if name != "deep_ep_experimental" and not any(
            (directory / ("_C" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES
        ):
            raise RuntimeError(f"Missing installed native extension {name}._C. {hint}")
    for name, module in tuple(sys.modules.items()):
        top = name.partition(".")[0]
        if top in ("deep_ep", "deep_ep_experimental"):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root / top):
                raise RuntimeError(
                    f"{name} was already loaded from {origin}; expected installed wheel at {root}. Restart the process."
                )
    if not sys.path or Path(sys.path[0]).resolve() != root:
        sys.path.insert(0, str(root))
    return root


def make_inputs(args, rank, world, global_tokens=True):
    """Generate reproducible BF16 CPU snapshots shared by all three paths."""
    import torch

    def rng(source, stream):
        seed = (args.seed + source * 1000003 + stream * 1000000007) % (1 << 63)
        return torch.Generator(device="cpu").manual_seed(seed)

    def uniform(generator, shape, bound):
        value = torch.empty(shape, dtype=torch.float32, device="cpu")
        return value.uniform_(-bound, bound, generator=generator).to(torch.bfloat16)

    xs, probabilities, upstreams, all_ids = [], [], [], []
    total = world * args.expert_per_rank
    for source in range(world):
        if args.routing_mode == "balanced":
            ids = (
                ((torch.arange(args.seq * args.topk, device="cpu") + source * args.seq * args.topk) % total)
                .reshape(args.seq, args.topk)
                .to(torch.int32)
            )
        else:
            ids = torch.empty((args.seq, args.topk), dtype=torch.int32, device="cpu")
            generator = rng(source, 1)
            skew = torch.zeros(total, dtype=torch.float32, device="cpu")
            if args.routing_mode == "imbalanced":
                skew = -2.0 * torch.arange(total, dtype=torch.float32, device="cpu") / max(total - 1, 1)
            for begin in range(0, args.seq, 256):
                stop = min(begin + 256, args.seq)
                uniform_scores = torch.rand(
                    (stop - begin, total), dtype=torch.float32, device="cpu", generator=generator
                ).clamp_min(torch.finfo(torch.float32).tiny)
                scores = -torch.log(-torch.log(uniform_scores)) + skew
                ids[begin:stop] = scores.topk(args.topk, dim=1).indices.to(torch.int32)
        all_ids.append(ids)
        if global_tokens or source == rank:
            xs.append(uniform(rng(source, 0), (args.seq, args.hidden), 0.1))
            logits = torch.empty((args.seq, args.topk), dtype=torch.float32, device="cpu")
            logits.uniform_(-1, 1, generator=rng(source, 2))
            probabilities.append(torch.softmax(logits, dim=1).to(torch.bfloat16))
            upstreams.append(uniform(rng(source, 3), (args.seq, args.hidden), 0.1))
    ids = torch.cat(all_ids)
    counts = torch.bincount(ids.reshape(-1).long(), minlength=total)
    w1 = uniform(rng(rank, 4), (args.expert_per_rank, args.gmm_n, args.hidden), 1.0)
    w2 = uniform(rng(rank, 5), (args.expert_per_rank, args.hidden, args.gmm_n // 2), 1.0)
    return {
        "x": torch.cat(xs),
        "ids": ids if global_tokens else all_ids[rank],
        "p": torch.cat(probabilities),
        "w1": w1,
        "w2": w2,
        "dy": torch.cat(upstreams),
        "all_counts": counts,
    }


def combine_baseline(partial, args, rank, world, device):
    """Move route values through Gloo, then apply the selected NPU combine."""
    import torch
    import torch.distributed as dist

    from npu_baseline import unpermute_routes

    m, h = args.seq, args.hidden
    for destination in range(world):
        # Every row/slot has one owning expert. FP32 carries BF16 values
        # exactly; this reduction transports them without combining routes.
        buffer = torch.zeros((m, args.topk, 2 * h + 1), dtype=torch.float32, device="cpu")
        for route in partial["routes"]:
            mask = route["rows"] // m == destination
            rows, slots = route["rows"][mask] % m, route["slots"][mask]
            buffer[rows, slots, :h] = route["y"][mask].float()
            buffer[rows, slots, h : 2 * h] = route["dx"][mask].float()
            buffer[rows, slots, -1] = route["dp"][mask].float()
        dist.reduce(buffer, dst=destination)
        if rank == destination:
            ids = partial["ids"][rank * m : (rank + 1) * m]
            result = unpermute_routes(buffer, ids, device, reference=partial["reference"])
    result.update(dw1=partial["dw1"], dw2=partial["dw2"])
    return result


def local_truth(inputs, args, rank):
    import torch.distributed as dist

    from cpu_reference import run_reference

    result = run_reference(inputs, rank * args.expert_per_rank, args.chunk_rows, args.swiglu_limit)
    sl = slice(rank * args.seq, (rank + 1) * args.seq)
    for name in ("y", "dx", "dtopk_weights"):
        dist.all_reduce(result[name])
        result[name] = result[name][sl].clone()
    return result


def measure(op, tensors, capacity, args):
    import torch

    x, ids, p, w1, w2, dy = tensors
    forward_times, backward_times = [], []
    for iteration in range(args.warmup + args.loops):
        for leaf in (x, p, w1, w2):
            leaf.grad = None
        start, middle, end = (torch.npu.Event(enable_timing=True) for _ in range(3))
        start.record()
        y, counts = op(
            x, ids, p, w1, w2, capacity, save_gmm1_output=args.save_gmm1_output, swiglu_limit=args.swiglu_limit
        )
        middle.record()
        y.backward(dy)
        end.record()
        torch.npu.synchronize()
        if iteration >= args.warmup:
            forward_times.append(float(start.elapsed_time(middle)))
            backward_times.append(float(middle.elapsed_time(end)))
    if not all(bool(torch.isfinite(value).all().item()) for value in (y, x.grad, p.grad, w1.grad, w2.grad)):
        raise RuntimeError("Non-finite output or gradient after timing")
    return {
        "forward_ms": {"samples": forward_times, "median": statistics.median(forward_times)},
        "backward_ms": {"samples": backward_times, "median": statistics.median(backward_times)},
        "scope": "Synchronized NPU Event latency of operator forward and autograd backward; reference, reporting and loss construction excluded",
    }, counts


def run_rank(local_rank, args, rank=None, world=None):
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from npu_baseline import run_baseline
    from precision_metrics import RATIO_LIMITS, evaluate, write_report

    rank = local_rank if rank is None else rank
    world = len(args.devices) if world is None else world
    print(f"[STAGE] rank={rank} logical_device={local_rank} physical_device={args.devices[local_rank]}", flush=True)
    torch.npu.set_device(local_rank)
    installed_root()
    for name in ("deep_ep._C", "deep_ep_experimental.megamoe_train._C"):
        importlib.import_module(name)
    from deep_ep_experimental.megamoe_train import MegaMoeBF16Train

    endpoint = urlsplit(args.shmem_ip_port)
    store = dist.TCPStore(
        endpoint.hostname, endpoint.port + 2, world, rank == 0, timeout=timedelta(seconds=args.timeout)
    )
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world, timeout=timedelta(seconds=args.timeout))
    try:
        config = {key: value for key, value in vars(args).items() if key not in ("devices", "output")}
        configurations = [None] * world
        dist.all_gather_object(configurations, config)
        if any(value != configurations[0] for value in configurations):
            raise ValueError("All ranks must use identical test configuration")
        directory = [
            str(
                args.output
                or (
                    Path(__file__).resolve().parent
                    / "output"
                    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
                )
            )
            if rank == 0
            else None
        ]
        dist.broadcast_object_list(directory, src=0)
        output = Path(directory[0])
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        precision = args.skip != "precision"
        performance = args.skip != "performance"
        print(f"[STAGE] rank={rank} prepare inputs", flush=True)
        inputs = make_inputs(args, rank, world, global_tokens=precision)
        expected = inputs["all_counts"][rank * args.expert_per_rank : (rank + 1) * args.expert_per_rank]
        capacity = (
            math.ceil(int(inputs["all_counts"].reshape(world, args.expert_per_rank).sum(dim=1).max().item()) / 128)
            * 128
        )
        raw_baseline = mixed_baseline = truth = None
        clipping_disabled = args.swiglu_limit <= 0 or args.swiglu_limit == float(torch.finfo(torch.float32).max)
        mode = ("mxfp8" if clipping_disabled else "bf16") if args.save_gmm1_output == "auto" else args.save_gmm1_output
        if precision:
            print(f"[STAGE] rank={rank} CPU FP64 reference", flush=True)
            truth = local_truth(inputs, args, rank)
            print(f"[STAGE] rank={rank} NPU BigOp BF16 baseline", flush=True)
            raw_baseline = combine_baseline(
                run_baseline(
                    inputs,
                    f"npu:{local_rank}",
                    rank * args.expert_per_rank,
                    "bf16",
                    args.swiglu_limit,
                    reference="bigop",
                ),
                args,
                rank,
                world,
                f"npu:{local_rank}",
            )
            print(f"[STAGE] rank={rank} NPU MegaMoE contract reference (save={mode})", flush=True)
            mixed_baseline = combine_baseline(
                run_baseline(inputs, f"npu:{local_rank}", rank * args.expert_per_rank, mode, args.swiglu_limit),
                args,
                rank,
                world,
                f"npu:{local_rank}",
            )
        sl = slice(rank * args.seq, (rank + 1) * args.seq) if precision else slice(None)

        def tensor(value, dtype=torch.bfloat16, grad=False):
            return value.detach().to(device=f"npu:{local_rank}", dtype=dtype).contiguous().requires_grad_(grad)

        x, ids, p, w1, w2, dy = (
            tensor(inputs["x"][sl], grad=True),
            tensor(inputs["ids"][sl], torch.int32),
            tensor(inputs["p"][sl], grad=True),
            tensor(inputs["w1"], grad=True),
            tensor(inputs["w2"], grad=True),
            tensor(inputs["dy"][sl]),
        )
        tensors = (x, ids, p, w1, w2, dy)
        del inputs
        needed = (72 << 10) + world * math.ceil(world * args.expert_per_rank / 128) * 128 * 4
        needed += 4 * args.seq * args.topk * args.hidden + math.ceil(2 * args.seq * args.topk / 512) * 512 + (2 << 20)
        local_mem = args.local_mem_size or math.ceil(max(2 << 30, needed) / (2 << 20)) * (2 << 20)
        os.environ["DEEPEP_SHMEM_ENDPOINT"] = args.shmem_ip_port
        print(
            f"[STAGE] rank={rank} fused forward/backward; max_recv_tokens={capacity} shmem_bytes={local_mem}",
            flush=True,
        )
        op = MegaMoeBF16Train(dist.group.WORLD, local_mem, timeout=args.timeout)
        # On runtime/device exceptions restart the job; do not attempt a collective
        # close while peers may still be executing a different operation.
        y, counts = op(
            x, ids, p, w1, w2, capacity, save_gmm1_output=args.save_gmm1_output, swiglu_limit=args.swiglu_limit
        )
        y.backward(dy)
        torch.npu.synchronize()
        values = dict(y=y, dx=x.grad, dtopk_weights=p.grad, dw1=w1.grad, dw2=w2.grad, counts=counts)
        if any(value is None for value in values.values()):
            raise RuntimeError("Fused forward/backward did not produce every required result")
        if counts.dtype != torch.int32 or tuple(counts.shape) != (args.expert_per_rank,):
            raise RuntimeError("Invalid expert-count dtype or shape")
        report = {
            "rank": rank,
            "config": config,
            "world_size": world,
            "resolved_save_mode": mode,
            "max_recv_tokens": capacity,
            "shmem_bytes": local_mem,
            "precision_run": precision,
            "performance_run": False,
            "references": {"mixed_tolerance": "megamoe_contract", "dual_baseline": "bigop_bf16_vs_cpu_fp64"},
        }
        if precision:
            actual = {name: value.detach().cpu().clone() for name, value in values.items()}
            report["precision"] = evaluate(actual, raw_baseline, mixed_baseline, truth, expected)
            if args.atk:
                from atk_compare import compare_with_atk

                report["precision"]["atk"] = compare_with_atk(
                    actual,
                    raw_baseline,
                    truth,
                    output / f"atk_rank_{rank}",
                    args.atk_ratios or tuple(RATIO_LIMITS[key] for key in ("MARE", "MERE", "RMSE")),
                )
                report["precision"]["passed"] &= report["precision"]["atk"]["passed"]
            report["passed"] = report["precision"]["passed"]
            checks = report["precision"]["outputs"]
            mixed_ok = all(value["mixed_tolerance"]["passed"] for value in checks.values())
            dual_ok = all(value["dual_baseline"]["passed"] for value in checks.values())
            ratios = {
                name: max(value["dual_baseline"].get("ratios", {}).get(name, math.inf) for value in checks.values())
                for name in RATIO_LIMITS
            }
            print(
                f"[PRECISION] rank={rank} mixed_tolerance={'PASS' if mixed_ok else 'FAIL'} reference=megamoe_contract "
                f"dual_baseline={'PASS' if dual_ok else 'FAIL'} "
                f"counts={'PASS' if report['precision']['counts_passed'] else 'FAIL'} "
                + " ".join(f"R_{name}={ratio:.6g}" for name, ratio in ratios.items()),
                flush=True,
            )
            if args.atk:
                atk_report = report["precision"]["atk"]
                print(
                    f"[ATK] rank={rank} {'PASS' if atk_report['passed'] else 'FAIL'} standard={atk_report['standard']} provider={atk_report['provider']}",
                    flush=True,
                )
                for name, check in atk_report["outputs"].items():
                    if not check["passed"]:
                        print(f"[ATK FAIL] rank={rank} {name}: {check['result']}", flush=True)
            for name, checks in report["precision"].get("outputs", {}).items():
                if not checks["mixed_tolerance"]["passed"] or not checks["dual_baseline"]["passed"]:
                    print(f"[FAIL] rank={rank} {name}: {checks}", flush=True)
            del actual
        else:
            report["passed"] = all(
                bool(torch.isfinite(value).all().item()) for name, value in values.items() if name != "counts"
            )
            report["passed"] &= torch.equal(counts.cpu().to(torch.int64), expected)
            report["precision"] = {
                "status": "NOT_RUN",
                "reason": "--skip precision",
                "scope": "finite/count smoke checks only",
            }
        del values, y, counts, raw_baseline, mixed_baseline, truth
        failed = torch.tensor(int(not report["passed"]), dtype=torch.int32)
        dist.all_reduce(failed)
        if not int(failed.item()) and performance:
            print(f"[STAGE] rank={rank} performance warmup/timing", flush=True)
            report["performance"], counts = measure(op, tensors, capacity, args)
            report["performance_run"] = True
            counts_ok = torch.equal(counts.cpu().to(torch.int64), expected)
            report["performance"]["passed"] = counts_ok
            report["passed"] &= counts_ok
        else:
            report["performance"] = {
                "status": "NOT_RUN",
                "reason": "--skip performance" if not performance else "precision or smoke checks failed",
            }
        op.close()
        write_report(output / f"rank_{rank}.json", report)
        reports = [None] * world
        dist.all_gather_object(reports, report)
        passed = all(value["passed"] for value in reports)
        if rank == 0:
            write_report(
                output / "summary.json",
                {
                    "passed": passed,
                    "skip": args.skip,
                    "atk_requested": args.atk,
                    "precision_run": precision,
                    "precision_passed": all(value["precision"]["passed"] for value in reports) if precision else None,
                    "performance_run": all(value["performance_run"] for value in reports),
                    "ranks": reports,
                    "precision_reference_commit": "568ac7c07e2e669582f7ce684bf5dd27f503efad",
                },
            )
            print(f"[RESULT] {'PASS' if passed else 'FAIL'} skip={args.skip or 'none'} output={output}", flush=True)
            if passed and performance:
                for value in reports:
                    perf = value["performance"]
                    print(
                        f"[PERF] rank={value['rank']} forward_ms={perf['forward_ms']['median']:.6f} backward_ms={perf['backward_ms']['median']:.6f}",
                        flush=True,
                    )
        if not passed:
            raise SystemExit(2)
    finally:
        dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    if args.suite:
        return run_suite(args)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.devices))
    installed_root()
    if args.atk:
        from atk_compare import require_atk

        require_atk()
    if "RANK" in os.environ:
        run_rank(int(os.environ["LOCAL_RANK"]), args, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
        return 0
    import torch.multiprocessing as mp

    try:
        mp.spawn(run_rank, args=(args,), nprocs=len(args.devices), join=True)
    except mp.ProcessExitedException as error:
        return 2 if error.exit_code == 2 else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
