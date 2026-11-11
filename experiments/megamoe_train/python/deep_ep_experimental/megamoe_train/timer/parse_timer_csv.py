# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""Decode one raw MegaMoe BF16 training timer dump into per-core CSV rows."""

import argparse
import csv
import json
import sys
from array import array
from pathlib import Path

from . import protocol as p


def load_raw(path):
    """Read a flat int64 dump (``.bin``) or the legacy CSV-of-integers form."""
    path = Path(path)
    values = array("q")
    if path.suffix == ".bin":
        if path.stat().st_size != p.TOTAL_BUFFER_SIZE * 8:
            raise ValueError(
                f"{path}: {path.stat().st_size} bytes; expected {p.TOTAL_BUFFER_SIZE * 8}. "
                "The C++ and Python timer schemas are out of sync."
            )
        with path.open("rb") as stream:
            values.fromfile(stream, p.TOTAL_BUFFER_SIZE)
        if sys.byteorder != "little":
            values.byteswap()
    else:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                values.extend(int(item.strip()) for item in row if item.strip())
        if len(values) != p.TOTAL_BUFFER_SIZE:
            raise ValueError(f"{path}: got {len(values)} int64 values, expected {p.TOTAL_BUFFER_SIZE}")
    return values


def _events(kernel, dynamic_counts):
    """Names and logical slots actually present in this dump.

    Forward writes its 11 per-expert types; backward writes the 9 that live
    after them. A slot whose device-side count is zero is omitted rather than
    printed as a zero row.
    """
    if kernel == "forward":
        fixed, dynamic = p.FIXED_NAMES, list(enumerate(p.FORWARD_DYNAMIC_NAMES))
    elif kernel == "backward":
        fixed = p.BACKWARD_FIXED_NAMES
        dynamic = list(enumerate(p.BACKWARD_DYNAMIC_NAMES, start=p.BACKWARD_DYNAMIC_BASE))
    else:
        raise ValueError(f"unknown kernel {kernel!r}; expected forward or backward")

    events = [(name, index) for index, name in enumerate(fixed)]
    counts = {}
    for type_index, name in dynamic:
        count = max(
            dynamic_counts[core * p.DYNAMIC_ITER_PER_CORE + type_index] for core in range(p.N_CORE_COUNT)
        )
        if not 0 <= count <= p.MAX_DYNAMIC_ITER:
            raise ValueError(
                f"{name}: device reported {count} iterations, capacity is {p.MAX_DYNAMIC_ITER}; "
                "raise MAX_DYNAMIC_ITER in both schemas and rebuild"
            )
        counts[name] = count
        for iteration in range(count):
            events.append((f"{name}_{iteration}", p.dynamic_logical_index(type_index, iteration)))
    return events, dynamic, counts


def decode(raw_path, kernel, *, cores=None):
    """Yield one dict per logical core with ``{event: (start, end, us)}``.

    A nonzero ``AIV_*`` span recorded by a cube core (or the reverse) means a
    probe landed on the wrong engine, so it is reported instead of being
    plotted on a lane where it cannot have happened.
    """
    raw = load_raw(raw_path)
    dynamic_counts = raw[p.N_TIMING_COUNTER:]
    events, dynamic, counts = _events(kernel, dynamic_counts)
    rows = []
    for core_id in range(p.N_CORE_COUNT if cores is None else cores):
        group, engine, lane = p.core_lane(core_id)
        values = {}
        for name, logical_index in events:
            base = p.counter_index(core_id, logical_index)
            start, end = raw[base], raw[base + 1]
            duration = (end - start) / p.CYCLE_TO_US if end > start else 0.0
            if duration > 0.0:
                _check_lane(raw_path, name, core_id, engine)
            values[name] = (start, end, duration)
        rows.append(
            dict(
                core_id=core_id,
                group=group,
                engine=engine,
                lane=lane,
                values=values,
                counts={name: counts[name] for _, name in dynamic},
            )
        )
    return rows


def _check_lane(raw_path, name, core_id, engine):
    expected = "AIC" if name.startswith("AIC_") else ("AIV" if name.startswith("AIV_") else None)
    if expected is not None and not engine.startswith(expected):
        raise ValueError(
            f"{raw_path}: slot {name} is nonzero on logical core {core_id} ({engine}); "
            f"only {expected} cores may record it. A probe is on the wrong engine path - "
            "recheck the Tik/Tok placement rather than the decoder."
        )


def write_csv(rows, path, kernel):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    events = list(rows[0]["values"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["core_id", "group", "engine", "lane"]
        for name in events:
            header.extend((f"{name}_start_cycle", f"{name}_end_cycle", f"{name}_duration(us)"))
        header.extend(f"{name}_iterations" for name in rows[0]["counts"])
        writer.writerow(header)
        for row in rows:
            line = [row["core_id"], row["group"], row["engine"], row["lane"]]
            for name in events:
                start, end, duration = row["values"][name]
                line.extend((start, end, f"{duration:.3f}"))
            line.extend(row["counts"][name] for name in row["counts"])
            writer.writerow(line)
    print(f"[TIMER] decoded {kernel} timer -> {path}")


def summarize(rows, *, top=16):
    """Per-stage slowest lane (us), i.e. the wall-clock contribution.

    Cross-core sums are meaningless here: cores split tiles of the same stage,
    so the slowest core is what bounds that stage.
    """
    totals = {}
    for row in rows:
        for name, (_, _, duration) in row["values"].items():
            if duration <= 0:
                continue
            key = (row["engine"], name)
            totals[key] = max(totals.get(key, 0.0), duration)
    result = []
    for (engine, name), duration in sorted(totals.items(), key=lambda item: -item[1])[:top]:
        result.append({"engine": engine, "stage": name, "max_us": duration})
    return result


def check_cpp_schema(root=None):
    """Compare this mirror against the device headers, when they are present.

    A silent drift between the C++ layout and this file would show up as
    plausible-looking garbage columns, so the check compares the numbers that
    define the layout rather than trusting a hand-maintained copy.
    """
    base = Path(root) if root else Path(__file__).resolve().parents[4] / "csrc" / "kernels" / "bf16" / "common"
    shared = base / "ascend_timer_v2.hpp"
    backward = base / "mega_moe_bf16_backward_timer.hpp"
    if not shared.exists() or not backward.exists():
        return {"status": "SKIPPED", "reason": f"device headers not found under {base}"}
    shared_text = shared.read_text(encoding="utf-8")
    backward_text = backward.read_text(encoding="utf-8")

    def count_entries(text, enum_name):
        body = text.split(f"enum {enum_name} {{", 1)[1].split("};", 1)[0]
        entries = []
        for line in body.splitlines():
            line = line.split("//", 1)[0].strip()
            if not line or not line.endswith(","):
                continue
            entries.append(line)
        return entries

    fixed_entries = count_entries(shared_text, "FixedTiming")
    dynamic_entries = count_entries(shared_text, "DynamicTimingType")
    max_iter = int(__import__("re").search(r"MAX_DYNAMIC_ITER\s*=\s*(\d+)", shared_text).group(1))
    cores = int(__import__("re").search(r"N_CORE_COUNT\s*=\s*(\d+)", shared_text).group(1))
    bases = [
        int(__import__("re").search(rf"{name}\s*=\s*(\d+)", backward_text).group(1))
        for name in ("AIC_DZ_WAIT_EXPERT_BASE", "AIV_DX_WAIT_EXPERT_BASE", "AIC_DO_REUSE_WAIT_EXPERT_BASE")
    ]

    problems = []
    if len(fixed_entries) != len(p.FIXED_NAMES):
        problems.append(f"fixed slots: C++ {len(fixed_entries)} vs Python {len(p.FIXED_NAMES)}")
    if len(dynamic_entries) != len(p.DYNAMIC_NAMES):
        problems.append(f"dynamic slots: C++ {len(dynamic_entries)} vs Python {len(p.DYNAMIC_NAMES)}")
    if max_iter != p.MAX_DYNAMIC_ITER:
        problems.append(f"MAX_DYNAMIC_ITER: C++ {max_iter} vs Python {p.MAX_DYNAMIC_ITER}")
    if cores != p.N_CORE_COUNT:
        problems.append(f"N_CORE_COUNT: C++ {cores} vs Python {p.N_CORE_COUNT}")
    expected_bases = [
        len(p.BACKWARD_FIXED_NAMES) - 3 * p.MAX_EXPERTS,
        len(p.BACKWARD_FIXED_NAMES) - 2 * p.MAX_EXPERTS,
        len(p.BACKWARD_FIXED_NAMES) - p.MAX_EXPERTS,
    ]
    if bases != expected_bases:
        problems.append(f"backward expert bases: C++ {bases} vs Python {expected_bases}")
    if problems:
        raise ValueError("timer schema drift: " + "; ".join(problems))
    return {
        "status": "OK",
        "fixed": len(fixed_entries),
        "dynamic": len(dynamic_entries),
        "max_iter": max_iter,
        "cores": cores,
        "total_buffer_size": p.TOTAL_BUFFER_SIZE,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="?", help="rank_<n> timer dump (.bin or CSV of int64)")
    parser.add_argument("--kernel", choices=("forward", "backward"))
    parser.add_argument("--output", type=Path, help="parsed CSV path; default next to the dump")
    parser.add_argument("--summary", action="store_true", help="print the slowest lanes per stage")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument(
        "--trace",
        type=Path,
        nargs="?",
        const=Path(""),
        help="also write a Chrome/Perfetto trace JSON (default next to the dump); "
        "drop it on ui.perfetto.dev to see the pipeline",
    )
    parser.add_argument("--rank", type=int, default=0, help="rank label used in the trace")
    parser.add_argument("--no-wait", action="store_true", help="omit dependency-wait slices from the trace")
    parser.add_argument("--kernel-bar", action="store_true", help="include the whole-kernel bar in the trace")
    parser.add_argument("--check-schema", action="store_true", help="only report the expected buffer size")
    args = parser.parse_args(argv)
    if args.check_schema:
        print(json.dumps({
            "python": {"total_buffer_size": p.TOTAL_BUFFER_SIZE, "bytes": p.timer_buffer_bytes()},
            "cpp": check_cpp_schema(),
        }))
        return 0
    if args.raw is None or args.kernel is None:
        parser.error("raw and --kernel are required unless --check-schema is given")
    output = args.output or args.raw.with_name(args.raw.stem + "_parsed.csv")
    rows = decode(args.raw, args.kernel)
    write_csv(rows, output, args.kernel)
    if args.trace is not None:
        from .trace import write_trace

        trace_path = args.trace if str(args.trace) else args.raw.with_name(args.raw.stem + "_trace.json")
        info = write_trace(
            rows,
            trace_path,
            args.kernel,
            rank=args.rank,
            include_wait=not args.no_wait,
            include_kernel=args.kernel_bar,
        )
        print(
            f"[TIMER] trace -> {info['path']} "
            f"(slices={info['events']}, lanes={info['lanes']}, envelope={info['envelope_us']}us)"
        )
    if args.summary or args.json:
        summary = summarize(rows)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"{'engine':<6}{'stage':<32}{'max_us':>12}")
            for item in summary:
                print(f"{item['engine']:<6}{item['stage']:<32}{item['max_us']:>12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
