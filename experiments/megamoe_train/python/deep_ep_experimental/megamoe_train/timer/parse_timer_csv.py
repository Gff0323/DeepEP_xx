#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026
"""Parse MegaMoe BF16 forward AscendTimerV2 raw per-rank dumps.

Ported from the training operator's ``test/megamoe_bf16_test/parse_timer_csv.py``
(ascend_deepep_hkq). Behaviour is unchanged: one row per logical core, three
columns per slot (start cycle, end cycle, duration us), plus the per-type
iteration counters. Only two things differ:

* slot names come from ``protocol.py``, the mirror the C++ layout is checked
  against, instead of a second hand-written copy;
* ``load_raw`` also accepts the flat ``int64`` ``.bin`` dump this repo writes,
  not just a CSV of integers.

Usage:
    python -m deep_ep_experimental.megamoe_train.timer forward \
        --dir <timer_dir> --rank-start 0 --rank-end 0
"""

import argparse
import csv
import sys
from array import array
from pathlib import Path

from . import protocol as p

FIXED_NAMES = p.FIXED_NAMES
DYNAMIC_NAMES = p.DYNAMIC_NAMES
MAX_DYNAMIC_ITER = p.MAX_DYNAMIC_ITER
CORE_COUNT = p.N_CORE_COUNT
COUNTERS_PER_CORE = p.COUNTERS_PER_CORE
DYNAMIC_ITER_PER_CORE = p.DYNAMIC_ITER_PER_CORE
BASE_COUNTERS = p.N_TIMING_COUNTER
TOTAL_SIZE = p.TOTAL_BUFFER_SIZE
CYCLE_TO_US = p.CYCLE_TO_US


def load_raw(path):
    """Read a flat int64 ``.bin`` dump, or the legacy CSV-of-integers form."""
    path = Path(path)
    values = array("q")
    if path.suffix == ".bin":
        if path.stat().st_size != TOTAL_SIZE * 8:
            raise RuntimeError(
                f"{path} has {path.stat().st_size} bytes; expected {TOTAL_SIZE * 8}"
            )
        with path.open("rb") as stream:
            values.fromfile(stream, TOTAL_SIZE)
        if sys.byteorder != "little":
            values.byteswap()
    else:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                values.extend(int(item) for item in row if item.strip())
    if len(values) != TOTAL_SIZE:
        raise RuntimeError(f"{path} contains {len(values)} values; expected {TOTAL_SIZE}")
    return values


def core_base(core_id):
    group_id, subgroup = divmod(core_id, 3)
    return group_id * COUNTERS_PER_CORE * 3 + subgroup * COUNTERS_PER_CORE


def event_pair(values, core_id, logical_index):
    index = core_base(core_id) + logical_index * 2
    start, end = values[index], values[index + 1]
    duration = (end - start) / CYCLE_TO_US if end > start else 0.0
    return start, end, duration


def parse_rank(raw_path, parsed_path):
    values = load_raw(raw_path)
    dynamic_counts = values[BASE_COUNTERS:]
    max_iters = [
        max(
            dynamic_counts[core_id * DYNAMIC_ITER_PER_CORE + type_index]
            for core_id in range(CORE_COUNT)
        )
        for type_index in range(len(DYNAMIC_NAMES))
    ]

    events = [(name, index) for index, name in enumerate(FIXED_NAMES)]
    for type_index, name in enumerate(DYNAMIC_NAMES):
        for iteration in range(min(max_iters[type_index], MAX_DYNAMIC_ITER)):
            logical_index = (
                len(FIXED_NAMES) + type_index * MAX_DYNAMIC_ITER + iteration
            )
            events.append((f"{name}_{iteration}", logical_index))

    with parsed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["core_id", "group_id", "core_type", "sub_id"]
        for name, _ in events:
            header.extend(
                [
                    f"{name}_start_cycle",
                    f"{name}_end_cycle",
                    f"{name}_duration(us)",
                ]
            )
        header.extend(f"{name}_IterCount" for name in DYNAMIC_NAMES)
        writer.writerow(header)

        for core_id in range(CORE_COUNT):
            group_id, subgroup = divmod(core_id, 3)
            core_type = "AIC" if subgroup == 0 else "AIV"
            sub_id = 0 if subgroup == 0 else subgroup - 1
            row = [core_id, group_id, core_type, sub_id]
            for _, logical_index in events:
                start, end, duration = event_pair(values, core_id, logical_index)
                row.extend([start, end, f"{duration:.2f}"])
            dynamic_base = core_id * DYNAMIC_ITER_PER_CORE
            row.extend(
                dynamic_counts[dynamic_base + index]
                for index in range(len(DYNAMIC_NAMES))
            )
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, required=True)
    parser.add_argument(
        "--raw-pattern",
        default="rank_{rank}_megamoe_bf16_forward_timer.bin",
        help="raw dump name; {rank} is substituted",
    )
    parser.add_argument(
        "--parsed-pattern",
        default="rank_{rank}_megamoe_bf16_forward_timer_parsed.csv",
        help="parsed CSV name; {rank} is substituted",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.dir / "timer"
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.rank_start, args.rank_end + 1):
        raw_path = args.dir / args.raw_pattern.format(rank=rank)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        parsed_path = output_dir / args.parsed_pattern.format(rank=rank)
        parse_rank(raw_path, parsed_path)
        print(f"[TIMER] parsed {raw_path} -> {parsed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
