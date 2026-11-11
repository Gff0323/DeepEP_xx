#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026
"""Generate Chrome/Perfetto trace JSON from a parsed MegaMoe BF16 timer CSV.

Ported from the training operator's ``test/megamoe_bf16_test/trace_parser.py``
(ascend_deepep_hkq). Two phases are supported, selected with ``--phase``:

  forward (default): reads parse_timer_csv.py output, renders the AIC*/AIV*
      fixed slots, and omits WAIT events (they stay in the parsed CSV for
      diagnostics).

  backward: reads parse_backward_timer.py output, renders the AIC_*/AIV_*
      fixed slots plus the per-expert blocks (AIC_GMM2GRAD_0..N etc.), and KEEPS
      the wait bars (AIC_SPIN_* / AIV_XSYNC_*): those sync points are the exact
      stalls this view is meant to expose, so dropping them would hide the
      problem.  AIV_P1_WAIT (accumulated per-lane dHp-tile wait) nests under the
      AIV_SWIGLU_GRAD_* bars like AIC_DO_WAIT_* under AIC_GMM2GRAD_*: Perfetto
      auto-stacks the overlap onto a second row.

Both emit one file per rank plus a merged all-rank trace JSON, openable in
Perfetto (https://ui.perfetto.dev) or chrome://tracing with zoom / pan /
click-to-inspect.
"""

import argparse
import csv
import json
from pathlib import Path

CYCLE_TO_TIME_BASE = 1000.0

# Backward stages that are pure waits: AIC spins on a GM flag, and the
# cross-rank barrier windows inside CrossRankSyncInWorldSize().  They are
# kept in the trace and tagged cat=wait so the timeline highlights the sync
# stalls being investigated.
BACKWARD_WAIT_NAMES = {
    "AIC_SPIN_AIV1",
    "AIC_SPIN_DZ",
    "AIC_SPIN_AIV0",
    "AIC_SPIN_HP",
    "AIC_P2_TAIL",
    "AIV_P1_WAIT",
    "AIV_XSYNC_GATHER",
    "AIV_XSYNC_UNPERMUTE",
    "AIV_XSYNC_DP",
    "AIV_XSYNC_TOPK",
}


def read_csv_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def collect_event_names(fieldnames, phase):
    events = []
    fields = set(fieldnames)
    for column in fieldnames:
        if not column.endswith("_start_cycle"):
            continue
        name = column[: -len("_start_cycle")]
        if phase == "backward":
            # Backward writes its fixed slots and the per-expert dynamic
            # blocks (AIC_GMM2GRAD_0..N) into the shared layout.  The
            # whole-kernel span would flatten every other bar.
            if name in ("BW_KERNEL_TIMING", "KERNEL", "KERNEL_TIMING"):
                continue
            if not (name.startswith("AIC") or name.startswith("AIV")):
                continue
        else:
            if name in ("KERNEL_TIMING", "BW_KERNEL_TIMING"):
                continue
            # WAIT events remain in the parsed CSV for diagnostics, but omitting
            # them keeps the Perfetto timeline focused on actual work.
            if "WAIT" in name.upper():
                continue
            if not (name.startswith("AIC") or name.startswith("AIV")):
                continue
        if f"{name}_end_cycle" in fields:
            events.append(name)
    return events


def find_min_start(csv_path, phase):
    min_start = None
    fieldnames, rows = read_csv_rows(csv_path)
    for event in collect_event_names(fieldnames, phase):
        start_column = f"{event}_start_cycle"
        end_column = f"{event}_end_cycle"
        for row in rows:
            start = int(row[start_column])
            end = int(row[end_column])
            if start > 0 and end > start:
                min_start = start if min_start is None else min(min_start, start)
    return min_start if min_start is not None else 0


def generate_rank_events(csv_path, rank, phase):
    fieldnames, rows = read_csv_rows(csv_path)
    events = collect_event_names(fieldnames, phase)
    base_cycle = find_min_start(csv_path, phase)
    trace_events = []

    for row in rows:
        group_id = int(row["group_id"])
        core_type = row["core_type"]
        sub_id = int(row["sub_id"])
        for event in events:
            start = int(row[f"{event}_start_cycle"])
            end = int(row[f"{event}_end_cycle"])
            if start <= 0 or end <= start:
                continue
            is_wait = phase == "backward" and (event in BACKWARD_WAIT_NAMES or
                event.startswith("AIC_DZ_WAIT_") or event.startswith("AIV_DX_WAIT_") or
                event.startswith("AIC_DO_REUSE_WAIT_"))
            trace_events.append(
                {
                    "ph": "X",
                    "cat": (
                        "wait" if is_wait
                        else ("aic_op" if core_type == "AIC" else "aiv_op")
                    ),
                    "pid": f"rank{rank}",
                    "tid": f"group_{group_id}_{core_type}_{sub_id}",
                    "name": event,
                    "ts": round((start - base_cycle) / CYCLE_TO_TIME_BASE, 6),
                    "dur": round((end - start) / CYCLE_TO_TIME_BASE, 6),
                    "args": {
                        "rank": rank,
                        "group_id": group_id,
                        "core_type": core_type,
                        "sub_id": sub_id,
                        "start_cycle": start,
                        "end_cycle": end,
                        "is_wait": is_wait,
                    },
                }
            )
    return trace_events


def parsed_csv_path(timer_dir, rank, phase):
    """Repo dump name first, then the training operator's historical name."""
    timer_dir = Path(timer_dir)
    candidates = [
        timer_dir / f"rank_{rank}_megamoe_bf16_{phase}_timer_parsed.csv",
        timer_dir / (f"rank_{rank}_backward_time_parsed.csv" if phase == "backward"
                     else f"rank_{rank}_npu_time_parsed.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def generate_trace_files(timer_dir, rank_start, rank_end, output_dir=None,
                         merged_name=None, name_suffix="", phase="forward"):
    timer_dir = Path(timer_dir)
    output_dir = Path(output_dir) if output_dir else timer_dir / "timer_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_events = []

    for rank in range(rank_start, rank_end + 1):
        csv_path = parsed_csv_path(timer_dir, rank, phase)
        if not csv_path.exists():
            if phase == "backward":
                # Only profiled ranks have a dump; skip the gaps.
                continue
            raise FileNotFoundError(f"Timer CSV does not exist: {csv_path}")
        rank_events = generate_rank_events(csv_path, rank, phase)
        all_events.extend(rank_events)
        suffix = f"_{name_suffix}" if name_suffix else ""
        rank_output = output_dir / f"rank_{rank}_megamoe_bf16_{phase}{suffix}_trace.json"
        with rank_output.open("w", encoding="utf-8") as stream:
            json.dump(rank_events, stream, indent=2, ensure_ascii=False)
        print(f"[TIMER] rank={rank} events={len(rank_events)} JSON={rank_output}")

    if merged_name is None:
        suffix = f"_{name_suffix}" if name_suffix else ""
        merged_name = f"all_rank_{phase}{suffix}_trace.json"
    merged_output = output_dir / merged_name
    with merged_output.open("w", encoding="utf-8") as stream:
        json.dump(all_events, stream, indent=2, ensure_ascii=False)
    return merged_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--name-suffix", default="")
    parser.add_argument(
        "--phase",
        choices=("forward", "backward"),
        default="forward",
        help="forward (parse_timer_csv output) or backward (parse_backward_timer output)",
    )
    args = parser.parse_args(argv)
    merged_output = generate_trace_files(
        args.timer_dir,
        args.rank_start,
        args.rank_end,
        args.output_dir,
        args.output,
        args.name_suffix,
        args.phase,
    )
    print(f"[TIMER] Merged trace JSON: {merged_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
