# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""Turn a decoded stage table into a Chrome/Perfetto trace.

``ui.perfetto.dev`` (and ``chrome://tracing``) read trace JSON, not CSV, so the
decoder needed this step: every recorded span becomes one ``ph: "X"`` slice on
the lane of the core that recorded it, which is what renders as a Gantt
pipeline. The layout mirrors the training operator's earlier bigop tooling
(``parse_timer_traces.py`` / ``trace_parser.py``): ``pid`` is the rank, ``tid``
is ``group_<g>_<AIC|AIV><sub>``, ``ts``/``dur`` are microseconds.

Timestamps are absolute ``GetSystemCycle()`` values minus the earliest start in
the dump. The counter is system-wide, so slices from different cores of one rank
share a time base and line up as the real pipeline; no interpolation is done.
"""

import json
from pathlib import Path

from .protocol import CYCLE_TO_US


def _lane(row):
    """One track per logical core: ``group_<g>_<AIC|AIV0|AIV1>``."""
    return f"group_{row['group']}_{row['engine']}"


def _category(name):
    if "WAIT" in name or "SYNC" in name or "SPIN" in name:
        return "dependency_wait"
    if name.startswith("AIC") or name.startswith("BW_KERNEL") or "GMM" in name or "DW" in name:
        return "aic_op"
    return "aiv_op"


def collect_slices(rows, *, include_wait=True, include_kernel=False):
    """Flatten decoded rows into (start, end, lane, engine, name) spans."""
    spans = []
    for row in rows:
        lane = _lane(row)
        for name, (start, end, _duration) in row["values"].items():
            if end <= start or start <= 0:
                continue
            if not include_kernel and name.endswith("KERNEL_TIMING"):
                continue
            if not include_wait and "WAIT" in name:
                continue
            spans.append((start, end, lane, row["engine"], name))
    return spans


def write_trace(rows, path, kernel, *, rank=0, include_wait=True, include_kernel=False):
    """Write one Perfetto/Chrome trace JSON for a single rank's dump."""
    spans = collect_slices(rows, include_wait=include_wait, include_kernel=include_kernel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not spans:
        raise ValueError(
            "no non-zero spans in this dump: the kernel was probably built without "
            "DEEPEP_MEGAMOE_TRAIN_TIMER=ON, or forward and backward were not separated"
        )
    base = min(start for start, _end, _lane_name, _engine, _name in spans)
    lanes = sorted({lane for _s, _e, lane, _engine, _name in spans})
    pid = f"rank{rank}"
    events = [
        {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": f"MegaMoE BF16 {kernel} rank {rank}"}},
        {"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": rank}},
    ]
    for index, lane in enumerate(lanes):
        events.append(
            {"ph": "M", "name": "thread_name", "pid": pid, "tid": lane, "args": {"name": lane}}
        )
        events.append(
            {"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": lane, "args": {"sort_index": index}}
        )
    for start, end, lane, engine, name in sorted(spans):
        events.append(
            {
                "ph": "X",
                "cat": _category(name),
                "pid": pid,
                "tid": lane,
                "name": name,
                "ts": round((start - base) / CYCLE_TO_US, 6),
                "dur": round((end - start) / CYCLE_TO_US, 6),
                "args": {
                    "rank": rank,
                    "kernel": kernel,
                    "engine": engine,
                    "start_cycle": start,
                    "end_cycle": end,
                },
            }
        )
    path.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    total = max((end for _s, end, _l, _e, _n in spans), default=base)
    return {
        "path": str(path),
        "events": len(events) - 2 - 2 * len(lanes),
        "lanes": len(lanes),
        "envelope_us": round((total - base) / CYCLE_TO_US, 3),
    }
