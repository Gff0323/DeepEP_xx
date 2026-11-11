# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""Host mirror of the MegaMoe BF16 training timer layout.

Keep every constant here in lockstep with
``csrc/kernels/bf16/common/ascend_timer_v2.hpp`` and
``csrc/kernels/bf16/common/mega_moe_bf16_backward_timer.hpp``. The device
buffer is a flat ``int64`` array; everything below is an index computation, so
a mismatch shows up as a size error (``--check-schema``) instead of silently
shifted columns.
"""

# Whole-invocation slots: all cores record their own view of the same stage.
FIXED_NAMES = [
    "KERNEL_TIMING",
    "AIV_INIT_PARAM",
    "AIV_INITROUTING_SORT",
    "AIV_INITROUTING_EXPERT_TOKEN_COUNT",
    "AIV_INITROUTING_GATHEROUT_QUANT",
    "AIV_ALLGATHER",
    "AIV_WAIT_COMB",
    "AIV_UNPERMUTE_WAIT",
    "AIV_UNPERMUTE",
    "AIC_WAIT_ALLGATHER",
]

# Forward per-expert slots, then the backward ones that share the same space.
FORWARD_DYNAMIC_NAMES = [
    "AIV_DISPATCH",
    "AIV_SWIGLU_WAIT",
    "AIV_SWIGLU",
    "AIV_COMBINE_WAIT",
    "AIV_COMBINE",
    "AIC_GMM1_WAIT",
    "AIC_GMM1",
    "AIC_GMM2_WAIT",
    "AIC_GMM2",
    "AIV_GMM1_PROLOGUE",
    "AIV_GMM2_PROLOGUE",
]
BACKWARD_DYNAMIC_NAMES = [
    "AIV_DISPATCH",
    "AIC_RECOMPUTE",
    "AIC_DO_WAIT",
    "AIC_GMM2GRAD",
    "AIV_SWIGLU_GRAD",
    "AIC_GMM1GRAD",
    "AIV_COMBINE",
    "AIC_DW1",
    "AIC_DW2",
]
DYNAMIC_NAMES = FORWARD_DYNAMIC_NAMES + [
    "AIV_BW_DISPATCH",
    "AIC_BW_RECOMPUTE",
    "AIC_BW_DO_WAIT",
    "AIC_BW_GMM2GRAD",
    "AIV_BW_SWIGLU_GRAD",
    "AIC_BW_GMM1GRAD",
    "AIV_BW_COMBINE",
    "AIC_BW_DW1",
    "AIC_BW_DW2",
]
# Index of the first backward entry inside DYNAMIC_NAMES.
BACKWARD_DYNAMIC_BASE = len(FORWARD_DYNAMIC_NAMES)
FORWARD_DYNAMIC_COUNT = len(FORWARD_DYNAMIC_NAMES)

# Backward fixed slots, including the three per-expert wait families.
BACKWARD_FIXED_NAMES = [
    "BW_KERNEL_TIMING",
    "AIV_GATHER",
    "AIV_XSYNC_GATHER",
    "AIV_RESTORE_TABLE",
    "AIC_SPIN_AIV1_RETIRED",
    "AIC_SPIN_DZ",
    "AIC_SPIN_AIV0_RETIRED",
    "AIC_SPIN_HP",
    "AIV_XSYNC_UNPERMUTE",
    "AIV_UNPERMUTE",
    "AIV_XSYNC_DP",
    "AIV_DP_REDUCE",
    "AIV_XSYNC_TOPK",
    "AIV_TOPK_UNPERMUTE",
    "AIV_TABLE_RESET",
    "AIV_P0_TAIL_RETIRED",
    "AIV_RECOMPUTE_PASS",
    "AIV_P1_WAIT",
    "AIC_P2_TAIL",
    "AIC_INIT",
    "AIV_INIT",
]
MAX_EXPERTS = 64
BACKWARD_EXPERT_WAIT_BASE = len(BACKWARD_FIXED_NAMES)
BACKWARD_FIXED_NAMES = (
    BACKWARD_FIXED_NAMES
    + [f"AIC_DZ_WAIT_{expert}" for expert in range(MAX_EXPERTS)]
    + [f"AIV_DX_WAIT_{expert}" for expert in range(MAX_EXPERTS)]
    + [f"AIC_DO_REUSE_WAIT_{expert}" for expert in range(MAX_EXPERTS)]
)

MAX_DYNAMIC_ITER = 64
N_CORE_COUNT = 96
CORE_NAMES = ("AIC", "AIV0", "AIV1")
CYCLE_TO_US = 1000.0

N_TIMING_ITEM_PER_CORE = len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER
COUNTERS_PER_CORE = (N_TIMING_ITEM_PER_CORE * 2 + 15) // 16 * 16
N_TIMING_COUNTER = COUNTERS_PER_CORE * N_CORE_COUNT
DYNAMIC_ITER_PER_CORE = (len(DYNAMIC_NAMES) + 15) // 16 * 16
TOTAL_BUFFER_SIZE = N_TIMING_COUNTER + DYNAMIC_ITER_PER_CORE * N_CORE_COUNT


def core_lane(core_id):
    """Logical core -> (group, engine, sub-block) using the published layout."""
    group, lane = divmod(core_id, 3)
    return group, CORE_NAMES[lane], lane


def counter_index(core_id, logical_index):
    return core_id * COUNTERS_PER_CORE + logical_index * 2


def dynamic_count_index(core_id, type_index):
    return N_TIMING_COUNTER + core_id * DYNAMIC_ITER_PER_CORE + type_index


def dynamic_logical_index(type_index, iteration):
    return len(FIXED_NAMES) + type_index * MAX_DYNAMIC_ITER + iteration


def timer_buffer_bytes():
    return TOTAL_BUFFER_SIZE * 8


def check_cpp_schema(root=None):
    """Compare this mirror against the device headers, when they are present.

    A silent drift between the C++ layout and this file would show up as
    plausible-looking garbage columns, so the check compares the numbers that
    define the layout rather than trusting a hand-maintained copy.
    """
    import json
    import re
    from pathlib import Path

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
    max_iter = int(re.search(r"MAX_DYNAMIC_ITER\s*=\s*(\d+)", shared_text).group(1))
    cores = int(re.search(r"N_CORE_COUNT\s*=\s*(\d+)", shared_text).group(1))
    bases = [
        int(re.search(rf"{name}\s*=\s*(\d+)", backward_text).group(1))
        for name in ("AIC_DZ_WAIT_EXPERT_BASE", "AIV_DX_WAIT_EXPERT_BASE", "AIC_DO_REUSE_WAIT_EXPERT_BASE")
    ]

    problems = []
    if len(fixed_entries) != len(FIXED_NAMES):
        problems.append(f"fixed slots: C++ {len(fixed_entries)} vs Python {len(FIXED_NAMES)}")
    if len(dynamic_entries) != len(DYNAMIC_NAMES):
        problems.append(f"dynamic slots: C++ {len(dynamic_entries)} vs Python {len(DYNAMIC_NAMES)}")
    if max_iter != MAX_DYNAMIC_ITER:
        problems.append(f"MAX_DYNAMIC_ITER: C++ {max_iter} vs Python {MAX_DYNAMIC_ITER}")
    if cores != N_CORE_COUNT:
        problems.append(f"N_CORE_COUNT: C++ {cores} vs Python {N_CORE_COUNT}")
    expected_bases = [
        len(BACKWARD_FIXED_NAMES) - 3 * MAX_EXPERTS,
        len(BACKWARD_FIXED_NAMES) - 2 * MAX_EXPERTS,
        len(BACKWARD_FIXED_NAMES) - MAX_EXPERTS,
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
        "total_buffer_size": TOTAL_BUFFER_SIZE,
    }
