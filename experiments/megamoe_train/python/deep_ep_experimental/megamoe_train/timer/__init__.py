# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""Offline decoder for MegaMoe BF16 training device timer buffers.

Pure Python on purpose: decoding a dump must not need Torch, an NPU, or the
extension. The C++ side of this schema is
``c/src/kernels/bf16/common/ascend_timer_v2.hpp`` (shared slots) and
``mega_moe_bf16_backward_timer.hpp`` (backward-only slots).
"""

from .protocol import (
    BACKWARD_FIXED_NAMES,
    CYCLE_TO_US,
    DYNAMIC_ITER_PER_CORE,
    DYNAMIC_NAMES,
    FIXED_NAMES,
    FORWARD_DYNAMIC_COUNT,
    N_CORE_COUNT,
    N_TIMING_COUNTER,
    TOTAL_BUFFER_SIZE,
)
from .parse_timer_csv import decode, load_raw, write_csv

__all__ = [
    "BACKWARD_FIXED_NAMES",
    "CYCLE_TO_US",
    "DYNAMIC_ITER_PER_CORE",
    "DYNAMIC_NAMES",
    "FIXED_NAMES",
    "FORWARD_DYNAMIC_COUNT",
    "N_CORE_COUNT",
    "N_TIMING_COUNTER",
    "TOTAL_BUFFER_SIZE",
    "decode",
    "load_raw",
    "write_csv",
]
