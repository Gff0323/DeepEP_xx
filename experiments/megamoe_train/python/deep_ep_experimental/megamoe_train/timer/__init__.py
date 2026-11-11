# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""Offline decoding of MegaMoe BF16 training device timer dumps.

Pure Python on purpose: decoding a dump must not need Torch, an NPU, or the
extension. The pipeline is the training operator's own tooling:

    raw dump (.bin / CSV of int64)
      -> parse_timer_csv.parse_rank        (forward)   -> parsed CSV
      -> parse_backward_timer.parse_rank   (backward)  -> parsed CSV + tables
      -> trace_parser.generate_trace_files -> Perfetto trace JSON

``protocol.py`` mirrors ``csrc/kernels/bf16/common/ascend_timer_v2.hpp`` and
``mega_moe_bf16_backward_timer.hpp``; ``--check-schema`` compares the two.
"""

from .protocol import (
    BACKWARD_DYNAMIC_BASE,
    BACKWARD_FIXED_NAMES,
    CYCLE_TO_US,
    DYNAMIC_ITER_PER_CORE,
    DYNAMIC_NAMES,
    FIXED_NAMES,
    FORWARD_DYNAMIC_COUNT,
    N_CORE_COUNT,
    N_TIMING_COUNTER,
    TOTAL_BUFFER_SIZE,
    check_cpp_schema,
)

__all__ = [
    "BACKWARD_DYNAMIC_BASE",
    "BACKWARD_FIXED_NAMES",
    "CYCLE_TO_US",
    "DYNAMIC_ITER_PER_CORE",
    "DYNAMIC_NAMES",
    "FIXED_NAMES",
    "FORWARD_DYNAMIC_COUNT",
    "N_CORE_COUNT",
    "N_TIMING_COUNTER",
    "TOTAL_BUFFER_SIZE",
    "check_cpp_schema",
]
