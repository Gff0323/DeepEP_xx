// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
//
// MegaMoE BF16 training timer protocol.
//
// Ported from the training operator's kernel timer (ascend_deepep_hkq,
// commit 7eefe6e^ "cleanup: remove kernel timer instrumentation"), which this
// tree inherited in a refactored layout.  The on-device layout is unchanged:
// every logical core owns a fixed block of [start, end] cycle pairs, followed
// by a per-(core, dynamic type) iteration-count table.
//
// This header is host-safe: the Host extension and the device kernels must
// agree on TOTAL_BUFFER_SIZE, so it depends only on <stdint.h>.

#ifndef DEEPEP_MEGAMOE_TRAIN_ASCEND_TIMER_V2_H
#define DEEPEP_MEGAMOE_TRAIN_ASCEND_TIMER_V2_H

#include <stdint.h>

namespace AscendTimer {

// Whole-invocation slots.  Every core records its own view of the same stage;
// the Host only uses them to place per-expert dynamic work on a timeline.
enum FixedTiming {
    KERNEL_TIMING_IDX = 0,
    AIV_INIT_PARAM,
    AIV_INITROUTING_SORT,
    AIV_INITROUTING_EXPERT_TOKEN_COUNT,
    AIV_INITROUTING_GATHEROUT_QUANT,
    AIV_ALLGATHER,
    AIV_WAIT_COMB,
    AIV_UNPERMUTE_WAIT,
    AIV_UNPERMUTE,
    AIC_WAIT_ALLGATHER,
    FIXED_TIMING_COUNT
};

// Per-expert slots shared by both kernels.  The dynamic iteration index is the
// expert ordinal, so AIV_SWIGLU_3 is expert 3's SwiGLU epilogue.
enum DynamicTimingType {
    AIV_DISPATCH = 0,
    AIV_SWIGLU_WAIT,
    AIV_SWIGLU,
    AIV_COMBINE_WAIT,
    AIV_COMBINE,
    AIC_GMM1_WAIT,
    AIC_GMM1,
    AIC_GMM2_WAIT,
    AIC_GMM2,
    AIV_GMM1_PROLOGUE,
    AIV_GMM2_PROLOGUE,
    // Backward per-expert stages (the dynamic iteration is expertIdx).  The
    // BW_ infix avoids colliding with the forward entries above; the parsers
    // display them without it (AIC_GMM2GRAD_0..N etc.).
    AIV_BW_DISPATCH,
    AIC_BW_RECOMPUTE,
    AIC_BW_DO_WAIT,
    AIC_BW_GMM2GRAD,
    AIV_BW_SWIGLU_GRAD,
    AIC_BW_GMM1GRAD,
    AIV_BW_COMBINE,
    AIC_BW_DW1,
    AIC_BW_DW2,
    DYNAMIC_TYPE_COUNT_ENUM
};

// 64 = the per-rank expert capacity the backward pipeline already assumes
// (its per-expert wait slots are reserved for 64 experts).  Raise it in both
// this header and the Python protocol when a build needs more experts.
static constexpr int MAX_DYNAMIC_ITER = 64;
static constexpr int N_CORE_COUNT = 96;
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;
static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;
static constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT;
static constexpr int TOTAL_BUFFER_SIZE = N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE;

// GetSystemCycle() ticks at 1 GHz on this part: cycles / 1000 = microseconds.
static constexpr double CYCLE_PER_MICROSECOND = 1000.0;

}  // namespace AscendTimer

// Wraps statements that must disappear when the timer is compiled out.
#define TIMER_BLOCK(code) \
    do {                  \
        code;             \
    } while (0)

#endif  // DEEPEP_MEGAMOE_TRAIN_ASCEND_TIMER_V2_H
