// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
//
// Backward-only fixed timing slots.
//
// The backward pipeline needs whole-stage bars that the forward schema has no
// name for (route gather, count-table restore, dX Combine tail, ...) plus three
// per-expert wait families that are too fine-grained for the shared dynamic
// slots.  They live in their own enum and reuse the same per-core item space:
// N_TIMING_ITEM_PER_CORE covers the forward slots plus these, which is why the
// static_assert below pins the width.
//
// Retired slots keep their numeric value so existing raw dumps and parsers stay
// valid; they simply stay zero and disappear from traces.
//
// Ported from ascend_deepep_hkq (7eefe6e^).

#ifndef DEEPEP_MEGAMOE_TRAIN_BF16_BACKWARD_TIMER_H
#define DEEPEP_MEGAMOE_TRAIN_BF16_BACKWARD_TIMER_H

#include "ascend_timer_v2.hpp"

namespace MegaMoeBF16BackwardTimer {

enum BackwardFixedTiming {
    BW_KERNEL_TIMING_IDX = 0,  // whole kernel (timerAll_)
    AIV_GATHER = 1,            // P0-A gather grad_y -> peermem ptrA0 (all AIV)
    AIV_XSYNC_GATHER = 2,      // cross-rank barrier after the gather
    AIV_RESTORE_TABLE = 3,     // P0-B restore the saved token-count table + SyncAll
    // Slot 4 was AIC_SPIN_AIV1 (retired). It now accumulates the pure MMAD
    // cycles of the fused P1 dHp GEMM, so a stage span can be split into
    // "cube working" and "waiting for the vector epilogue" without adding a
    // barrier to the ping-pong pipeline.
    AIC_MMAD_BUSY_GMM2GRAD = 4,
    // Slot 6 was AIC_SPIN_AIV0 and slot 15 was AIV_P0_TAIL; both are still
    // retired: the static P0 staging region removed the P0->P1 UB ownership
    // handover, so there is no whole-stage spin or pipe-teardown tail any more.
    AIC_SPIN_DZ = 5,           // AIC waits the last dZ tiles + both AIVs done (P1 exit)
    AIC_SPIN_HP = 7,           // AIC waits Hp/dpProducts published (P4 entry)
    AIV_XSYNC_UNPERMUTE = 8,   // cross-rank barrier before P5 reads ptrD
    AIV_UNPERMUTE = 9,         // P5 grad_x = sum over topk slots of ptrD
    AIV_XSYNC_DP = 10,         // cross-rank barrier: ptrD readers drained
    AIV_DP_REDUCE = 11,        // P6a dp = sum_F(dHp*H), routed to source ranks
    AIV_XSYNC_TOPK = 12,       // cross-rank barrier: remote dp writes visible
    AIV_TOPK_UNPERMUTE = 13,   // P6b grad_topk from the routed dp
    AIV_TABLE_RESET = 14,      // re-zero the symmetric token table
    AIV_RECOMPUTE_PASS = 16,   // AIV bookkeeping walk of the recompute phase
    // Accumulated WaitForCube time of every P1 tile this AIV lane consumed
    // (Accumulate mode: end-start is the sum of waits, so AIV_SWIGLU_GRAD_e
    // splits into this wait plus epilogue compute).
    AIV_P1_WAIT = 17,          // P1 per-lane accumulated dHp-tile wait
    AIC_P2_TAIL = 18,          // AIC waits until AIV0 consumed every dX tile (P2 exit)
    AIC_INIT = 19,             // AIC wakeup-to-first-stage window
    AIV_INIT = 20,             // AIV wakeup-to-first-stage window
    AIC_DZ_WAIT_EXPERT_BASE = 21,        // + expert: AIC waits this expert's dZ/Hp
    AIV_DX_WAIT_EXPERT_BASE = 85,        // + expert: AIV0 waits this expert's dX tiles
    AIC_DO_REUSE_WAIT_EXPERT_BASE = 149, // + expert: AIC waits dW2 readers released dO
    BW_BACKWARD_TIMING_COUNT = AIC_DO_REUSE_WAIT_EXPERT_BASE + AscendTimer::MAX_DYNAMIC_ITER
};

inline constexpr int kMaxExperts = AscendTimer::MAX_DYNAMIC_ITER;
static_assert(AIV_DX_WAIT_EXPERT_BASE - AIC_DZ_WAIT_EXPERT_BASE == kMaxExperts,
              "expert wait slots must match the timer layout");
static_assert(AIC_DO_REUSE_WAIT_EXPERT_BASE - AIV_DX_WAIT_EXPERT_BASE == kMaxExperts,
              "expert wait slots must match the timer layout");
static_assert(BW_BACKWARD_TIMING_COUNT <= AscendTimer::N_TIMING_ITEM_PER_CORE,
              "backward timing slots must fit in the AscendTimerV2 per-core item space");

}  // namespace MegaMoeBF16BackwardTimer

#endif  // DEEPEP_MEGAMOE_TRAIN_BF16_BACKWARD_TIMER_H
