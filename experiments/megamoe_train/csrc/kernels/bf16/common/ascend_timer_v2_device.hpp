// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
//
// Device side of the MegaMoE BF16 training timer.
//
// Ported unchanged in behaviour from ascend_deepep_hkq (7eefe6e^).  Only the
// qualification style follows this tree (AscendC::).  Include it through
// mega_moe_bf16_timer.hpp so that a build without the timer switch keeps the
// object, the cycle reads, the pipe barriers and the GM writes out of the
// generated kernel entirely.

#ifndef DEEPEP_MEGAMOE_TRAIN_ASCEND_TIMER_V2_DEVICE_H
#define DEEPEP_MEGAMOE_TRAIN_ASCEND_TIMER_V2_DEVICE_H

#include <type_traits>

#include "ascend_timer_v2.hpp"

// Write modes for AscendTimerDevice::Tok.
struct Accumulate {};  // add this span to whatever the slot already holds
struct Overwrite {};   // replace the slot (default)

class AscendTimerDevice {
public:
    __aicore__ inline AscendTimerDevice() = default;

    // Maps the physical block to the published [AIC, AIV0, AIV1] layout: the
    // trace always presents three logical slots per AIC group regardless of
    // which engine reads it, so AIC block 1 is slot 3 and never slot 2.
    __aicore__ inline void Init(GM_ADDR time_addr)
    {
        time_addr_ = time_addr;
        block_id_ = AscendC::GetBlockIdx();
        group_id_ = block_id_;
        core_id_ = block_id_;
        subblock_id_ = 0;
        if ASCEND_IS_AIV {
            const int task_ratio = AscendC::GetTaskRation();
            group_id_ = block_id_ / task_ratio;
            subblock_id_ = AscendC::GetSubBlockIdx();
            core_id_ = group_id_ * 3 + subblock_id_ + 1;
        } else {
            core_id_ = group_id_ * 3;
        }
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles_[i] = 0;
            dynamic_start_iters_[i] = -1;
            dynamic_valid_[i] = false;
        }
    }

    // Hosts and parsers read this table to know how many expert slots exist;
    // write it at Init so an untouched slot cannot masquerade as a task.
    __aicore__ inline void setDynamicActualIter(AscendTimer::DynamicTimingType type, int64_t iter)
    {
        if (time_addr_ == nullptr || core_id_ < 0 || core_id_ >= AscendTimer::N_CORE_COUNT ||
            static_cast<int>(type) < 0 || static_cast<int>(type) >= AscendTimer::DYNAMIC_TYPE_COUNT) {
            return;
        }
        const int32_t index =
            AscendTimer::N_TIMING_COUNTER + core_id_ * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN + static_cast<int>(type);
        AscendC::GlobalTensor<int64_t> timer;
        timer.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(time_addr_));
        timer(index) = iter;
    }

    // Opens a span closed by the overload below (fixed slots and nested waits).
    __aicore__ inline void Tik()
    {
        if (time_addr_ == nullptr) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        start_cycle_ = AscendC::GetSystemCycle();
    }

    // Opens a per-expert span; the matching Tok is keyed by (type, iter) so an
    // abandoned span (early continue) can never close a later expert's slot.
    __aicore__ inline void Tik(AscendTimer::DynamicTimingType type, int iter)
    {
        if (time_addr_ == nullptr || static_cast<int>(type) < 0 ||
            static_cast<int>(type) >= AscendTimer::DYNAMIC_TYPE_COUNT || iter < 0 ||
            iter >= AscendTimer::MAX_DYNAMIC_ITER) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        const int type_index = static_cast<int>(type);
        dynamic_start_cycles_[type_index] = AscendC::GetSystemCycle();
        dynamic_start_iters_[type_index] = iter;
        dynamic_valid_[type_index] = true;
        WriteStart(AscendTimer::FIXED_TIMING_COUNT + type_index * AscendTimer::MAX_DYNAMIC_ITER + iter,
                   dynamic_start_cycles_[type_index]);
    }

    template <typename ModeTag = Overwrite>
    __aicore__ inline void Tok(int32_t timing_index)
    {
        if (time_addr_ == nullptr || timing_index < 0 || timing_index >= AscendTimer::N_TIMING_ITEM_PER_CORE) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        const int64_t end_cycle = AscendC::GetSystemCycle();
        WriteTime<ModeTag>(timing_index, start_cycle_, end_cycle);
    }

    template <typename ModeTag = Overwrite>
    __aicore__ inline void Tok()
    {
        Tok<ModeTag>(AscendTimer::KERNEL_TIMING_IDX);
    }

    template <typename ModeTag = Overwrite>
    __aicore__ inline void Tok(AscendTimer::DynamicTimingType type, int iter)
    {
        const int type_index = static_cast<int>(type);
        if (time_addr_ == nullptr || type_index < 0 || type_index >= AscendTimer::DYNAMIC_TYPE_COUNT || iter < 0 ||
            iter >= AscendTimer::MAX_DYNAMIC_ITER || !dynamic_valid_[type_index] ||
            dynamic_start_iters_[type_index] != iter) {
            return;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        const int64_t end_cycle = AscendC::GetSystemCycle();
        const int timing_index = AscendTimer::FIXED_TIMING_COUNT + type_index * AscendTimer::MAX_DYNAMIC_ITER + iter;
        WriteTime<ModeTag>(timing_index, dynamic_start_cycles_[type_index], end_cycle);
        dynamic_valid_[type_index] = false;
        dynamic_start_iters_[type_index] = -1;
    }

private:
    __aicore__ inline int32_t CalculateIndex(int32_t timing_index) const
    {
        if ASCEND_IS_AIV {
            return group_id_ * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3 +
                   AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * (1 + subblock_id_) + timing_index * 2;
        }
        return group_id_ * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3 + timing_index * 2;
    }

    __aicore__ inline void WriteStart(int32_t timing_index, int64_t start)
    {
        const int32_t index = CalculateIndex(timing_index);
        if (index < 0 || index + 1 >= AscendTimer::N_TIMING_COUNTER) {
            return;
        }
        AscendC::GlobalTensor<int64_t> timer;
        timer.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(time_addr_));
        timer(index) = start;
    }

    template <typename ModeTag>
    __aicore__ inline void WriteTime(int32_t timing_index, int64_t start, int64_t end)
    {
        const int32_t index = CalculateIndex(timing_index);
        if (index < 0 || index + 1 >= AscendTimer::N_TIMING_COUNTER) {
            return;
        }
        AscendC::GlobalTensor<int64_t> timer;
        timer.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(time_addr_));
        if constexpr (std::is_same_v<ModeTag, Accumulate>) {
            const int64_t previous_start = timer(index);
            const int64_t previous_end = timer(index + 1);
            const int64_t elapsed = end - start;
            if (previous_start == 0 && previous_end == 0) {
                timer(index) = start;
                timer(index + 1) = end;
            } else {
                start = previous_start;
                end = previous_end + elapsed;
            }
        }
        timer(index) = start;
        timer(index + 1) = end;
    }

    GM_ADDR time_addr_ = nullptr;
    int64_t start_cycle_ = 0;
    int64_t dynamic_start_cycles_[AscendTimer::DYNAMIC_TYPE_COUNT]{};
    int dynamic_start_iters_[AscendTimer::DYNAMIC_TYPE_COUNT]{};
    bool dynamic_valid_[AscendTimer::DYNAMIC_TYPE_COUNT]{};
    int block_id_ = 0;
    int group_id_ = 0;
    int subblock_id_ = 0;
    int core_id_ = 0;
};

#endif  // DEEPEP_MEGAMOE_TRAIN_ASCEND_TIMER_V2_DEVICE_H
