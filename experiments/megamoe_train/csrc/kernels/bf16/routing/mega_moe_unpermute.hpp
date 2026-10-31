#pragma once

#include "kernel_operator.h"
#include "layout/mega_moe_unpermute_layout.hpp"

namespace MegaMoeBF16Routing {

__aicore__ inline void LoadUnpermuteInput(AscendC::LocalTensor<bfloat16_t> input,
                                          AscendC::GlobalTensor<bfloat16_t> expandedX, int32_t row, uint32_t hidden,
                                          uint32_t slot)
{
    using namespace AscendC;
    const auto event = slot == 0 ? EVENT_ID0 : EVENT_ID1;
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopy(input, expandedX[uint64_t{static_cast<uint32_t>(row)} * hidden], hidden);
    SetFlag<HardEvent::MTE2_V>(event);
}

// Called only after Combine completion and the existing cross-rank barrier.
// BF16, MGroup (through BF16) and UDMA share the same local ptrD layout.
// False leaves unsupported shapes to the caller's original three-row path.
__aicore__ inline bool TryUnpermute(GM_ADDR indicesAddr, GM_ADDR expandedXAddr, GM_ADDR outputAddr, uint32_t hidden,
                                    uint32_t topK, uint32_t tokenBegin, uint32_t tokenCount, uint32_t ubBytes)
{
    using namespace AscendC;
    const UnpermutePlan plan(hidden, topK, ubBytes);
    if (!plan.enabled) {
        return false;
    }
    if (tokenCount == 0) {
        return true;
    }

    TPipe pipe;
    TBuf<> rowsBuf;
    TBuf<> indicesBuf;
    pipe.InitBuffer(rowsBuf, 4 * plan.rowBytes);
    pipe.InitBuffer(indicesBuf, UnpermutePlan::kIndexBytes);
    auto rows = rowsBuf.Get<bfloat16_t>();
    auto indices = indicesBuf.Get<int32_t>();
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<bfloat16_t> expandedX;
    GlobalTensor<bfloat16_t> output;
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indicesAddr));
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(expandedXAddr));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(outputAddr));

    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    for (uint32_t offset = 0; offset < tokenCount;) {
        const uint32_t count = plan.BatchCount(tokenCount - offset);
        // Scalar must finish reading the old index batch before MTE2 reuses UB.
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(indices, indicesGm[(uint64_t{tokenBegin} + offset) * topK],
                    DataCopyExtParams{1U, count * topK * uint32_t{sizeof(int32_t)}, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        for (uint32_t token = 0; token < count; ++token) {
            const uint32_t outputSlot = (offset + token) % 2;
            const auto outputEvent = outputSlot == 0 ? EVENT_ID0 : EVENT_ID1;
            auto result = rows[outputSlot * hidden];
            const uint32_t indexBase = token * topK;

            // Prime two inputs before consuming either. The other input can
            // arrive while Vector consumes this one, including for odd TopK.
            LoadUnpermuteInput(rows[2 * hidden], expandedX, indices.GetValue(indexBase), hidden, 0);
            if (topK > 1) {
                LoadUnpermuteInput(rows[3 * hidden], expandedX, indices.GetValue(indexBase + 1), hidden, 1);
            }
            // Only reuse of this output slot waits for MTE3. Input loads do
            // not depend on the previous token's output transfer.
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            for (uint32_t expert = 0; expert < topK; ++expert) {
                const uint32_t inputSlot = expert % 2;
                const auto inputEvent = inputSlot == 0 ? EVENT_ID0 : EVENT_ID1;
                auto input = rows[(2 + inputSlot) * hidden];
                WaitFlag<HardEvent::MTE2_V>(inputEvent);
                if (expert == 0) {
                    Adds(result, input, 0.0F, hidden);
                } else {
                    Add(result, result, input, hidden);
                }
                // Keep the original per-route BF16 rounding and TopK order.
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(inputEvent);
                if (expert + 2 < topK) {
                    LoadUnpermuteInput(input, expandedX, indices.GetValue(indexBase + expert + 2), hidden, inputSlot);
                }
            }
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            DataCopy(output[(uint64_t{tokenBegin} + offset + token) * hidden], result, hidden);
            SetFlag<HardEvent::MTE3_V>(outputEvent);
        }
        offset += count;
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    // Waits above retire both rings; drain the pipes before releasing their UB.
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
    return true;
}

}  // namespace MegaMoeBF16Routing
