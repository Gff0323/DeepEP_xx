#pragma once

#include "kernel_operator.h"
#include "layout/mega_moe_topk_weight_gather_layout.hpp"

namespace MegaMoeBF16Routing {

// Produce safe byte offsets before Gather: an index outside this window reads
// element zero, never an out-of-range element of a short second window.
__aicore__ inline void MakeWeightWindowOffsets(AscendC::LocalTensor<int32_t> indices,
                                               AscendC::LocalTensor<int32_t> offsets, uint32_t count, bool second)
{
    auto in = (__ubuf__ int32_t*)indices.GetPhyAddr();
    auto out = (__ubuf__ int32_t*)offsets.GetPhyAddr();
    const uint16_t blocks = static_cast<uint16_t>((count + 63) / 64);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> boundary, zero;
        AscendC::Reg::Duplicate(boundary, int32_t{65536});
        AscendC::Reg::Duplicate(zero, int32_t{0});
        uint32_t remaining = count;
        for (uint16_t i = 0; i < blocks; ++i) {
            auto valid = AscendC::Reg::UpdateMask<int32_t>(remaining);
            AscendC::Reg::RegTensor<int32_t> index, local, safe;
            AscendC::Reg::MaskReg belongs;
            AscendC::Reg::DataCopy(index, in + i * 64);
            if (second) {
                AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::GE>(belongs, index, boundary, valid);
                AscendC::Reg::Sub(local, index, boundary, valid);
            } else {
                AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(belongs, index, boundary, valid);
                AscendC::Reg::Adds(local, index, int32_t{0}, valid);
            }
            AscendC::Reg::Select<int32_t>(safe, local, zero, belongs);
            AscendC::Reg::Muls(safe, safe, int32_t{2}, valid);
            AscendC::Reg::DataCopy(out + i * 64, safe, valid);
        }
    }
}

// Select raw BF16 bits; there is no float Cast or numerical round trip.
__aicore__ inline void MergeWeightWindows(AscendC::LocalTensor<int32_t> indices, AscendC::LocalTensor<uint16_t> first,
                                          AscendC::LocalTensor<uint16_t> second, uint32_t count)
{
    auto idx = (__ubuf__ int32_t*)indices.GetPhyAddr();
    auto a = (__ubuf__ uint16_t*)first.GetPhyAddr();
    auto b = (__ubuf__ uint16_t*)second.GetPhyAddr();
    const uint16_t blocks = static_cast<uint16_t>((count + 63) / 64);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> boundary;
        AscendC::Reg::Duplicate(boundary, int32_t{65536});
        uint32_t remaining = count;
        for (uint16_t i = 0; i < blocks; ++i) {
            auto valid = AscendC::Reg::UpdateMask<int32_t>(remaining);
            AscendC::Reg::RegTensor<int32_t> index;
            AscendC::Reg::RegTensor<uint16_t> av, bv, result;
            AscendC::Reg::MaskReg belongs;
            AscendC::Reg::DataCopy(index, idx + i * 64);
            AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(belongs, index, boundary, valid);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(av, a + i * 64);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bv, b + i * 64);
            AscendC::Reg::Select<uint32_t>((AscendC::Reg::RegTensor<uint32_t>&)result,
                                           (AscendC::Reg::RegTensor<uint32_t>&)av,
                                           (AscendC::Reg::RegTensor<uint32_t>&)bv, belongs);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(a + i * 64, result, valid);
        }
    }
}

__aicore__ inline bool GatherTopkWeightsTwoWindows(GM_ADDR weightsAddr, GM_ADDR sortedIndicesAddr, GM_ADDR outputAddr,
                                                   uint64_t totalRoutes, const TopkWeightGatherPlan& plan)
{
    using namespace AscendC;
    TPipe pipe;
    TBuf<> sourceBuf, indicesBuf, offsetsBuf, firstBuf, secondBuf;
    pipe.InitBuffer(sourceBuf, plan.sourceBytes);  // First allocation stays at UB zero.
    pipe.InitBuffer(indicesBuf, plan.kIndexBytes);
    pipe.InitBuffer(offsetsBuf, plan.kIndexBytes);
    pipe.InitBuffer(firstBuf, plan.kOutputBytes);
    pipe.InitBuffer(secondBuf, plan.kOutputBytes);
    auto source = sourceBuf.Get<uint16_t>();
    auto indices = indicesBuf.Get<int32_t>();
    auto offsets = offsetsBuf.Get<int32_t>();
    auto first = firstBuf.Get<uint16_t>();
    auto second = secondBuf.Get<uint16_t>();
    GlobalTensor<uint16_t> weightsGm, outputGm;
    GlobalTensor<int32_t> indicesGm;
    weightsGm.SetGlobalBuffer((__gm__ uint16_t*)weightsAddr);
    indicesGm.SetGlobalBuffer((__gm__ int32_t*)sortedIndicesAddr);
    outputGm.SetGlobalBuffer((__gm__ uint16_t*)outputAddr);
    for (uint32_t offset = 0; offset < plan.count; offset += plan.kBatchRoutes) {
        const uint32_t count = plan.BatchCount(offset);
        SyncFunc<HardEvent::V_MTE2>();
        SyncFunc<HardEvent::MTE3_V>();
        DataCopyPad(indices, indicesGm[plan.begin + offset], DataCopyExtParams{1U, count * 4U, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        DataCopyPad(source, weightsGm, DataCopyExtParams{1U, plan.kMaxRoutes * 2U, 0U, 0U, 0U},
                    DataCopyPadExtParams<uint16_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_V>();
        MakeWeightWindowOffsets(indices, offsets, count, false);
        PipeBarrier<PIPE_V>();
        Gather(first, source, offsets.ReinterpretCast<uint32_t>(), 0U, count);
        // Gather must finish reading the first window before MTE2 overwrites it.
        SyncFunc<HardEvent::V_MTE2>();
        DataCopyPad(source, weightsGm[plan.kMaxRoutes],
                    DataCopyExtParams{1U, static_cast<uint32_t>(totalRoutes - plan.kMaxRoutes) * 2U, 0U, 0U, 0U},
                    DataCopyPadExtParams<uint16_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_V>();
        MakeWeightWindowOffsets(indices, offsets, count, true);
        PipeBarrier<PIPE_V>();
        Gather(second, source, offsets.ReinterpretCast<uint32_t>(), 0U, count);
        PipeBarrier<PIPE_V>();
        MergeWeightWindows(indices, first, second, count);
        SyncFunc<HardEvent::V_MTE3>();
        DataCopyPad(outputGm[plan.begin + offset], first, DataCopyExtParams{1U, count * 2U, 0U, 0U, 0U});
    }
    SyncFunc<HardEvent::MTE3_S>();
    pipe.Destroy();
    return true;
}

// Shared by BF16, MGroup and UDMA before their Dispatch/transport UB setup.
// No persistent state or tiling changes; false requests the scalar fallback.
__aicore__ inline bool TryGatherTopkWeights(GM_ADDR weightsAddr, GM_ADDR sortedIndicesAddr, GM_ADDR outputAddr,
                                            uint64_t totalRoutes, uint32_t coreIndex, uint32_t coreCount,
                                            bool allowTwoWindows = false)
{
    using namespace AscendC;
    const TopkWeightGatherPlan plan(totalRoutes, coreIndex, coreCount, allowTwoWindows);
    if (!plan.enabled) {
        return false;
    }
    if (plan.count == 0) {
        return true;
    }
    if (plan.twoWindows) {
        return GatherTopkWeightsTwoWindows(weightsAddr, sortedIndicesAddr, outputAddr, totalRoutes, plan);
    }

    TPipe pipe;
    TBuf<> sourceBuf;
    TBuf<> indicesBuf;
    TBuf<> outputBuf;
    pipe.InitBuffer(sourceBuf, plan.sourceBytes);
    pipe.InitBuffer(indicesBuf, TopkWeightGatherPlan::kIndexBytes);
    pipe.InitBuffer(outputBuf, TopkWeightGatherPlan::kOutputBytes);
    auto source = sourceBuf.Get<uint16_t>();
    auto indices = indicesBuf.Get<int32_t>();
    auto output = outputBuf.Get<uint16_t>();
    GlobalTensor<uint16_t> weightsGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<uint16_t> outputGm;
    weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(weightsAddr));
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortedIndicesAddr));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(outputAddr));

    // Copy only valid GM bytes; the UB allocation alone is rounded up.
    DataCopyPad(source, weightsGm, DataCopyExtParams{1U, static_cast<uint32_t>(totalRoutes * 2), 0U, 0U, 0U},
                DataCopyPadExtParams<uint16_t>{false, 0U, 0U, 0U});
    for (uint32_t offset = 0; offset < plan.count; offset += TopkWeightGatherPlan::kBatchRoutes) {
        const uint32_t count = plan.BatchCount(offset);
        // Previous Gather must finish reading indices before MTE2 reuses them.
        SyncFunc<HardEvent::V_MTE2>();
        DataCopyPad(indices, indicesGm[plan.begin + offset],
                    DataCopyExtParams{1U, count * uint32_t{sizeof(int32_t)}, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_V>();
        // Gather takes byte offsets. Reinterpret BF16 as uint16, without Cast.
        Muls(indices, indices, int32_t{2}, count);
        PipeBarrier<PIPE_V>();
        SyncFunc<HardEvent::MTE3_V>();
        Gather(output, source, indices.ReinterpretCast<uint32_t>(), 0U, count);
        SyncFunc<HardEvent::V_MTE3>();
        DataCopyPad(outputGm[plan.begin + offset], output,
                    DataCopyExtParams{1U, count * uint32_t{sizeof(uint16_t)}, 0U, 0U, 0U});
    }
    // Count notification must not overtake the MTE3 publication to SHMEM.
    // Release this UB phase before FirstBuffInit allocates Dispatch buffers.
    SyncFunc<HardEvent::MTE3_S>();
    pipe.Destroy();
    return true;
}

}  // namespace MegaMoeBF16Routing
