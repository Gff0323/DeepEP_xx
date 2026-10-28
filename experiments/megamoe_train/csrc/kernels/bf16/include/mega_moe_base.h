/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_base.h
 * \brief
 */

#ifndef MEGA_MOE_BASE_H
#define MEGA_MOE_BASE_H

#include "lib/std/tuple.h"
#include "mega_moe_tiling.h"
#include "moe_init_routing_v3/arch35/moe_init_routing_v3_apt.h"
#include "mega_moe_workspace_info.h"

using namespace AscendC;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t epHcclBuffer[1024];
};

struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    GM_ADDR cGlobal;
    __gm__ int32_t* groupFlagList;
    __gm__ int32_t* groupFlagList2;
};

struct PeermemInfo {
    GM_ADDR ptrBase;
    GM_ADDR ptrCompactTokenCount;
    GM_ADDR ptrCompactTokenCountBlockSums;
    GM_ADDR ptrTokenPerExpert;
    GM_ADDR ptrA0;  // initrouting结果，包括data+scale
    GM_ADDR ptrRoutedTopkWeights;
    GM_ADDR ptrD;
    uint64_t tokenCountsOffset;
    uint64_t payloadOffset;
    bool compactTokenCountExchange;
    __aicore__ inline PeermemInfo() = default;
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData* tilingData,
                                  bool allowCompactTokenCountExchange = false)
    {
        ptrBase = base;
        const bool compactEnabled = allowCompactTokenCountExchange && tilingData->compactTokenCountExchange != 0;
        const MegaMoeBF16Training::CompactTokenCountExchangeLayout countLayout(
            tilingData->epWorldSize, tilingData->expertPerRank, compactEnabled);
        compactTokenCountExchange = countLayout.enabled;
        ptrCompactTokenCount = countLayout.enabled ? base + countLayout.packetOffset : nullptr;
        ptrCompactTokenCountBlockSums = countLayout.enabled ? base + countLayout.blockSumsOffset : nullptr;
        tokenCountsOffset = countLayout.tokenCountsOffset;
        payloadOffset = countLayout.payloadOffset;
        int64_t offset = static_cast<int64_t>(tokenCountsOffset);
        ptrTokenPerExpert = base + offset;
        offset = static_cast<int64_t>(payloadOffset);
        ptrA0 = base + offset;
        offset += static_cast<uint64_t>(tilingData->m) * tilingData->topK * tilingData->k * sizeof(bfloat16_t);
        ptrRoutedTopkWeights = base + offset;
        offset += Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData->m) * tilingData->topK * static_cast<int64_t>(sizeof(bfloat16_t)),
            static_cast<int64_t>(ALIGN_512));
        ptrD = base + offset;
    }
};

struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR expertIdxGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR bScaleGmAddr;
    GM_ADDR b2GmAddr;
    GM_ADDR b2ScaleGmAddr;
    GM_ADDR probsGmAddr;
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    GM_ADDR permutedInputOutGmAddr;
    GM_ADDR actualRecvTokensOutGmAddr;
    GM_ADDR savedMetaOutGmAddr;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;
    MegaMoeTilingData* tilingData;
};

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];
__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

#endif
