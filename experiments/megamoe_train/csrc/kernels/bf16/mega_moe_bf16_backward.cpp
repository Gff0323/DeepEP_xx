/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * DeepEP MegaMoe BF16 training backward kernel entry.
 *
 * ABI:
 *   deepep_train_mega_moe_bf16_backward_ready_k_kernel_do(
 *       block_dim, stream,
 *       context,           Mc2MoeContext (peermem window addresses + rank)
 *       grad_y,            [M, K] BF16
 *       topk_weights,      [M, T] BF16
 *       weight1,           canonical [E,N,K] or legacy [E,K,N] BF16
 *       weight2,           canonical [E,K,F] or legacy [E,F,K] BF16
 *       grad_x,            [M, K] BF16 (host pre-zeroed)
 *       grad_topk_weights, [M, T] BF16 (host pre-zeroed)
 *       grad_weight1,      same shape/layout as weight1 (host pre-zeroed)
 *       grad_weight2,      same shape/layout as weight2 (host pre-zeroed)
 *       training_saved_state, UInt8 compact forward-persistent state
 *       dispatch_output,      packed X followed by packed TopK weights
 *       backward_workspace, UInt8 backward-only scratch
 *       tiling_host, ffts_addr)
 */
#include "kernel_operator.h"
#include "mega_moe_quant_constants.h"
#include "mega_moe_tiling.h"
#include "mega_moe_bf16_backward.h"
#include "csrc/bf16/kernel_launch.hpp"

using namespace AscendC;
using namespace MegaMoeImpl;

#if defined(ASCEND_MEGAMOE_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
    __global__ __aicore__
    void mega_moe_bf16_backward(GM_ADDR context, GM_ADDR gradY, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
                                GM_ADDR gradX, GM_ADDR gradTopkWeights, GM_ADDR gradWeight1, GM_ADDR gradWeight2,
                                GM_ADDR trainingSavedState, GM_ADDR dispatchOutput, GM_ADDR backwardWorkspace,
                                MegaMoeTilingData tilingData, uint32_t splitExperts)
{
    InitSocState();
#if !defined(ASCEND_MEGAMOE_USE_NEW_ASCENDC_FRONTEND)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    MegaMoeBF16Backward op;
    op.Init(context, gradY, topkWeights, weight1, weight2, gradX, gradTopkWeights, gradWeight1, gradWeight2,
            trainingSavedState, dispatchOutput, backwardWorkspace, &tilingData, splitExperts);
    op.Process();
}

extern "C" ASCEND_MEGAMOE_EXPORT void deepep_train_mega_moe_bf16_backward_ready_k_kernel_do(
    uint32_t block_dim, void* stream, uint8_t* context, uint8_t* grad_y, uint8_t* topk_weights, uint8_t* weight1,
    uint8_t* weight2, uint8_t* grad_x, uint8_t* grad_topk_weights, uint8_t* grad_weight1, uint8_t* grad_weight2,
    uint8_t* training_saved_state, uint8_t* dispatch_output, uint8_t* backward_workspace, const uint8_t* tiling_host,
    uint64_t ffts_addr, uint32_t split_experts)
{
    (void)ffts_addr;
    const auto& tiling = *reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    mega_moe_bf16_backward<<<block_dim, nullptr, stream>>>(
        context, grad_y, topk_weights, weight1, weight2, grad_x, grad_topk_weights, grad_weight1, grad_weight2,
        training_saved_state, dispatch_output, backward_workspace, tiling, split_experts);
}
