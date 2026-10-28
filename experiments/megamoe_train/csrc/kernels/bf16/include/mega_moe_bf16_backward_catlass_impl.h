/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * BF16 grouped GEMM wrappers for the DeepEP MegaMoe backward pipeline.
 *
 * The AIC MMAD / AIV vector split and the per-expert scheduler follow the
 * forward mega_moe_bf16_catlass_impl.h.  The backward needs three data-flow
 * variants of the same BlockMmadTla:
 *
 *   P0-C GMM1 recompute : Z = Xpacked @ W1.logical (A RowMajor; B follows
 *                         the input weight layout; Fixpipe writes gate/up GM).
 *   P1    GMM2 Grad     : dH = dO @ W2.logical.T (A RowMajor; B uses the
 *                         layout opposite to the forward path).
 *   P2    GMM1 Grad     : dX = dZ @ W1.logical.T (same B convention as P1).
 *   P3    GMM1 dW       : canonical dZ^T @ Xpacked -> [N,K], or legacy
 *                         Xpacked^T @ dZ -> [K,N].
 *   P4    GMM2 dW       : canonical dO^T @ H -> [K,F], or legacy
 *                         H^T @ dO -> [F,K].
 *
 * Layout note: the data-gradient stages use RowMajor A and select B's layout
 * from transW. The weight-gradient stages use ColumnMajor A, which catlass on
 * A5 maps to the native
 * transpose-A path (GM ColumnMajor -> L1 nZ -> L0A zN, "Transpose A" in
 * copy_l1_to_l0a.hpp).  No AIV transpose materialization is required.
 */
#ifndef DEEPEP_MEGA_MOE_BF16_BACKWARD_CATLASS_IMPL_H
#define DEEPEP_MEGA_MOE_BF16_BACKWARD_CATLASS_IMPL_H

#include <type_traits>

#include "kernel_operator.h"
#include "op_kernel/mc2_kernel_utils.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
// BF16-only workaround for CATLASS 48f7461 W4A8 VF members under CANN 9.2.
// Keep in sync with the forward include; W4A8 itself is not repaired here.
#define CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP
#define CATLASS_GEMM_BLOCK_MMAD_A8W4_MX_HPP
// The pinned CATLASS aggregate also parses unused HSTU templates whose
// unqualified tuple helpers collide with AscendC::Std under CANN 9.2 beta.
// Exclude only those unused specializations in this BF16 translation unit.
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_QK_SPLIT_ROW_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP
#include "catlass/gemm/block/block_mmad.hpp"
#undef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP
#undef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_QK_SPLIT_ROW_TLA_HPP
#undef CATLASS_GEMM_BLOCK_MMAD_A8W4_MX_HPP
#undef CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "block_epilogue_swiglu_backward_bf16.h"
#include "copy_l0c_to_ub.hpp"
#include "mega_moe_base.h"
#include "backward_dispatch_ready.h"
#include "backward_dispatch_schedule.h"
#include "mega_moe_combine_send.h"

namespace MegaMoeBF16BackwardCatlassImpl {

using ArchTag = Catlass::Arch::Ascend950;
using RowMajor = Catlass::layout::RowMajor;
using ColumnMajor = Catlass::layout::ColumnMajor;
using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>;
using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

constexpr uint32_t L1_TILE_M = 128;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
static_assert(L1_TILE_M == ProducerPhaseMath::kRows, "Backward Dispatch scheduling must use the GEMM M tile size");
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN = L1_TILE_M * L1_TILE_N;
constexpr uint32_t DUAL_AIV_TARGET = 2;
static_assert(2U * MAX_SINGLE_MN * sizeof(bfloat16_t) <= ArchTag::UB_SIZE, "The two BF16 GMM tiles must fit in A5 UB");

// P1 uses one BlockMmad instance (and therefore one CATLASS L1/L0 pipeline),
// but Fixpipe must be able to select either paired AIV at runtime. Encode the
// target AIV in the destination tensor's row coordinate: row 0 means AIV0,
// row L1_TILE_M means AIV1. The physical UB offset remains zero on the chosen
// AIV; the extra logical row band is only a control marker for this copier.
template <class TensorSrc, class TensorDst>
struct CopyL0CToDualAivUBTla {
    using ElementDst = typename TensorDst::Element;
    using ElementSrc = typename TensorSrc::Element;
    static constexpr auto quantPre =
        Catlass::Gemm::Tile::CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst,
                                                   Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;

    template <class DstTensor, class SrcTensor>
    CATLASS_DEVICE void operator()(DstTensor const& dstTensor, SrcTensor const& srcTensor, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.reluEn = false;
        intriParams.unitFlag = unitFlag;

        const uint32_t logicalRow = static_cast<uint32_t>(tla::get<0>(dstTensor.coord()));
        intriParams.subBlockId = logicalRow >= L1_TILE_M ? 1U : 0U;

        // GetTile preserves the parent tensor coordinate. Remove the logical
        // target-AIV band so both physical AIV UBs are addressed at offset 0.
        const auto logicalDstOffset = dstTensor.layout()(dstTensor.coord());
        const auto dstOffset = logicalDstOffset % MAX_SINGLE_MN;
        const auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(dstTensor.data()[dstOffset],
                                                                   srcTensor.data()[srcOffset], intriParams);
    }
};

// Backward TileCopy, parameterized on the A/B operand layouts so that the
// P1/P2 data-gradient stages (RowMajor B) and the P3/P4 weight-gradient
// stages (ColumnMajor A, RowMajor B) select the matching catlass
// specialization.  GmOutput selects L0C -> GM (pure AIC dW stages) vs
// L0C -> the AIV<TargetUBIdx> UB (recompute / data-gradient stages).
template <class LayoutA, class LayoutB, uint32_t TargetUBIdx, bool GmOutput>
struct BackwardBF16TileCopy : Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutA, bfloat16_t, LayoutB,
                                                                     bfloat16_t, RowMajor, void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutA, bfloat16_t, LayoutB, bfloat16_t,
                                                        RowMajor, void>;
    using TensorL0C = typename Base::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst =
        std::conditional_t<TargetUBIdx == DUAL_AIV_TARGET, CopyL0CToDualAivUBTla<TensorL0C, TensorC>,
                           Catlass::Gemm::Tile::CopyL0CToTargetUBTla<
                               ArchTag, TensorL0C, TensorC, Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
                               Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, TargetUBIdx, false>>;
};

// L0C -> GM specialization (pure AIC weight-gradient stages).
template <class LayoutA, class LayoutB, uint32_t TargetUBIdx>
struct BackwardBF16TileCopy<LayoutA, LayoutB, TargetUBIdx, true>
    : Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutA, bfloat16_t, LayoutB, bfloat16_t, RowMajor,
                                             void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutA, bfloat16_t, LayoutB, bfloat16_t,
                                                        RowMajor, void>;
    using TensorL0C = typename Base::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst = Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC,
                                                             Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>;
};

// Backward group GEMM handle: owns the BlockMmadTla and the two UB output
// tiles.  The per-GEMM A/B/C layouts and tensors are constructed per call so
// that per-expert contraction widths (dW stages) are supported.
template <class LayoutA, class LayoutB, uint32_t TargetUBIdx = 0, bool GmOutput = false>
class BackwardGroupMatmulParam {
public:
    using TileCopy = BackwardBF16TileCopy<LayoutA, LayoutB, TargetUBIdx, GmOutput>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, bfloat16_t,
                                                         bfloat16_t, bfloat16_t, void, TileCopy>;

    __aicore__ inline BackwardGroupMatmulParam() : blockMmad(resource)
    {
        blockNum = AscendC::GetBlockNum();
        subBlockIdx = AscendC::GetSubBlockIdx();
        blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        l0cOutUbFirst = resource.ubBuf.template GetBufferByByte<bfloat16_t>(0);
        l0cOutUbSecond = resource.ubBuf.template GetBufferByByte<bfloat16_t>(MAX_SINGLE_MN * sizeof(bfloat16_t));
    }

    Catlass::Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;
    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst;
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond;
    uint32_t blockNum;
    uint32_t subBlockIdx;
    uint32_t blockIdx;
};

// P0-C: GMM1 recompute. Z = Xpacked @ W1.T. The two weight halves are
// independent [M,F] GEMMs and Fixpipe writes them directly to the stable
// gate/up GM planes. This removes the AIV1 UB handoff so recompute can overlap
// P0 gather/dispatch on the vector cores.
template <class WeightLayout>
__aicore__ inline void GroupMatmulRecompute(BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true>& gmm1Params,
                                            const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                            const GMMAddrInfo& gmmAddrInfo, GM_ADDR gatePlane, GM_ADDR upPlane,
                                            uint32_t& startBlockIdx, uint32_t expertIdx)
{
    if constexpr (g_coreType == AscendC::AIC) {
    }
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));
    const uint32_t outputN = n / SWIGLU_N_HALF;

    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    AscendC::GlobalTensor<bfloat16_t> gateGlobal;
    AscendC::GlobalTensor<bfloat16_t> upGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);
    gateGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gatePlane);
    upGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)upPlane);

    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, WeightLayout>(k, n);
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, outputN);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorGate = tla::MakeTensor(gateGlobal, layoutC, Catlass::Arch::PositionGM{});
    auto tensorUp = tla::MakeTensor(upGlobal, layoutC, Catlass::Arch::PositionGM{});

    // Keep this scheduler and startBlockIdx evolution identical to P1.  The
    // same AIC that writes a gate/up tile subsequently produces that tile's
    // dHp and publishes the P1 PIPE_FIX notification, which is the readiness
    // edge used by AIV1 before loading gate/up from GM.
    BlockScheduler scheduler({m, outputN, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx ? gmm1Params.blockIdx + gmm1Params.blockNum : gmm1Params.blockIdx) -
        startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm1Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        if constexpr (g_coreType == AscendC::AIC) {
            auto tensorBlockA =
                tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto tensorBlockB = tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                                                 tla::MakeShape(actualShape.k(), actualShape.n()));
                auto& tensorC = weightBlock == 0 ? tensorGate : tensorUp;
                auto tensorBlockC =
                    tla::GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
                gmm1Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
            }
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
    if constexpr (g_coreType == AscendC::AIC) {
    }
}

// GM experiment: unique output addresses decouple the AIC producer from
// AIV UB reuse. Publish one arrival per AIC after the whole expert's Fixpipe
// writes retire; AIVs retain the original scheduler and alternating ownership.
// There are no per-tile cross-core events here, so an AIC may run ahead without
// exhausting event credits or waiting for a consumer acknowledgement.
// dHpBase aliases this expert's Hp plane: each tile has one AIV owner, which
// loads the complete dHp into UB before replacing it with Hp. All AIC writes
// retire before either lane starts consuming this expert. Other experts use
// disjoint row bands; later dW2 waits for all P1 writes to finish.
template <class WeightLayout>
__aicore__ inline void GMM2GradSwiGLUGradGM(BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true>& gmm2Params,
                                            MegaMoeBF16Impl::BlockEpilogueSwigluBackwardBF16<L1TileShape>& epilogueOp,
                                            const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                            const GMMAddrInfo& gmmAddrInfo, GM_ADDR dHpBase, uint32_t& startBlockIdx,
                                            uint32_t& nextTargetAiv, uint32_t expertIdx, int64_t expertRowBase,
                                            __gm__ int32_t* dispatchReadyFlag, int32_t dispatchTarget,
                                            const BackwardDispatchReadyView& rowReady, __gm__ int32_t* dHpReadyFlag)
{
    if constexpr (g_coreType == AscendC::AIC) {
    } else {
    }
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = Get<N_VALUE>(problemShape);
    const uint32_t k = Get<K_VALUE>(problemShape);
    AscendC::GlobalTensor<bfloat16_t> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gmmAddrInfo.aGlobal));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gmmAddrInfo.bGlobal));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dHpBase));
    auto tensorA = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, RowMajor>(m, k), Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(k, n), Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(c, tla::MakeLayout<bfloat16_t, RowMajor>(m, n), Catlass::Arch::PositionGM{});
    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t loops = scheduler.GetCoreLoops();
    const uint32_t first =
        (gmm2Params.blockIdx < startBlockIdx ? gmm2Params.blockIdx + gmm2Params.blockNum : gmm2Params.blockIdx) -
        startBlockIdx;
    ProducerPhaseSchedule producerSchedule;
    if constexpr (g_coreType == AscendC::AIC) producerSchedule.Init(rowReady, dHpReadyFlag, gmm2Params.blockNum, m, n);
    if constexpr (g_coreType == AscendC::AIC) {
        if (first < loops && !rowReady.enabled) {
            while (dispatchTarget != AscendC::ReadGmByPassDCache(dispatchReadyFlag)) {
            }
        }
    } else {
        epilogueOp.UpdateGlobalAddr(expertRowBase);

        while (AscendC::ReadGmByPassDCache(dHpReadyFlag) < static_cast<int32_t>(gmm2Params.blockNum)) {
        }
    }
    for (uint32_t index = first; index < loops; index += gmm2Params.blockNum, ++nextTargetAiv) {
        auto coord = scheduler.GetBlockCoord(index);
        ProducerPhaseTile producerTile{};
        if constexpr (g_coreType == AscendC::AIC) {
            if (producerSchedule.Enabled()) {
                producerTile = producerSchedule.Get(index);
                coord = Catlass::GemmCoord(producerTile.m, producerTile.n, 0);
            }
        }
        const auto shape = scheduler.GetActualBlockShape(coord);
        const uint32_t row = coord.m() * L1_TILE_M;
        const uint32_t col = coord.n() * L1_TILE_N;
        const uint32_t depth = coord.k() * L1_TILE_K;
        if constexpr (g_coreType == AscendC::AIC) {
            if (rowReady.enabled) {
                if (producerSchedule.Enabled() && producerTile.known)
                    WaitProducerPhaseRows(rowReady, producerTile, row, shape.m());
                else
                    rowReady.WaitRows(row, shape.m());
            }
            auto tileA = tla::GetTile(tensorA, tla::MakeCoord(row, depth), tla::MakeShape(shape.m(), shape.k()));
            auto tileB = tla::GetTile(tensorB, tla::MakeCoord(depth, col), tla::MakeShape(shape.k(), shape.n()));
            auto tileC = tla::GetTile(tensorC, tla::MakeCoord(row, col), tla::MakeShape(shape.m(), shape.n()));
            gmm2Params.blockMmad(tileA, tileB, tileC, shape);
        } else if (gmm2Params.subBlockIdx == (nextTargetAiv & 1U)) {
            epilogueOp.LoadDHpFromGM(dHpBase + (static_cast<int64_t>(row) * n + col) * sizeof(bfloat16_t), shape.m(),
                                     shape.n(), n);
            AscendC::Shape<int64_t, int64_t, int64_t, int64_t> epiShape{shape.m(), shape.n(), 0, 0};
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> offset{
                static_cast<int64_t>(row) * n + col, 0, 0, 0, 0, 0};
            epilogueOp.SwiGLUGrad(epiShape, offset);
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        // Explicit completion includes PIPE_FIX -> GM before PIPE_S publishes
        // readiness. Also publish for cores with zero tiles to avoid deadlock.
        AscendC::PipeBarrier<PIPE_ALL>();
        AtomicAdd(dHpReadyFlag, int32_t(1));

    } else {
    }
    startBlockIdx = (startBlockIdx + loops) % gmm2Params.blockNum;
}

// P1: GMM2 Grad || SwiGLU Grad. AIC alternates completed dHp tiles between
// its two paired AIVs. Each AIV owns an independent UB slot, so Cube can
// produce tile i+1 while the other AIV consumes tile i. A slot is reused only
// after its AIV has published completion through the matching cross-core
// event.
template <class WeightLayout>
__aicore__ inline void GMM2GradSwiGLUGradDualAiv(
    BackwardGroupMatmulParam<RowMajor, WeightLayout, DUAL_AIV_TARGET>& gmm2Params,
    MegaMoeBF16Impl::BlockEpilogueSwigluBackwardBF16<L1TileShape>& epilogueOp, const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape, const GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx, int32_t& outstandingMask, uint32_t& nextTargetAiv, uint32_t expertIdx,
    int64_t expertRowBase, __gm__ int32_t* dispatchReadyFlag, int32_t dispatchTarget, uint32_t p1Event)
{
    if constexpr (g_coreType == AscendC::AIC) {
    } else {
        // Both paired AIVs consume alternating P1 tiles. Keep one dynamic
        // per-expert bar on each AIV lane so load balance is directly visible.
    }
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // F
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // K
    (void)params;
    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);

    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, WeightLayout>(k, n);
    // Two logical row bands encode the Fixpipe target; both resolve to UB
    // offset zero in CopyL0CToDualAivUBTla.
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(2 * L1_TILE_M, L1_TILE_N);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorUbDual = tla::MakeTensor(gmm2Params.l0cOutUbFirst, layoutC, Catlass::Arch::PositionUB{});

    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm2Params.blockIdx < startBlockIdx ? gmm2Params.blockIdx + gmm2Params.blockNum : gmm2Params.blockIdx) -
        startBlockIdx;

    if constexpr (g_coreType == AscendC::AIV) {
        epilogueOp.UpdateGlobalAddr(expertRowBase);
    }

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm2Params.blockNum, ++nextTargetAiv) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        // Carry the parity across expert calls. Some shapes assign only one
        // tile per AIC for each expert; resetting here would send every such
        // tile to AIV0 and silently collapse back to a single-AIV pipeline.
        const uint32_t targetAiv = nextTargetAiv & 1U;
        const int32_t targetBit = 1 << targetAiv;
        const uint16_t targetEvent = static_cast<uint16_t>(p1Event + (targetAiv == 0 ? 0 : FLAG_ID_MAX));

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                while (dispatchTarget != AscendC::ReadGmByPassDCache(dispatchReadyFlag)) {
                    const int64_t start = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - start < 100) {
                    }
                }
            }
            if ((outstandingMask & targetBit) != 0) {
                WaitForVector(targetEvent);
            }
            auto tensorBlockA =
                tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockB =
                tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
            auto tensorBlockUb = tla::GetTile(tensorUbDual, tla::MakeCoord(targetAiv * L1_TILE_M, 0),
                                              tla::MakeShape(actualShape.m(), actualShape.n()));
            gmm2Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb, actualShape);
            NotifyVector(targetEvent);
            outstandingMask |= targetBit;
        }

        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm2Params.subBlockIdx == targetAiv) {
                AscendC::Shape<int64_t, int64_t, int64_t, int64_t> epilogueShape{actualShape.m(), actualShape.n(), 0,
                                                                                 0};
                AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    mLoc * n + nLoc, 0, 0, 0, 0, 0};

                WaitForCube(p1Event);

                epilogueOp.SwiGLUGrad(epilogueShape, epilogueOffset);
                NotifyCube(p1Event);
            }
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm2Params.blockNum;
    if constexpr (g_coreType == AscendC::AIC) {
    } else {
    }
}

// GM dX avoids writing AIV0's P1 UB while later experts are
// still in SwiGLU. AIC runs P2 after its complete P1 producer loop; AIV0
// enters this consumer loop only after its complete P1 epilogue loop.
// The caller retires dZ/Hp producers and all dW2 readers before dX overwrites dO.
template <class WeightLayout>
__aicore__ inline void GMM1GradCombineGM(BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true>& mm,
                                         const Params& params,
                                         const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                         const GMMAddrInfo& addr, GM_ADDR dxBase, uint32_t& startBlockIdx,
                                         uint32_t expertIdx, __gm__ int32_t* dxReady)
{
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = Get<N_VALUE>(problemShape);
    const uint32_t k = Get<K_VALUE>(problemShape);
    if constexpr (g_coreType == AscendC::AIC) {
        // Keep the GEMM interval separate from its input wait.

    } else if (mm.subBlockIdx == 0) {
        while (AscendC::ReadGmByPassDCache(dxReady) < static_cast<int32_t>(mm.blockNum)) {
        }
    }
    AscendC::GlobalTensor<bfloat16_t> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addr.aGlobal));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(addr.bGlobal));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dxBase));
    auto tensorA = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, RowMajor>(m, k), Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(k, n), Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(c, tla::MakeLayout<bfloat16_t, RowMajor>(m, n), Catlass::Arch::PositionGM{});
    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t loops = scheduler.GetCoreLoops();
    const uint32_t first = (mm.blockIdx < startBlockIdx ? mm.blockIdx + mm.blockNum : mm.blockIdx) - startBlockIdx;
    for (uint32_t index = first; index < loops; index += mm.blockNum) {
        const auto coord = scheduler.GetBlockCoord(index);
        auto shape = scheduler.GetActualBlockShape(coord);
        const uint32_t row = coord.m() * L1_TILE_M;
        const uint32_t col = coord.n() * L1_TILE_N;
        const uint32_t depth = coord.k() * L1_TILE_K;
        if constexpr (g_coreType == AscendC::AIC) {
            auto tileA = tla::GetTile(tensorA, tla::MakeCoord(row, depth), tla::MakeShape(shape.m(), shape.k()));
            auto tileB = tla::GetTile(tensorB, tla::MakeCoord(depth, col), tla::MakeShape(shape.k(), shape.n()));
            auto tileC = tla::GetTile(tensorC, tla::MakeCoord(row, col), tla::MakeShape(shape.m(), shape.n()));
            mm.blockMmad(tileA, tileB, tileC, shape);
        } else if (mm.subBlockIdx == 0) {
            // P1 has retired before this phase. Protect loop-carried reuse:
            // MTE3 reads UB -> MTE2 overwrites UB -> MTE3 reads new tile.
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(shape.m()),
                                            static_cast<uint32_t>(shape.n() * sizeof(bfloat16_t)),
                                            static_cast<uint32_t>((n - shape.n()) * sizeof(bfloat16_t)), 0, 0};
            AscendC::DataCopyPadExtParams<bfloat16_t> pad{false, 0U, 0U, 0U};
            AscendC::DataCopyPad(mm.l0cOutUbFirst, c[static_cast<int64_t>(row) * n + col], copy, pad);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            MegaMoeCombineImpl::CombineTokens<bfloat16_t, Catlass::GemmCoord>(
                row, col, n, expertIdx, 0, params.tilingData->rankId, mm.l0cOutUbFirst, shape, params);
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        AscendC::PipeBarrier<PIPE_ALL>();  // Fixpipe GM writes before ready
        AtomicAdd(dxReady, int32_t(1));    // also for zero-tile cores

    } else if (mm.subBlockIdx == 0) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    startBlockIdx = (startBlockIdx + loops) % mm.blockNum;
}

// P2: GMM1 Grad || backward Combine.  AIC computes dX = dZ @ W1 into the
// AIV0 UB; AIV0 sends each dX tile back to the source ranks (CombineTokens).
// Ping-pong tile protocol, identical to the forward GMM2.
template <uint32_t TargetUBIdx, class WeightLayout>
__aicore__ inline void GMM1GradCombine(BackwardGroupMatmulParam<RowMajor, WeightLayout, TargetUBIdx>& gmm1Params,
                                       const Params& params,
                                       const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                       const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom,
                                       uint32_t expertIdx, int64_t expertRowBase, uint16_t& pingpongIdx)
{
    if constexpr (g_coreType == AscendC::AIC) {
    } else {
        if (gmm1Params.subBlockIdx == TargetUBIdx) {
        }
    }
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // K
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // N

    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);

    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, WeightLayout>(k, n);
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(L1_TILE_M, L1_TILE_N);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(gmm1Params.l0cOutUbFirst, layoutC, Catlass::Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(gmm1Params.l0cOutUbSecond, layoutC, Catlass::Arch::PositionUB{});

    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    constexpr uint16_t targetEventOffset = TargetUBIdx == 0 ? 0 : FLAG_ID_MAX;
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx ? gmm1Params.blockIdx + gmm1Params.blockNum : gmm1Params.blockIdx) -
        startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm1Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;
        auto tensorUb = pingpongIdx == 0 ? tensorUbFirst : tensorUbSecond;
        auto outputUb = pingpongIdx == 0 ? gmm1Params.l0cOutUbFirst : gmm1Params.l0cOutUbSecond;

        if constexpr (g_coreType == AscendC::AIC) {
            if (vecSetSyncCom >= 2) {
                WaitForVector(pingpongIdx + targetEventOffset);
            }
            auto tensorBlockA =
                tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockB =
                tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
            auto tensorBlockUb =
                tla::GetTile(tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(actualShape.m(), actualShape.n()));
            gmm1Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb, actualShape);
            NotifyVector(pingpongIdx + targetEventOffset);
        }

        ++vecSetSyncCom;
        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm1Params.subBlockIdx == TargetUBIdx) {
                WaitForCube(pingpongIdx);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                MegaMoeCombineImpl::CombineTokens<bfloat16_t, Catlass::GemmCoord>(
                    mLoc, nLoc, n, expertIdx, expertRowBase, params.tilingData->rankId, outputUb, actualShape, params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                NotifyCube(pingpongIdx);
            }
        }
        pingpongIdx = 1 - pingpongIdx;
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
    if constexpr (g_coreType == AscendC::AIC) {
    } else {
        if (gmm1Params.subBlockIdx == TargetUBIdx) {
        }
    }
}

// P3: GMM1 weight gradient, pure AIC, L0C -> GM. Canonical computes
// dZ^T @ Xpacked -> [N,K]; legacy computes Xpacked^T @ dZ -> [K,N]. In either
// case A is a RowMajor expert band viewed as its ColumnMajor transpose.
__aicore__ inline void GMM1DW(BackwardGroupMatmulParam<ColumnMajor, RowMajor, 0, true>& gmm1Params,
                              const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                              const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t expertIdx)
{
    const uint32_t m = Get<M_VALUE>(problemShape);                         // N
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // K
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // Me

    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    AscendC::GlobalTensor<bfloat16_t> cGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);
    cGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.cGlobal);

    // A = dZ^T: ColumnMajor (N, Me) view of the dZ [Me, N] expert rows.
    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n);
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, n);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(cGlobal, layoutC, Catlass::Arch::PositionGM{});

    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx ? gmm1Params.blockIdx + gmm1Params.blockNum : gmm1Params.blockIdx) -
        startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm1Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;
        auto tensorBlockA =
            tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
        auto tensorBlockB =
            tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
        auto tensorBlockC =
            tla::GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
        gmm1Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
}

// P4: GMM2 weight gradient, pure AIC, L0C -> GM. Canonical computes
// dO^T @ H -> [K,F]; legacy computes H^T @ dO -> [F,K].
__aicore__ inline void GMM2DW(BackwardGroupMatmulParam<ColumnMajor, RowMajor, 1, true>& gmm2Params,
                              const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                              const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t expertIdx)
{
    const uint32_t m = Get<M_VALUE>(problemShape);                         // K
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(problemShape));  // F
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(problemShape));  // Me

    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    AscendC::GlobalTensor<bfloat16_t> cGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.bGlobal);
    cGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)gmmAddrInfo.cGlobal);

    // A = dO^T: ColumnMajor (K, Me) view of the dO [Me, K] expert rows.
    auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(m, k);
    auto layoutB = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n);
    auto layoutC = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, n);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Catlass::Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(cGlobal, layoutC, Catlass::Arch::PositionGM{});

    BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm2Params.blockIdx < startBlockIdx ? gmm2Params.blockIdx + gmm2Params.blockNum : gmm2Params.blockIdx) -
        startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += gmm2Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;
        auto tensorBlockA =
            tla::GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), tla::MakeShape(actualShape.m(), actualShape.k()));
        auto tensorBlockB =
            tla::GetTile(tensorB, tla::MakeCoord(kLoc, nLoc), tla::MakeShape(actualShape.k(), actualShape.n()));
        auto tensorBlockC =
            tla::GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(actualShape.m(), actualShape.n()));
        gmm2Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm2Params.blockNum;
}

}  // namespace MegaMoeBF16BackwardCatlassImpl

#endif  // DEEPEP_MEGA_MOE_BF16_BACKWARD_CATLASS_IMPL_H
