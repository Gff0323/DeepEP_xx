/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * BF16 SwiGLU backward epilogues for the DeepEP MegaMoe backward pipeline.
 *
 * The vector-side epilogue follows the FP32 intermediate / BF16 output
 * convention of the forward BlockEpilogueSwigluForwardBF16: SiLU, its product with
 * Up, and the local derivative intermediates stay FP32 until their BF16
 * tensor boundary. Do not insert the standalone F.silu(BF16) rounding.
 *
 *   SwiGLUGrad         : dHp tile (GMM2 Grad output, in UB) + packed p +
 *                       recomputed H -> Hp, dHp*H and dZ.  The route weight
 *                       is applied before the SwiGLU derivative: dH=p*dHp.
 *
 * The register instruction organisation (Exp / Div / Mul / Cast in
 * __VEC_SCOPE__) mirrors ComputeSwiglu() of the forward epilogue.
 *
 * Tile addressing convention (matches the forward GMM1 epilogue): the caller
 * passes a single flattened block offset `mLoc * F + nLoc` (rows * plane
 * width + tile column base) in Get<0>(blockCoord).  Row strides are handled
 * by DataCopyPad.
 */
#ifndef DEEPEP_BLOCK_EPILOGUE_SWIGLU_BACKWARD_BF16_HPP
#define DEEPEP_BLOCK_EPILOGUE_SWIGLU_BACKWARD_BF16_HPP

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"
#include "layout/mega_moe_context.hpp"
#include "layout/mega_moe_layout.hpp"

namespace MegaMoeBF16Impl {

static constexpr AscendC::Reg::DivSpecificMode kBackwardDivMode = {AscendC::Reg::MaskMergeMode::ZEROING, true};
static constexpr AscendC::Reg::CastTrait kBackwardCastBF16ToFP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait kBackwardCastFP32ToBF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// Amplifier for the clamp range indicator: |raw - clamped| is exactly zero
// for unclamped elements, so multiplying by this scale saturates any nonzero
// difference before min(1, .) turns it into a clean 0/1 mask.
static constexpr float kClampMaskScale = 1.0e30F;

template <typename L0TileShape_>
class BlockEpilogueSwigluBackwardBF16 {
public:
    struct Arguments {
        GM_ADDR gatePlaneGmAddr{nullptr};
        GM_ADDR upPlaneGmAddr{nullptr};
        GM_ADDR gateScaleGmAddr{nullptr};
        GM_ADDR upScaleGmAddr{nullptr};
        GM_ADDR gateHighGmAddr{nullptr};
        GM_ADDR upLowGmAddr{nullptr};
        GM_ADDR upHighGmAddr{nullptr};
        GM_ADDR gradZGmAddr{nullptr};
        GM_ADDR hpGmAddr{nullptr};
        GM_ADDR dpProductsGmAddr{nullptr};
        GM_ADDR packedTopkWeightsGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        // dsv4 swiglu_limit clamp; kMegaMoeSwigluLimitDisabled disables it.
        float swigluLimit{kMegaMoeSwigluLimitDisabled};
        uint32_t savedGateUpMode{MegaMoeBF16Training::kSaveNone};
    };

    using Params = Arguments;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    __aicore__ inline void Init(Params const& params)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            // Store the whole parameter set: ComputeForwardAux /
            // ComputeSwigluGrad / ComputeSwigluRecompute read
            // params_.swigluLimit from it.  Without this copy the limit
            // stays at the default (disabled) value and the clamp is
            // silently skipped in the backward pass.
            params_ = params;
            gatePlaneGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.gatePlaneGmAddr);
            upPlaneGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.upPlaneGmAddr);
            gateMxFp8Gm_.SetGlobalBuffer((__gm__ int8_t*)params.gatePlaneGmAddr);
            upMxFp8Gm_.SetGlobalBuffer((__gm__ int8_t*)params.upPlaneGmAddr);
            gateScaleGm_.SetGlobalBuffer((__gm__ int8_t*)params.gateScaleGmAddr);
            upScaleGm_.SetGlobalBuffer((__gm__ int8_t*)params.upScaleGmAddr);
            gateHighGm_.SetGlobalBuffer((__gm__ int8_t*)params.gateHighGmAddr);
            upLowGm_.SetGlobalBuffer((__gm__ int8_t*)params.upLowGmAddr);
            upHighGm_.SetGlobalBuffer((__gm__ int8_t*)params.upHighGmAddr);
            gradZGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.gradZGmAddr);
            hpGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.hpGmAddr);
            dpPartialsGm_.SetGlobalBuffer((__gm__ float*)params.dpProductsGmAddr);
            packedTopkWeightsGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.packedTopkWeightsGmAddr);
        }
    }

    // baseRows: cumulative expert row offset (M_OFFSET equivalent). All four
    // planes are addressed from this single row base: gate/up/H planes have
    // row width F, gradZ has row width N.
    __aicore__ inline void UpdateGlobalAddr(int64_t baseRows)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            rowBase_ = baseRows;
        }
    }

    // problemShape = {0, N, K, 0}; F = N / 2.
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        outputN_ = Get<1>(problemShape);
        outputF_ = outputN_ / SWIGLU_N_HALF;
        dpPartialTiles_ =
            Ops::Base::CeilDiv(static_cast<uint64_t>(outputF_), static_cast<uint64_t>(BACKWARD_DP_PARTIAL_TILE_N));
    }

    // Saved-mode GM output: each AIV loads its assigned dHp tile into its
    // private first work tile, then reuses the unchanged SwiGLU epilogue.
    __aicore__ inline void LoadDHpFromGM(GM_ADDR src, uint32_t rows, uint32_t columns, uint32_t globalStride)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::GlobalTensor<bfloat16_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(src));
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows),
                                        static_cast<uint32_t>(columns * sizeof(bfloat16_t)),
                                        static_cast<uint32_t>((globalStride - columns) * sizeof(bfloat16_t)), 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> pad{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(l0cOutUbFirst_, gm, copy, pad);
        // Complete the GM read before V uses dHp. Later Hp writes use the
        // V_MTE3 edge, so MTE2 -> V -> MTE3 protects in-place GM replacement.
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    // ---- P1: early-topk backward epilogue. dHp is the GMM2 Grad fixpipe
    // output at UB offset 0.  First produce Hp and dHp*H, store both, reload
    // gate/up, then apply dH=p*dHp and produce dZ.  Reloading is intentional:
    // three A5 UB tiles cannot simultaneously retain dHp/gate/up/Hp/dp. ----
    __aicore__ inline void SwiGLUGrad(const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        const uint32_t singleM = Get<0>(blockShape);
        const uint32_t singleN = Get<1>(blockShape);
        if (singleM == 0 || singleN == 0) {
            return;
        }
        const int64_t flatOffset = Get<0>(blockCoord);
        const uint32_t mLoc = static_cast<uint32_t>(flatOffset / outputF_);
        const uint32_t nLoc = static_cast<uint32_t>(flatOffset % outputF_);
        const uint32_t rowStrideInPlane = Ops::Base::CeilAlign(
            static_cast<uint32_t>(singleN), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(bfloat16_t)));

        // One contiguous BF16 load, reused by the two separately rounded
        // products Hp=p*H and dH=p*dHp. No GM payload/layout change.
        LoadPackedWeights(singleM, mLoc);

        // The caller's WaitForCube runs on PIPE_V and protects the P1 dHp
        // Fixpipe tile. Bridge PIPE_V to MTE2 before loading gate/up into the
        // other two work tiles. In save=none they come from current-backward
        // recompute GM; saved modes read forward-persistent GM instead.
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        // Load gate/up while dHp remains in the Fixpipe-produced first tile.
        LoadGateUpTiles(rowBase_ + mLoc, nLoc, singleM, singleN, l0cOutUbSecond_, extraA_);

        // No partial-tile layout changes. Gate/up is already loaded exactly once.
        if (singleM == kTileRows && singleN == kTileColumns && rowStrideInPlane == singleN) {
            RunFusedFullTile(singleM, singleN, mLoc, nLoc);
            return;
        }
        ComputeForwardAux(singleM, singleN, rowStrideInPlane);
        AscendC::PipeBarrier<PIPE_V>();
        // dp uses the actual BF16 H consumed by the forward multiply, not the
        // pre-cast FP32 register value. Form the BF16 product in-place, then
        // reduce each row to one FP32 partial while the tile is still in UB.
        for (uint32_t row = 0; row < singleM; ++row) {
            AscendC::Mul(l0cOutUbSecond_[row * rowStrideInPlane], l0cOutUbFirst_[row * rowStrideInPlane],
                         extraA_[row * rowStrideInPlane], singleN);
        }
        ComputeDpPartials(singleM, singleN, rowStrideInPlane);
        // Convert H to Hp with the prefetched per-row weights, retaining
        // the separate BF16 product before dHp is multiplied by the same p.
        ApplyPackedWeight(extraA_, singleM, singleN, rowStrideInPlane, mLoc);

        // The FP32 partials reuse the start of each row in tile1; tile2 holds
        // Hp. Both are V outputs consumed by MTE3.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const uint32_t dpTileIdx = nLoc / BACKWARD_DP_PARTIAL_TILE_N;
        AscendC::DataCopyExtParams partialCopyParams{
            static_cast<uint16_t>(singleM), static_cast<uint32_t>(sizeof(float)),
            // For UB -> GM DataCopyPad, srcStride is counted in 32-byte
            // DataBlocks. A 4-byte blockLen still occupies one source
            // DataBlock, so skip the remaining blocks in the padded UB row.
            static_cast<uint32_t>(rowStrideInPlane * sizeof(bfloat16_t) / AscendC::ONE_BLK_SIZE - 1U),
            static_cast<uint32_t>((dpPartialTiles_ - 1) * sizeof(float)), 0};
        AscendC::DataCopyPad(dpPartialsGm_[(rowBase_ + mLoc) * dpPartialTiles_ + dpTileIdx],
                             l0cOutUbSecond_.template ReinterpretCast<float>(), partialCopyParams);
        AscendC::DataCopyExtParams auxCopyParams{static_cast<uint16_t>(singleM),
                                                 static_cast<uint32_t>(singleN * sizeof(bfloat16_t)), 0,
                                                 static_cast<uint32_t>((outputF_ - singleN) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(hpGm_[rowBase_ * outputF_ + flatOffset], extraA_, auxCopyParams);

        // Both buffers are MTE3 sources and will be overwritten by MTE2.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        LoadGateUpTiles(rowBase_ + mLoc, nLoc, singleM, singleN, l0cOutUbSecond_, extraA_);

        // dHp -> dH before the SwiGLU derivative.
        ApplyPackedWeight(l0cOutUbFirst_, singleM, singleN, rowStrideInPlane, mLoc);
        AscendC::PipeBarrier<PIPE_V>();
        ComputeSwigluGrad(singleM, singleN, rowStrideInPlane);

        // extraA_ holds the up tile.  PIPE_V reads it while computing dGate,
        // and the next loop iteration reuses the same UB through PIPE_MTE2.
        // Keep that loop-carried WAR dependency explicit; MTE2_V above only
        // orders the current load before V and does not protect the next load.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);

        // dGate/dUp are produced by PIPE_V and consumed by PIPE_MTE3.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        // dGate tile (UB offset 0) -> ptrGradZ rows at col nLoc.
        // dUp tile (UB offset kTileBytes) -> ptrGradZ rows at col F + nLoc.
        AscendC::DataCopyExtParams gradCopyParams{static_cast<uint16_t>(singleM),
                                                  static_cast<uint32_t>(singleN * sizeof(bfloat16_t)), 0,
                                                  static_cast<uint32_t>((outputN_ - singleN) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(gradZGm_[rowBase_ * outputN_ + mLoc * outputN_ + nLoc], l0cOutUbFirst_, gradCopyParams);
        AscendC::DataCopyPad(gradZGm_[rowBase_ * outputN_ + mLoc * outputN_ + outputF_ + nLoc], l0cOutUbSecond_,
                             gradCopyParams);
        // l0cOutUbSecond_ is the MTE3 source for dUp and becomes the MTE2
        // destination for the next gate tile.  Do not let that next load
        // overwrite dUp while MTE3 is still reading it.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        // dZ must be visible to the AIC's P2 GMM1 Grad before NotifyCube,
        // which is issued by PIPE_V in the caller.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

private:
    static constexpr uint32_t kTileRows = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t kTileColumns = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t kTileElements = kTileRows * kTileColumns;
    static constexpr uint32_t kTileBytes = kTileElements * sizeof(bfloat16_t);
    static constexpr uint32_t kDecodeRows = 16;
    static constexpr uint32_t kMxDataElements = kDecodeRows * kTileColumns;
    static constexpr uint32_t kMxScaleRowBytes = 32;
    static constexpr uint32_t kMxScaleBytes = kDecodeRows * kMxScaleRowBytes;
    // DIST_INTLV_B16/B32 are dual-register stores.  They ignore the mask and
    // always write 2 * VL (512 bytes on A5), even though only the compact
    // prefix is consumed below.  Reserve the full physical store footprint;
    // otherwise each following scale row overwrites the beginning of
    // mxDecoded_ and corrupts an already decoded row in the current chunk.
    static constexpr uint32_t kMxDualRegisterStoreBytes = 2U * 256U;
    static constexpr uint32_t kMxScaleBf16Elements = kMxDualRegisterStoreBytes / sizeof(bfloat16_t);
    static constexpr uint32_t kMxScaleBf16Bytes = kMxScaleBf16Elements * sizeof(bfloat16_t);
    static constexpr uint32_t kMxScaleFp32Elements = kMxDualRegisterStoreBytes / sizeof(float);
    static constexpr uint32_t kMxScaleFp32Bytes = kMxScaleFp32Elements * sizeof(float);
    static constexpr uint32_t kMxDecodedElements = kDecodeRows * kTileColumns;
    static constexpr uint32_t kMxDecodedBytes = kMxDecodedElements * sizeof(float);
    static constexpr uint32_t kMxDataOffset = 3U * kTileBytes;
    static constexpr uint32_t kMxScaleOffset = kMxDataOffset + kMxDataElements;
    static constexpr uint32_t kMxScaleBf16Offset = kMxScaleOffset + kMxScaleBytes;
    static constexpr uint32_t kMxScaleFp32Offset = kMxScaleBf16Offset + kMxScaleBf16Bytes;
    static constexpr uint32_t kMxDecodedOffset = kMxScaleFp32Offset + kMxScaleFp32Bytes;
    // Retain every decoded row until the whole-tile gradient mask consumes it.
    static constexpr uint32_t kClipMaskBytes = kTileRows * kMxScaleRowBytes;
    static constexpr uint32_t kGateHighOffset = kMxDecodedOffset + kMxDecodedBytes;
    static constexpr uint32_t kUpLowOffset = kGateHighOffset + kClipMaskBytes;
    static constexpr uint32_t kUpHighOffset = kUpLowOffset + kClipMaskBytes;
    static constexpr uint32_t kMxScratchEnd = kUpHighOffset + kClipMaskBytes;
    static_assert(kMxScratchEnd <= Catlass::Arch::Ascend950::UB_SIZE,
                  "The backward SwiGLU work tiles and MXFP8 decoder must fit in A5 UB");
    static constexpr uint32_t kWeightBytes = (kTileRows * sizeof(bfloat16_t) + 31U) / 32U * 32U;
    static constexpr uint32_t kWeightOffset = Catlass::Arch::Ascend950::UB_SIZE - kWeightBytes;
    static_assert(kWeightOffset >= kMxScratchEnd, "Prefetched weights must not alias live MXFP8 decode or clip masks");

    __aicore__ inline void LoadPackedWeights(uint32_t rows, uint32_t mLoc)
    {
        // On this AIV P0 staging has retired before SwiGLUGrad is called;
        // paired AIC Fixpipe uses only the first tile, below this region.
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::DataCopyExtParams p{1U, static_cast<uint32_t>(rows * sizeof(bfloat16_t)), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<bfloat16_t> pad{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(packedWeightUb_, packedTopkWeightsGm_[rowBase_ + mLoc], p, pad);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline void LoadGateUpTiles(uint32_t rowStart, uint32_t nLoc, uint32_t singleM, uint32_t singleN,
                                           AscendC::LocalTensor<bfloat16_t>& gateDst,
                                           AscendC::LocalTensor<bfloat16_t>& upDst)
    {
        if (params_.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8) {
            DecodeMxFp8Plane(gateMxFp8Gm_, gateScaleGm_, gateHighGm_, gateHighGm_, rowStart, nLoc, singleM, singleN,
                             gateDst, true);
            DecodeMxFp8Plane(upMxFp8Gm_, upScaleGm_, upLowGm_, upHighGm_, rowStart, nLoc, singleM, singleN, upDst,
                             false);
            return;
        }
        const int64_t index = static_cast<int64_t>(rowStart) * outputF_ + nLoc;
        CopyPlaneTileToUb(gatePlaneGm_, index, singleM, singleN, outputF_, gateDst);
        CopyPlaneTileToUb(upPlaneGm_, index, singleM, singleN, outputF_, upDst);
    }

    __aicore__ inline void CopyMxRowsToUb(const AscendC::GlobalTensor<int8_t>& src, int64_t index, uint16_t rows,
                                          uint32_t rowBytes, uint32_t srcRowBytes, AscendC::LocalTensor<int8_t> dst)
    {
        AscendC::DataCopyExtParams copyParams{rows, rowBytes, srcRowBytes - rowBytes, 0, 0};
        AscendC::DataCopyPadExtParams<int8_t> padParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(dst, src[index], copyParams, padParams);
    }

    // Decode one E4M3FN plane in small row chunks. The E8M0 conversion and
    // group broadcast mirror the production MXFP8 combine path; no full
    // decompressed GM plane is materialized.
    __aicore__ inline void DecodeMxFp8Plane(const AscendC::GlobalTensor<int8_t>& dataGm,
                                            const AscendC::GlobalTensor<int8_t>& scaleGm,
                                            const AscendC::GlobalTensor<int8_t>& lowMaskGm,
                                            const AscendC::GlobalTensor<int8_t>& highMaskGm, uint32_t rowStart,
                                            uint32_t nLoc, uint32_t singleM, uint32_t singleN,
                                            AscendC::LocalTensor<bfloat16_t>& dst, bool gatePlane)
    {
        const uint32_t scaleWidth = outputF_ / 32;
        const uint32_t groups = singleN / 32;
        const bool hasLowMask = params_.gateHighGmAddr != nullptr;
        for (uint32_t chunkRow = 0; chunkRow < singleM; chunkRow += kDecodeRows) {
            const uint16_t rows =
                static_cast<uint16_t>(singleM - chunkRow > kDecodeRows ? kDecodeRows : singleM - chunkRow);
            const uint32_t globalRow = rowStart + chunkRow;
            const int64_t dataIndex = static_cast<int64_t>(globalRow) * outputF_ + nLoc;
            const int64_t scaleIndex = static_cast<int64_t>(globalRow) * scaleWidth + nLoc / 32;
            // The scratch regions are overwritten by MTE2 in the next chunk;
            // wait for all vector consumers from this chunk first.
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            CopyMxRowsToUb(dataGm, dataIndex, rows, singleN, outputF_, mxData_);
            CopyMxRowsToUb(scaleGm, scaleIndex, rows, groups, scaleWidth, mxScale_);
            if (hasLowMask) {
                const uint32_t maskWidthBytes = scaleWidth * sizeof(uint32_t);
                const int64_t maskIndex =
                    static_cast<int64_t>(globalRow) * maskWidthBytes + (nLoc / 32) * sizeof(uint32_t);
                if (gatePlane) {
                    CopyMxRowsToUb(lowMaskGm, maskIndex, rows, groups * sizeof(uint32_t), maskWidthBytes,
                                   gateHighMask_[chunkRow * kMxScaleRowBytes]);
                } else {
                    CopyMxRowsToUb(lowMaskGm, maskIndex, rows, groups * sizeof(uint32_t), maskWidthBytes,
                                   upLowMask_[chunkRow * kMxScaleRowBytes]);
                }
                if (!gatePlane) {
                    CopyMxRowsToUb(highMaskGm, maskIndex, rows, groups * sizeof(uint32_t), maskWidthBytes,
                                   upHighMask_[chunkRow * kMxScaleRowBytes]);
                }
                AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
            }
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
            DecodeMxFp8Rows(rows, singleN, groups);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(dst[chunkRow * singleN], mxDecoded_, AscendC::RoundMode::CAST_RINT,
                          static_cast<uint32_t>(rows) * singleN);
            if (hasLowMask) {
                // The masks record decisions made from raw forward BF16.
                // Force masked values beyond the finite limit so the existing
                // FP32 clamp restores exactly +/-L and produces a zero range
                // indicator, without re-inferring the decision from MXFP8.
                AscendC::SyncFunc<AscendC::HardEvent::V_S>();
                ApplyClipSentinels(dst[chunkRow * singleN], rows, singleN, groups, gatePlane, chunkRow);
                AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
                AscendC::SyncFunc<AscendC::HardEvent::S_V>();
            }
        }
    }

    __aicore__ inline void DecodeMxFp8Rows(uint16_t rows, uint32_t columns, uint32_t groups)
    {
        constexpr uint32_t bf16PerReg = 256U / sizeof(bfloat16_t);
        constexpr uint32_t fp32PerReg = 256U / sizeof(float);
        auto data = (__ubuf__ fp8_e4m3fn_t*)mxData_.GetPhyAddr();
        auto scales = (__ubuf__ fp8_e8m0_t*)mxScale_.GetPhyAddr();
        auto scaleBf16 = (__ubuf__ bfloat16_t*)mxScaleBf16_.GetPhyAddr();
        auto scaleFp32 = (__ubuf__ float*)mxScaleFp32_.GetPhyAddr();
        auto decoded = (__ubuf__ float*)mxDecoded_.GetPhyAddr();
        __VEC_SCOPE__
        {
            static constexpr AscendC::Reg::CastTrait fp8ToWide = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN};
            for (uint16_t row = 0; row < rows; ++row) {
                uint32_t scaleRemaining = groups;
                const uint16_t scaleRepeats = static_cast<uint16_t>(Ops::Base::CeilDiv(groups, bf16PerReg));
                for (uint16_t i = 0; i < scaleRepeats; ++i) {
                    auto mask = AscendC::Reg::UpdateMask<bfloat16_t>(scaleRemaining);
                    AscendC::Reg::RegTensor<fp8_e8m0_t> src;
                    AscendC::Reg::RegTensor<bfloat16_t> value;
                    AscendC::Reg::DataCopy<fp8_e8m0_t, AscendC::Reg::LoadDist::DIST_UNPACK_B8>(
                        src, scales + row * kMxScaleRowBytes + i * bf16PerReg);
                    AscendC::Reg::Cast<bfloat16_t, fp8_e8m0_t, fp8ToWide>(value, src, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_INTLV_B16>(
                        scaleBf16 + i * bf16PerReg * 2, value, value, mask);
                }
                AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                uint32_t scaleWideRemaining = groups * 2;
                const uint16_t scaleWideRepeats = static_cast<uint16_t>(Ops::Base::CeilDiv(groups * 2, fp32PerReg));
                for (uint16_t i = 0; i < scaleWideRepeats; ++i) {
                    auto mask = AscendC::Reg::UpdateMask<float>(scaleWideRemaining);
                    AscendC::Reg::RegTensor<bfloat16_t> src;
                    AscendC::Reg::RegTensor<float> value;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        src, scaleBf16 + i * fp32PerReg);
                    AscendC::Reg::Cast<float, bfloat16_t, fp8ToWide>(value, src, mask);
                    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_INTLV_B32>(
                        scaleFp32 + i * fp32PerReg * 2, value, value, mask);
                }
                AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                uint32_t dataRemaining = columns;
                const uint16_t dataRepeats = static_cast<uint16_t>(Ops::Base::CeilDiv(columns, fp32PerReg));
                for (uint16_t i = 0; i < dataRepeats; ++i) {
                    auto mask = AscendC::Reg::UpdateMask<float>(dataRemaining);
                    AscendC::Reg::RegTensor<float> scale;
                    AscendC::Reg::RegTensor<fp8_e4m3fn_t> src;
                    AscendC::Reg::RegTensor<float> value;
                    AscendC::Reg::RegTensor<float> result;
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_E2B_B32>(scale, scaleFp32 + i * 8);
                    AscendC::Reg::DataCopy<fp8_e4m3fn_t, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
                        src, data + row * columns + i * fp32PerReg);
                    AscendC::Reg::Cast<float, fp8_e4m3fn_t, fp8ToWide>(value, src, mask);
                    AscendC::Reg::Mul(result, scale, value, mask);
                    AscendC::Reg::DataCopy(decoded + row * columns + i * fp32PerReg, result, mask);
                }
            }
        }
    }

    __aicore__ inline void ApplyClipSentinels(AscendC::LocalTensor<bfloat16_t> dst, uint16_t rows, uint32_t columns,
                                              uint32_t groups, bool gatePlane, uint32_t maskRow)
    {
        auto dstBits = dst.template ReinterpretCast<uint16_t>();
        auto gateHighWords = gateHighMask_.template ReinterpretCast<uint32_t>();
        auto upLowWords = upLowMask_.template ReinterpretCast<uint32_t>();
        auto highWords = upHighMask_.template ReinterpretCast<uint32_t>();
        // Infinity is only an internal branch sentinel. The following FP32
        // clamp turns it into the finite user limit before any arithmetic.
        // Unlike max-finite BF16, this remains outside every finite limit.
        constexpr uint16_t positiveInfinityBf16 = 0x7f80U;
        constexpr uint16_t negativeInfinityBf16 = 0xff80U;
        for (uint16_t row = 0; row < rows; ++row) {
            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t low = gatePlane ? gateHighWords.GetValue((maskRow + row) * 8 + group)
                                               : upLowWords.GetValue((maskRow + row) * 8 + group);
                const uint32_t high = gatePlane ? 0U : highWords.GetValue((maskRow + row) * 8 + group);
                if ((low | high) == 0U) {
                    continue;
                }
                for (uint32_t bit = 0; bit < 32; ++bit) {
                    const uint32_t index = row * columns + group * 32 + bit;
                    if ((low & (1U << bit)) != 0U) {
                        dstBits.SetValue(index, gatePlane ? positiveInfinityBf16 : negativeInfinityBf16);
                    } else if ((high & (1U << bit)) != 0U) {
                        dstBits.SetValue(index, positiveInfinityBf16);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyPlaneTileToUb(const AscendC::GlobalTensor<bfloat16_t>& srcPlane, int64_t index,
                                             uint32_t singleM, uint32_t singleN, uint32_t planeWidth,
                                             AscendC::LocalTensor<bfloat16_t>& dstUb)
    {
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(singleM),
                                              static_cast<uint32_t>(singleN * sizeof(bfloat16_t)),
                                              static_cast<uint32_t>((planeWidth - singleN) * sizeof(bfloat16_t)), 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> padParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstUb, srcPlane[index], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline void ApplyPackedWeight(AscendC::LocalTensor<bfloat16_t>& tile, uint16_t mSize, uint16_t nSize,
                                             uint32_t rowStride, uint32_t mLoc)
    {
        auto* values = (__ubuf__ bfloat16_t*)tile.GetPhyAddr();
        auto* weights = (__ubuf__ bfloat16_t*)packedWeightUb_.GetPhyAddr();
        constexpr uint16_t lanes = 256U / sizeof(float);
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                AscendC::Reg::RegTensor<bfloat16_t> pBf;
                AscendC::Reg::RegTensor<float> pFp;
                auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(pBf, weights + row);
                AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(pFp, pBf, all);
                uint32_t remaining = nSize;
                for (uint16_t col = 0; col < nSize; col += lanes) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::RegTensor<bfloat16_t> valueBf, outBf;
                    AscendC::Reg::RegTensor<float> valueFp, outFp;
                    const uint32_t offset = static_cast<uint32_t>(row) * rowStride + col;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(valueBf,
                                                                                                values + offset);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(valueFp, valueBf, mask);
                    AscendC::Reg::Mul(outFp, valueFp, pFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(outBf, outFp, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(values + offset, outBf,
                                                                                               mask);
                }
            }
        }
    }

    // Reduce one BF16 dHp*H tile to one FP32 partial per row.  The product
    // tile is dead afterwards, so its first FP32 slot in every row stores the
    // result without increasing the three-tile UB footprint.
    __aicore__ inline void ComputeDpPartials(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        constexpr uint16_t kBf16PerReg = 256U / sizeof(bfloat16_t);
        const uint16_t repeatsPerRow =
            static_cast<uint16_t>(Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kBf16PerReg)));
        __ubuf__ bfloat16_t* products = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ float* partials = reinterpret_cast<__ubuf__ float*>(products);

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                AscendC::Reg::RegTensor<float> accumulated;
                AscendC::Reg::MaskReg oneLane = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
                AscendC::Reg::Duplicate(accumulated, 0.0F, oneLane);
                uint32_t remaining = nSize;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    const uint32_t chunk = remaining > kBf16PerReg ? kBf16PerReg : remaining;
                    uint16_t partRemaining = static_cast<uint16_t>(chunk);
                    AscendC::Reg::RegTensor<bfloat16_t> packed0, packed1;
                    AscendC::Reg::RegTensor<float> value0, value1;
                    const uint32_t offset = row * rowStride + repeat * kBf16PerReg;
                    AscendC::Reg::DataCopy(packed0, products + offset);
                    AscendC::Reg::Interleave(packed0, packed1, packed0, packed1);
                    const uint16_t partLanesMax = static_cast<uint16_t>(kBf16PerReg / 2);
                    // bisheng vectorizes this reduction loop and requires the
                    // induction variable and every loop-condition operand to be
                    // uint16_t.  Precompute the trip count so the condition is
                    // only `part < partCount`.
                    const uint16_t partCount = partRemaining > partLanesMax ? 2U : 1U;
                    for (uint16_t part = 0; part < partCount; ++part) {
                        const uint16_t partLanes = partRemaining > partLanesMax ? partLanesMax : partRemaining;
                        uint32_t partMaskLanes = partLanes;
                        AscendC::Reg::MaskReg partMask = AscendC::Reg::UpdateMask<float>(partMaskLanes);
                        auto packed = part == 0 ? packed0 : packed1;
                        auto value = part == 0 ? value0 : value1;
                        AscendC::Reg::RegTensor<float> rowSum;
                        AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(value, packed, partMask);
                        AscendC::Reg::ReduceSum(rowSum, value, partMask);
                        AscendC::Reg::Add(accumulated, accumulated, rowSum, oneLane);
                        partRemaining -= partLanes;
                    }
                    remaining -= chunk;
                }
                AscendC::Reg::DataCopy(partials + static_cast<uint32_t>(row) * (rowStride / 2), accumulated, oneLane);
            }
        }
    }

    // Produce BF16 H.  The caller forms dHp*H and then scales H in-place to
    // Hp.  The product matrix is reduced in a later, single-owner phase.
    __aicore__ inline void ComputeForwardAux(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        constexpr uint16_t kElementsPerRepeat = 128U;
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        // The disabled limit (FLT_MAX) makes the clamps numerical no-ops;
        // skip them entirely so the unlimited path pays nothing extra.
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        __ubuf__ bfloat16_t* gate = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ bfloat16_t* up = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();
        __ubuf__ bfloat16_t* hp = up;

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                uint32_t remaining = nSize;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
                    AscendC::Reg::RegTensor<float> oneReg, limitReg, negLimitReg;
                    if (limitEnabled) {
                        AscendC::Reg::Duplicate(oneReg, kOne, mask);
                        AscendC::Reg::Muls(limitReg, oneReg, limit, mask);
                        AscendC::Reg::Muls(negLimitReg, oneReg, -limit, mask);
                    }
                    AscendC::Reg::RegTensor<bfloat16_t> g0, g1, u0, u1;
                    AscendC::Reg::RegTensor<float> gf0, gf1, uf0, uf1;
                    const uint32_t offset = row * rowStride + repeat * kElementsPerRepeat;
                    AscendC::Reg::DataCopy(g0, gate + offset);
                    AscendC::Reg::DataCopy(u0, up + offset);
                    AscendC::Reg::Interleave(g0, g1, g0, g1);
                    AscendC::Reg::Interleave(u0, u1, u0, u1);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gf0, g0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gf1, g1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(uf0, u0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(uf1, u1, mask);
                    constexpr uint16_t kParts = 2U;
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto gf = part == 0 ? gf0 : gf1;
                        auto uf = part == 0 ? uf0 : uf1;
                        AscendC::Reg::RegTensor<float> neg, ex, den, silu;
                        AscendC::Reg::RegTensor<float> h;
                        AscendC::Reg::RegTensor<bfloat16_t> hOut;
                        // dsv4 swiglu_limit clamp, mirroring the forward
                        // epilogue so the recomputed H matches forward.
                        // Skipped entirely when the limit is disabled.
                        if (limitEnabled) {
                            AscendC::Reg::Min(gf, gf, limitReg, mask);
                            AscendC::Reg::Max(uf, uf, negLimitReg, mask);
                            AscendC::Reg::Min(uf, uf, limitReg, mask);
                        }
                        AscendC::Reg::Muls(neg, gf, -kOne, mask);
                        AscendC::Reg::Exp(ex, neg, mask);
                        AscendC::Reg::Adds(den, ex, kOne, mask);
                        AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gf, den, mask);
                        // Match forward: BF16(SiLU_FP32(gate) * Up_FP32).
                        AscendC::Reg::Mul(h, silu, uf, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(hOut, h, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            hp + offset + part * kPartSize, hOut, mask);
                    }
                }
            }
        }
    }

    // Per-row / per-repeat computation of:
    //   s     = sigmoid(gate)
    //   dUp   = dH * gate * s
    //   dGate = dH * up * s * (1 + gate * (1 - s))
    // dGate accumulates into l0cOutUbFirst_ (overwriting dH), dUp into
    // l0cOutUbSecond_ (overwriting gate). gate/up row-chunks are read before
    // the same offset is overwritten.
    __aicore__ inline void ComputeSwigluGrad(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        if (params_.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8 && params_.gateHighGmAddr != nullptr) {
            // MXFP8 can move an untruncated value across the clipping
            // boundary. Clamp the reconstructed value, but derive the
            // derivative switch exclusively from the forward bit planes.
            ComputeSwigluGradImpl<true>(mSize, nSize, rowStride);
        } else {
            ComputeSwigluGradImpl<false>(mSize, nSize, rowStride);
        }
    }

    template <bool UseSavedClipMask>
    __aicore__ inline void ComputeSwigluGradImpl(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        uint16_t vectorRows = mSize;
        uint16_t vectorColumns = nSize;
        uint32_t vectorStride = rowStride;
        constexpr uint16_t kElementsPerRepeat = 128U;
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(vectorColumns), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        // The disabled limit (FLT_MAX) makes the clamps numerical no-ops;
        // skip them (and the range indicators) entirely so the unlimited
        // path pays nothing extra.
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        __ubuf__ bfloat16_t* dH = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t* gate = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ bfloat16_t* up = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();
        __ubuf__ bfloat16_t* dGate = dH;  // reuse dH tile
        __ubuf__ bfloat16_t* dUp = gate;  // reuse gate tile

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < vectorRows; ++row) {
                uint32_t remaining = vectorColumns;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
                    AscendC::Reg::RegTensor<bfloat16_t> dH0, dH1, gate0, gate1, up0, up1;
                    AscendC::Reg::RegTensor<float> dHFp0, dHFp1, gateFp0, gateFp1, upFp0, upFp1;
                    const uint32_t offset = row * vectorStride + repeat * kElementsPerRepeat;
                    AscendC::Reg::DataCopy(dH0, dH + offset);
                    AscendC::Reg::DataCopy(gate0, gate + offset);
                    AscendC::Reg::DataCopy(up0, up + offset);
                    AscendC::Reg::Interleave(dH0, dH1, dH0, dH1);
                    AscendC::Reg::Interleave(gate0, gate1, gate0, gate1);
                    AscendC::Reg::Interleave(up0, up1, up0, up1);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHFp0, dH0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHFp1, dH1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp0, gate0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp1, gate1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp0, up0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp1, up1, mask);

                    constexpr uint16_t kParts = 2U;
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto dHFp = part == 0 ? dHFp0 : dHFp1;
                        auto gateFp = part == 0 ? gateFp0 : gateFp1;
                        auto upFp = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> oneFp, neg, exp, denom, sig, silu;
                        AscendC::Reg::RegTensor<float> gradSwish;
                        AscendC::Reg::RegTensor<float> oneMinusS, inner, term;
                        AscendC::Reg::RegTensor<float> dUpFp, dGateFp;
                        AscendC::Reg::RegTensor<float> gateC, upC, gateInd, upInd, diff;
                        AscendC::Reg::RegTensor<float> limitReg, negLimitReg;
                        AscendC::Reg::RegTensor<bfloat16_t> dUpOut, dGateOut;
                        // sigmoid(gate) = 1 / (1 + exp(-gate)).  Reg::Div only
                        // takes tensor operands, so materialize a constant 1.
                        AscendC::Reg::Duplicate(oneFp, kOne, mask);
                        // dsv4 swiglu_limit clamp, mirroring the eager path:
                        // gateC = min(gate, L); upC = clamp(up, -L, L).
                        if (limitEnabled) {
                            AscendC::Reg::Muls(limitReg, oneFp, limit, mask);
                            AscendC::Reg::Muls(negLimitReg, oneFp, -limit, mask);
                            AscendC::Reg::Min(gateC, gateFp, limitReg, mask);
                            AscendC::Reg::Max(upC, upFp, negLimitReg, mask);
                            AscendC::Reg::Min(upC, upC, limitReg, mask);
                            if constexpr (UseSavedClipMask) {
                                // The scalar post-pass below applies the exact
                                // forward mask. Keep all lanes enabled here so
                                // an unmasked decoded value crossing L cannot
                                // invent a new zero-gradient branch.
                                gateInd = oneFp;
                                upInd = oneFp;
                            } else {
                                // Range indicators (1 = inside, 0 = clamped):
                                // BF16/none use the exact raw GMM1 value.
                                AscendC::Reg::Sub(diff, gateFp, gateC, mask);
                                AscendC::Reg::Abs(diff, diff, mask);
                                AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                                AscendC::Reg::Min(diff, diff, oneFp, mask);
                                AscendC::Reg::Sub(gateInd, oneFp, diff, mask);
                                AscendC::Reg::Sub(diff, upFp, upC, mask);
                                AscendC::Reg::Abs(diff, diff, mask);
                                AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                                AscendC::Reg::Min(diff, diff, oneFp, mask);
                                AscendC::Reg::Sub(upInd, oneFp, diff, mask);
                            }
                        } else {
                            // No limit: the clamp is the identity, so alias
                            // the raw registers (no instruction) and let the
                            // gradients flow through unmodified.
                            gateC = gateFp;
                            upC = upFp;
                        }
                        // sigmoid / silu / silu' use the CLAMPED gate.
                        AscendC::Reg::Muls(neg, gateC, -kOne, mask);
                        AscendC::Reg::Exp(exp, neg, mask);
                        AscendC::Reg::Adds(denom, exp, kOne, mask);
                        AscendC::Reg::Div<float, &kBackwardDivMode>(sig, oneFp, denom, mask);
                        // Forward does not materialize a BF16 SiLU tensor.
                        AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gateC, denom, mask);
                        AscendC::Reg::Mul(dUpFp, dHFp, silu, mask);
                        if (limitEnabled) {
                            AscendC::Reg::Mul(dUpFp, dUpFp, upInd, mask);
                        }
                        // The fused derivative keeps dS=dH*up in FP32, just
                        // as the forward SiLU/Up product stays in FP32.
                        AscendC::Reg::Mul(gradSwish, dHFp, upC, mask);
                        // dGate = dS * sig * (1 + gateC * (1 - sig)),
                        // zeroed where gate was clamped.
                        AscendC::Reg::Muls(oneMinusS, sig, -kOne, mask);
                        AscendC::Reg::Adds(oneMinusS, oneMinusS, kOne, mask);
                        AscendC::Reg::Mul(inner, gateC, oneMinusS, mask);
                        AscendC::Reg::Adds(inner, inner, kOne, mask);
                        AscendC::Reg::Mul(term, sig, inner, mask);
                        AscendC::Reg::Mul(dGateFp, gradSwish, term, mask);
                        if (limitEnabled) {
                            AscendC::Reg::Mul(dGateFp, dGateFp, gateInd, mask);
                        }
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dUpOut, dUpFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dGateOut, dGateFp, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            dUp + offset + part * kPartSize, dUpOut, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            dGate + offset + part * kPartSize, dGateOut, mask);
                    }
                }
            }
        }
        if constexpr (UseSavedClipMask) {
            // Vector stores must retire before scalar clears masked gradient
            // lanes. Order the scalar stores against both possible consumers:
            // the next MTE2 reuse and the current MTE3 dZ writeout.
            AscendC::SyncFunc<AscendC::HardEvent::V_S>();
            ApplySavedClipGradientMask(mSize, nSize, rowStride);
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        }
    }

    template <bool UseSavedClipMask>
    __aicore__ inline void ComputeFusedFullTile()
    {
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        auto* dGate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        auto* dUp = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        auto* hp = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();
        auto* weights = (__ubuf__ bfloat16_t*)packedWeightUb_.GetPhyAddr();
        auto* partials = (__ubuf__ float*)fusedDpUb_.GetPhyAddr();
        constexpr uint16_t rows = kTileRows, columns = kTileColumns;
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < rows; ++row) {
                auto mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                auto oneLane = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
                AscendC::Reg::RegTensor<float> accumulated, pFp;
                AscendC::Reg::RegTensor<bfloat16_t> pBf;
                AscendC::Reg::Duplicate(accumulated, 0.0F, oneLane);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(pBf, weights + row);
                AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(pFp, pBf, mask);
                for (uint16_t col = 0; col < columns; col += 64U) {
                    const uint32_t offset = static_cast<uint32_t>(row) * kTileColumns + col;
                    AscendC::Reg::RegTensor<bfloat16_t> gBf, uBf, dHpBf;
                    AscendC::Reg::RegTensor<float> gateFp, upFp, dHpFp, dHFp;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gBf, dUp + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(uBf, hp + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(dHpBf, dGate + offset);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp, gBf, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp, uBf, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHpFp, dHpBf, mask);
                    AscendC::Reg::RegTensor<float> oneFp, neg, exp, denom, sig;
                    AscendC::Reg::RegTensor<float> gradSwish;
                    AscendC::Reg::RegTensor<float> oneMinusS, inner, term;
                    AscendC::Reg::RegTensor<float> dUpFp, dGateFp;
                    AscendC::Reg::RegTensor<float> gateC, upC, gateInd, upInd, diff;
                    AscendC::Reg::RegTensor<float> limitReg, negLimitReg;
                    AscendC::Reg::RegTensor<bfloat16_t> dUpOut, dGateOut;
                    // sigmoid(gate) = 1 / (1 + exp(-gate)).  Reg::Div only
                    // takes tensor operands, so materialize a constant 1.
                    AscendC::Reg::Duplicate(oneFp, kOne, mask);
                    // dsv4 swiglu_limit clamp, mirroring the eager path:
                    // gateC = min(gate, L); upC = clamp(up, -L, L).
                    if (limitEnabled) {
                        AscendC::Reg::Muls(limitReg, oneFp, limit, mask);
                        AscendC::Reg::Muls(negLimitReg, oneFp, -limit, mask);
                        AscendC::Reg::Min(gateC, gateFp, limitReg, mask);
                        AscendC::Reg::Max(upC, upFp, negLimitReg, mask);
                        AscendC::Reg::Min(upC, upC, limitReg, mask);
                        if constexpr (UseSavedClipMask) {
                            // The scalar post-pass below applies the exact
                            // forward mask. Keep all lanes enabled here so
                            // an unmasked decoded value crossing L cannot
                            // invent a new zero-gradient branch.
                            gateInd = oneFp;
                            upInd = oneFp;
                        } else {
                            // Range indicators (1 = inside, 0 = clamped):
                            // BF16/none use the exact raw GMM1 value.
                            AscendC::Reg::Sub(diff, gateFp, gateC, mask);
                            AscendC::Reg::Abs(diff, diff, mask);
                            AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                            AscendC::Reg::Min(diff, diff, oneFp, mask);
                            AscendC::Reg::Sub(gateInd, oneFp, diff, mask);
                            AscendC::Reg::Sub(diff, upFp, upC, mask);
                            AscendC::Reg::Abs(diff, diff, mask);
                            AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                            AscendC::Reg::Min(diff, diff, oneFp, mask);
                            AscendC::Reg::Sub(upInd, oneFp, diff, mask);
                        }
                    } else {
                        // No limit: the clamp is the identity, so alias
                        // the raw registers (no instruction) and let the
                        // gradients flow through unmodified.
                        gateC = gateFp;
                        upC = upFp;
                    }
                    // sigmoid / silu / silu' use the CLAMPED gate.
                    AscendC::Reg::Muls(neg, gateC, -kOne, mask);
                    AscendC::Reg::Exp(exp, neg, mask);
                    AscendC::Reg::Adds(denom, exp, kOne, mask);
                    // H reconstruction uses gate/den, while the derivative
                    // retains (1/den)*gate. These are separate accurate Divs.
                    AscendC::Reg::RegTensor<float> auxSilu, hFp, hRounded;
                    AscendC::Reg::RegTensor<float> hpFp, productFp, productRounded, rowSum, dhFp;
                    AscendC::Reg::RegTensor<bfloat16_t> hBf, hpBf, productBf, dhBf;
                    AscendC::Reg::Div<float, &kBackwardDivMode>(auxSilu, gateC, denom, mask);
                    // Same H as forward, with no intermediate BF16 SiLU.
                    AscendC::Reg::Mul(hFp, auxSilu, upC, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(hBf, hFp, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(hRounded, hBf, mask);
                    AscendC::Reg::Mul(productFp, dHpFp, hRounded, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(productBf, productFp, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(productRounded, productBf, mask);
                    AscendC::Reg::ReduceSum(rowSum, productRounded, mask);
                    AscendC::Reg::Add(accumulated, accumulated, rowSum, oneLane);
                    AscendC::Reg::Mul(hpFp, hRounded, pFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(hpBf, hpFp, mask);
                    AscendC::Reg::Mul(dhFp, dHpFp, pFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dhBf, dhFp, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHFp, dhBf, mask);

                    AscendC::Reg::Div<float, &kBackwardDivMode>(sig, oneFp, denom, mask);
                    // Reuse the FP32 SiLU used to reconstruct forward H.
                    AscendC::Reg::Mul(dUpFp, dHFp, auxSilu, mask);
                    if (limitEnabled) {
                        AscendC::Reg::Mul(dUpFp, dUpFp, upInd, mask);
                    }
                    // No standalone BF16 dS tensor in the fused graph.
                    AscendC::Reg::Mul(gradSwish, dHFp, upC, mask);
                    // dGate = dS * sig * (1 + gateC * (1 - sig)),
                    // zeroed where gate was clamped.
                    AscendC::Reg::Muls(oneMinusS, sig, -kOne, mask);
                    AscendC::Reg::Adds(oneMinusS, oneMinusS, kOne, mask);
                    AscendC::Reg::Mul(inner, gateC, oneMinusS, mask);
                    AscendC::Reg::Adds(inner, inner, kOne, mask);
                    AscendC::Reg::Mul(term, sig, inner, mask);
                    AscendC::Reg::Mul(dGateFp, gradSwish, term, mask);
                    if (limitEnabled) {
                        AscendC::Reg::Mul(dGateFp, dGateFp, gateInd, mask);
                    }
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dUpOut, dUpFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dGateOut, dGateFp, mask);

                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dGate + offset, dGateOut,
                                                                                               mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dUp + offset, dUpOut,
                                                                                               mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(hp + offset, hpBf, mask);
                }
                // Exactly the old row's four 64-element reductions, in order.
                AscendC::Reg::DataCopy(partials + row * 8U, accumulated, oneLane);
            }
        }
        if constexpr (UseSavedClipMask) {
            AscendC::SyncFunc<AscendC::HardEvent::V_S>();
            ApplySavedClipGradientMask(kTileRows, kTileColumns, kTileColumns);
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        }
    }

    __aicore__ inline void RunFusedFullTile(uint32_t rows, uint32_t columns, uint32_t mLoc, uint32_t nLoc)
    {
        static_assert(kMxScratchEnd + kTileRows * 32U <= kWeightOffset,
                      "Fused partials overlap neither masks nor weights");
        if (params_.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8 && params_.gateHighGmAddr != nullptr) {
            ComputeFusedFullTile<true>();
        } else {
            ComputeFusedFullTile<false>();
        }
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyExtParams partialCopy{static_cast<uint16_t>(rows), sizeof(float), 0U,
                                               static_cast<uint32_t>((dpPartialTiles_ - 1) * sizeof(float)), 0U};
        AscendC::DataCopyPad(dpPartialsGm_[(rowBase_ + mLoc) * dpPartialTiles_ + nLoc / BACKWARD_DP_PARTIAL_TILE_N],
                             fusedDpUb_, partialCopy);
        AscendC::DataCopyExtParams hpCopy{static_cast<uint16_t>(rows),
                                          static_cast<uint32_t>(columns * sizeof(bfloat16_t)), 0U,
                                          static_cast<uint32_t>((outputF_ - columns) * sizeof(bfloat16_t)), 0U};
        AscendC::DataCopyPad(hpGm_[(rowBase_ + mLoc) * outputF_ + nLoc], extraA_, hpCopy);
        AscendC::DataCopyExtParams gradCopy{static_cast<uint16_t>(rows),
                                            static_cast<uint32_t>(columns * sizeof(bfloat16_t)), 0U,
                                            static_cast<uint32_t>((outputN_ - columns) * sizeof(bfloat16_t)), 0U};
        AscendC::DataCopyPad(gradZGm_[(rowBase_ + mLoc) * outputN_ + nLoc], l0cOutUbFirst_, gradCopy);
        AscendC::DataCopyPad(gradZGm_[(rowBase_ + mLoc) * outputN_ + outputF_ + nLoc], l0cOutUbSecond_, gradCopy);
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void ApplySavedClipGradientMask(uint16_t rows, uint32_t columns, uint32_t rowStride)
    {
        auto dGateBits = l0cOutUbFirst_.template ReinterpretCast<uint16_t>();
        auto dUpBits = l0cOutUbSecond_.template ReinterpretCast<uint16_t>();
        auto gateHighWords = gateHighMask_.template ReinterpretCast<uint32_t>();
        auto upLowWords = upLowMask_.template ReinterpretCast<uint32_t>();
        auto upHighWords = upHighMask_.template ReinterpretCast<uint32_t>();
        const uint32_t groups = columns / 32;
        for (uint16_t row = 0; row < rows; ++row) {
            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t gateHigh = gateHighWords.GetValue(row * 8 + group);
                const uint32_t upClipped = upLowWords.GetValue(row * 8 + group) | upHighWords.GetValue(row * 8 + group);
                if ((gateHigh | upClipped) == 0U) {
                    continue;
                }
                for (uint32_t bit = 0; bit < 32; ++bit) {
                    const uint32_t index = row * rowStride + group * 32 + bit;
                    if ((gateHigh & (1U << bit)) != 0U) {
                        dGateBits.SetValue(index, 0U);
                    }
                    if ((upClipped & (1U << bit)) != 0U) {
                        dUpBits.SetValue(index, 0U);
                    }
                }
            }
        }
    }

    // H = SiLU(gate) * up, written to extraA_ (UB offset 2*kTileBytes).
    __aicore__ inline void ComputeSwigluRecompute(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        constexpr uint16_t kElementsPerRepeat = 256U / sizeof(bfloat16_t);
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        // The disabled limit (FLT_MAX) makes the clamps numerical no-ops;
        // skip them entirely so the unlimited path pays nothing extra.
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        __ubuf__ bfloat16_t* gate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t* up = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ bfloat16_t* h = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                uint32_t remaining = nSize;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
                    AscendC::Reg::RegTensor<float> oneReg, limitReg, negLimitReg;
                    if (limitEnabled) {
                        AscendC::Reg::Duplicate(oneReg, kOne, mask);
                        AscendC::Reg::Muls(limitReg, oneReg, limit, mask);
                        AscendC::Reg::Muls(negLimitReg, oneReg, -limit, mask);
                    }
                    AscendC::Reg::RegTensor<bfloat16_t> gate0, gate1, up0, up1;
                    AscendC::Reg::RegTensor<float> gateFp0, gateFp1, upFp0, upFp1;
                    const uint32_t offset = row * rowStride + repeat * kElementsPerRepeat;
                    AscendC::Reg::DataCopy(gate0, gate + offset);
                    AscendC::Reg::DataCopy(up0, up + offset);
                    AscendC::Reg::Interleave(gate0, gate1, gate0, gate1);
                    AscendC::Reg::Interleave(up0, up1, up0, up1);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp0, gate0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp1, gate1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp0, up0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp1, up1, mask);

                    constexpr uint16_t kParts = sizeof(float) / sizeof(bfloat16_t);
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto gateFp = part == 0 ? gateFp0 : gateFp1;
                        auto upFp = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> neg, exp, denom, silu, result;
                        AscendC::Reg::RegTensor<bfloat16_t> output;
                        // H = SiLU(gate) * up; SiLU(gate) = gate * sigma(gate).
                        // Apply the dsv4 swiglu_limit clamp first so the
                        // recomputed H matches the forward epilogue.
                        // Skipped entirely when the limit is disabled.
                        if (limitEnabled) {
                            AscendC::Reg::Min(gateFp, gateFp, limitReg, mask);
                            AscendC::Reg::Max(upFp, upFp, negLimitReg, mask);
                            AscendC::Reg::Min(upFp, upFp, limitReg, mask);
                        }
                        AscendC::Reg::Muls(neg, gateFp, -kOne, mask);
                        AscendC::Reg::Exp(exp, neg, mask);
                        AscendC::Reg::Adds(denom, exp, kOne, mask);
                        AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gateFp, denom, mask);
                        AscendC::Reg::Mul(result, silu, upFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(output, result, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            h + offset + part * kPartSize, output, mask);
                    }
                }
            }
        }
    }

    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, kTileElements};
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond_{AscendC::TPosition::VECIN, kTileBytes, kTileElements};
    AscendC::LocalTensor<bfloat16_t> extraA_{AscendC::TPosition::VECIN, 2 * kTileBytes, kTileElements};
    AscendC::LocalTensor<bfloat16_t> packedWeightUb_{AscendC::TPosition::VECIN, kWeightOffset, kTileRows};
    AscendC::LocalTensor<float> fusedDpUb_{AscendC::TPosition::VECCALC, kMxScratchEnd, kTileRows * 8U};
    AscendC::LocalTensor<int8_t> mxData_{AscendC::TPosition::VECIN, kMxDataOffset, kMxDataElements};
    AscendC::LocalTensor<int8_t> mxScale_{AscendC::TPosition::VECIN, kMxScaleOffset, kMxScaleBytes};
    AscendC::LocalTensor<bfloat16_t> mxScaleBf16_{AscendC::TPosition::VECCALC, kMxScaleBf16Offset,
                                                  kMxScaleBf16Elements};
    AscendC::LocalTensor<float> mxScaleFp32_{AscendC::TPosition::VECCALC, kMxScaleFp32Offset, kMxScaleFp32Elements};
    AscendC::LocalTensor<float> mxDecoded_{AscendC::TPosition::VECCALC, kMxDecodedOffset, kMxDecodedElements};
    AscendC::LocalTensor<int8_t> gateHighMask_{AscendC::TPosition::VECIN, kGateHighOffset, kClipMaskBytes};
    AscendC::LocalTensor<int8_t> upLowMask_{AscendC::TPosition::VECIN, kUpLowOffset, kClipMaskBytes};
    AscendC::LocalTensor<int8_t> upHighMask_{AscendC::TPosition::VECIN, kUpHighOffset, kClipMaskBytes};

    AscendC::GlobalTensor<bfloat16_t> gatePlaneGm_;
    AscendC::GlobalTensor<bfloat16_t> upPlaneGm_;
    AscendC::GlobalTensor<int8_t> gateMxFp8Gm_;
    AscendC::GlobalTensor<int8_t> upMxFp8Gm_;
    AscendC::GlobalTensor<int8_t> gateScaleGm_;
    AscendC::GlobalTensor<int8_t> upScaleGm_;
    AscendC::GlobalTensor<int8_t> gateHighGm_;
    AscendC::GlobalTensor<int8_t> upLowGm_;
    AscendC::GlobalTensor<int8_t> upHighGm_;
    AscendC::GlobalTensor<bfloat16_t> gradZGm_;
    AscendC::GlobalTensor<bfloat16_t> hpGm_;
    AscendC::GlobalTensor<float> dpPartialsGm_;
    AscendC::GlobalTensor<bfloat16_t> packedTopkWeightsGm_;
    int64_t rowBase_{0};
    Params params_{};
    int64_t outputN_{0};
    int64_t outputF_{0};
    int64_t dpPartialTiles_{0};
};

}  // namespace MegaMoeBF16Impl

#endif  // DEEPEP_BLOCK_EPILOGUE_SWIGLU_BACKWARD_BF16_HPP
