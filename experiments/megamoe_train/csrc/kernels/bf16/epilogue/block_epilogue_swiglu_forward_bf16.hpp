/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * BF16 SwiGLU epilogue for the DeepEP MegaMoe pipeline. GMM1 writes two
 * BF16 tiles to UB. The vector side evaluates SwiGLU in FP32 and stores the
 * result as BF16. Variants that apply route weights before GMM2 then multiply
 * each BF16 row by its dispatched BF16 route weight read directly from GM.
 */
#ifndef DEEPEP_BLOCK_EPILOGUE_SWIGLU_FORWARD_BF16_HPP
#define DEEPEP_BLOCK_EPILOGUE_SWIGLU_FORWARD_BF16_HPP

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"
#include "layout/mega_moe_context.hpp"
#include "epilogue/block_saved_gate_up_mxfp8.hpp"

namespace MegaMoeBF16Impl {

static constexpr AscendC::Reg::DivSpecificMode kDivMode = {AscendC::Reg::MaskMergeMode::ZEROING, true};
static constexpr AscendC::Reg::CastTrait kCastBF16ToFP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait kCastFP32ToBF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename L0TileShape_, bool ApplyTopkWeight_>
class BlockEpilogueSwigluForwardBF16 {
public:
    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR groupFlagListGmAddr{nullptr};
        GM_ADDR routedTopkWeightGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        // dsv4 swiglu_limit clamp; kMegaMoeSwigluLimitDisabled disables it.
        float swigluLimit{kMegaMoeSwigluLimitDisabled};
        GM_ADDR savedGateGmAddr{nullptr};
        GM_ADDR savedUpGmAddr{nullptr};
        GM_ADDR savedGateScaleGmAddr{nullptr};
        GM_ADDR savedUpScaleGmAddr{nullptr};
        GM_ADDR savedGateHighGmAddr{nullptr};
        GM_ADDR savedUpLowGmAddr{nullptr};
        GM_ADDR savedUpHighGmAddr{nullptr};
    };

    using Params = Arguments;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    __aicore__ inline void Init(Params const& params)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            params_ = params;
        }
    }

    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            savedBaseOffset_ = Get<0>(baseOffset);
            outputGlobal_.SetGlobalBuffer((__gm__ bfloat16_t*)params_.yGmAddr + Get<0>(baseOffset));
            // Gate/up use the same packed-row x F coordinates as GMM2 input.
            if (params_.savedGateGmAddr != nullptr) {
                savedGateGlobal_.SetGlobalBuffer((__gm__ bfloat16_t*)params_.savedGateGmAddr + Get<0>(baseOffset));
                savedUpGlobal_.SetGlobalBuffer((__gm__ bfloat16_t*)params_.savedUpGmAddr + Get<0>(baseOffset));
                // Persisted activations are consumed by a later backward.
                savedGateGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                savedUpGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            groupFlagListGmAddr_ = (__gm__ int32_t*)params_.groupFlagListGmAddr + Get<2>(baseOffset) * INT_CACHELINE;
            if constexpr (ApplyTopkWeight_) {
                routedTopkWeightGlobal_.SetGlobalBuffer((__gm__ bfloat16_t*)params_.routedTopkWeightGmAddr +
                                                        Get<3>(baseOffset));
                routedTopkWeightGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
        }
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        outputN_ = Get<1>(problemShape);
    }

    template <uint32_t SaveMode>
    __aicore__ inline void Run(const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }

        const uint32_t singleM = Get<0>(blockShape);
        const uint32_t singleN = Get<1>(blockShape);
        if (singleM == 0 || singleN == 0) {
            return;
        }

        const uint32_t tileRow = static_cast<uint32_t>(Get<0>(blockCoord) / outputN_);
        if constexpr (SaveMode != MegaMoeBF16Training::kSaveNone) {
            SaveGateUp<SaveMode>(singleM, singleN, Get<0>(blockCoord));
        }
#if defined(DEEPEP_BF16_M192_FORWARD)
        ComputeSwigluDirect(singleM, singleN, tileRow);
#else
        ComputeSwiglu(singleM, singleN);
        if constexpr (ApplyTopkWeight_) {
            ApplyTopkWeight(singleM, singleN, tileRow);
        }
#endif

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(singleM),
                                              static_cast<uint32_t>(singleN * sizeof(bfloat16_t)), 0,
                                              static_cast<uint32_t>((outputN_ - singleN) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(outputGlobal_[Get<0>(blockCoord)], l0cOutUbFirst_, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::AtomicAdd(groupFlagListGmAddr_, 1);
    }

    __aicore__ inline void LoadRawTile(GM_ADDR gate, GM_ADDR up, uint64_t expertOffset, uint16_t mSize, uint16_t nSize,
                                       int64_t offset)
    {
        AscendC::GlobalTensor<bfloat16_t> rawGate, rawUp;
        rawGate.SetGlobalBuffer((__gm__ bfloat16_t*)gate + expertOffset);
        rawUp.SetGlobalBuffer((__gm__ bfloat16_t*)up + expertOffset);
        rawGate.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        rawUp.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::DataCopyExtParams copyParams{mSize, static_cast<uint32_t>(nSize * sizeof(bfloat16_t)),
                                              static_cast<uint32_t>((outputN_ - nSize) * sizeof(bfloat16_t)), 0, 0};
        AscendC::DataCopyPad(l0cOutUbFirst_, rawGate[offset], copyParams, {});
        AscendC::DataCopyPad(l0cOutUbSecond_, rawUp[offset], copyParams, {});
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

private:
    static constexpr uint32_t kTileRows = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t kTileColumns = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t kTileElements = kTileRows * kTileColumns;
    static_assert(2U * kTileElements * sizeof(bfloat16_t) <= Catlass::Arch::Ascend950::UB_SIZE,
                  "The BF16 SwiGLU input tiles must fit in A5 UB");

    template <uint32_t SaveMode>
    __aicore__ inline void SaveGateUp(uint16_t mSize, uint16_t nSize, int64_t offset)
    {
        if constexpr (SaveMode == MegaMoeBF16Training::kSaveMxFp8) {
            quantSave_.Save(l0cOutUbFirst_, l0cOutUbSecond_, params_.savedGateGmAddr, params_.savedUpGmAddr,
                            params_.savedGateScaleGmAddr, params_.savedUpScaleGmAddr, params_.savedGateHighGmAddr,
                            params_.savedUpLowGmAddr, params_.savedUpHighGmAddr, params_.swigluLimit, mSize, nSize,
                            outputN_, savedBaseOffset_ + offset);
            return;
        }
        // BF16 mode retains the validated serial save path with or without clipping.
        // WaitForCube in the caller has completed both Fixpipe input tiles.
        // Save raw BF16 gate/up before clipping, activation, or route weighting.
        // DataCopyPad uses the same aligned UB row stride as ComputeSwiglu.
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyExtParams copyParams{mSize, static_cast<uint32_t>(nSize * sizeof(bfloat16_t)), 0,
                                              static_cast<uint32_t>((outputN_ - nSize) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(savedGateGlobal_[offset], l0cOutUbFirst_, copyParams);
        AscendC::DataCopyPad(savedUpGlobal_[offset], l0cOutUbSecond_, copyParams);
        // MTE3 must finish reading both tiles before vector work overwrites UB.
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void ComputeSwigluDirect(uint16_t mSize, uint16_t nSize, uint32_t tileRow)
    {
        constexpr uint16_t lanes = 256 / sizeof(float);
        const uint32_t stride = Ops::Base::CeilAlign(uint32_t(nSize), uint32_t(16));
        const bool clip = params_.swigluLimit != kMegaMoeSwigluLimitDisabled;
        const float limit = params_.swigluLimit;
        auto* gate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        auto* up = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        // Place weights beyond every MXFP8 data/scale/clamp-mask buffer: Save
        // may still have MTE3 readers. No alias and no early pipeline drain.
        constexpr uint32_t weightBytes = (kTileRows * sizeof(bfloat16_t) + 31) / 32 * 32;
        constexpr uint32_t weightOffset = Catlass::Arch::Ascend950::UB_SIZE - weightBytes;
        static_assert(weightOffset >= SavedGateUpMxFp8<kTileRows>::Layout::totalBytesWithClipMasks);
        AscendC::LocalTensor<bfloat16_t> weightUb{AscendC::TPosition::VECIN, weightOffset, kTileRows};
        if constexpr (ApplyTopkWeight_) {
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyExtParams p{1, uint32_t(mSize * sizeof(bfloat16_t)), 0, 0, 0};
            AscendC::DataCopyPad(weightUb, routedTopkWeightGlobal_[tileRow], p, {});
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        auto* weights = (__ubuf__ bfloat16_t*)weightUb.GetPhyAddr();
        // Direct B16 unpack follows the reference implementation. Arithmetic,
        // accurate division, clipping and both BF16 roundings remain ours.
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> lim, negLim;
            if (clip) {
                AscendC::Reg::Duplicate(lim, limit);
                AscendC::Reg::Duplicate(negLim, -limit);
            }
            for (uint16_t row = 0; row < mSize; ++row) {
                AscendC::Reg::RegTensor<bfloat16_t> weightBf;
                AscendC::Reg::RegTensor<float> weightFp;
                if constexpr (ApplyTopkWeight_) {
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(weightBf, weights + row);
                    auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(weightFp, weightBf, all);
                }
                uint32_t remaining = nSize;
                for (uint16_t col = 0; col < nSize; col += lanes) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::RegTensor<bfloat16_t> g, u, output;
                    AscendC::Reg::RegTensor<float> gf, uf, neg, exp, denom, silu, value;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        g, gate + row * stride + col);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        u, up + row * stride + col);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gf, g, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(uf, u, mask);
                    if (clip) {
                        AscendC::Reg::Min(gf, gf, lim, mask);
                        AscendC::Reg::Max(uf, uf, negLim, mask);
                        AscendC::Reg::Min(uf, uf, lim, mask);
                    }
                    AscendC::Reg::Muls(neg, gf, -1.0F, mask);
                    AscendC::Reg::Exp(exp, neg, mask);
                    AscendC::Reg::Adds(denom, exp, 1.0F, mask);
                    AscendC::Reg::Div<float, &kDivMode>(silu, gf, denom, mask);
                    AscendC::Reg::Mul(value, silu, uf, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(output, value, mask);
                    if constexpr (ApplyTopkWeight_) {
                        // Keep BF16(SwiGLU) * BF16(weight) -> BF16, including
                        // the intermediate rounding absent in the reference.
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(value, output, mask);
                        AscendC::Reg::Mul(value, value, weightFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(output, value, mask);
                    }
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                        gate + row * stride + col, output, mask);
                }
            }
        }
    }

    __aicore__ inline void ComputeSwiglu(uint16_t mSize, uint16_t nSize)
    {
        constexpr uint16_t kElementsPerRepeat = 256U / sizeof(bfloat16_t);
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kElementsPerRepeat)));
        const uint32_t rowStride = Ops::Base::CeilAlign(
            static_cast<uint32_t>(nSize), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(bfloat16_t)));
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        // The disabled limit (FLT_MAX) makes the clamps numerical no-ops;
        // skip them entirely so the unlimited path pays nothing extra.
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        __ubuf__ bfloat16_t* first = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t* second = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();

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
                    AscendC::Reg::DataCopy(gate0, first + offset);
                    AscendC::Reg::DataCopy(up0, second + offset);
                    AscendC::Reg::Interleave(gate0, gate1, gate0, gate1);
                    AscendC::Reg::Interleave(up0, up1, up0, up1);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gateFp0, gate0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gateFp1, gate1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(upFp0, up0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(upFp1, up1, mask);

                    constexpr uint16_t kParts = sizeof(float) / sizeof(bfloat16_t);
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto gateFp = part == 0 ? gateFp0 : gateFp1;
                        auto upFp = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> neg, exp, denom, silu, swigluFp;
                        AscendC::Reg::RegTensor<bfloat16_t> output;
                        // dsv4 swiglu_limit clamp (matches the eager path):
                        // gate = min(gate, L); up = clamp(up, -L, L).
                        // Skipped entirely when the limit is disabled.
                        if (limitEnabled) {
                            AscendC::Reg::Min(gateFp, gateFp, limitReg, mask);
                            AscendC::Reg::Max(upFp, upFp, negLimitReg, mask);
                            AscendC::Reg::Min(upFp, upFp, limitReg, mask);
                        }
                        AscendC::Reg::Muls(neg, gateFp, -kOne, mask);
                        AscendC::Reg::Exp(exp, neg, mask);
                        AscendC::Reg::Adds(denom, exp, kOne, mask);
                        AscendC::Reg::Div<float, &kDivMode>(silu, gateFp, denom, mask);
                        AscendC::Reg::Mul(swigluFp, silu, upFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(output, swigluFp, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            first + offset + part * kPartSize, output, mask);
                    }
                }
            }
        }
    }

    __aicore__ inline void ApplyTopkWeight(uint16_t mSize, uint16_t nSize, uint32_t tileRow)
    {
        const uint32_t rowStride = Ops::Base::CeilAlign(
            static_cast<uint32_t>(nSize), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(bfloat16_t)));

        // Preserve the validated numerical order:
        // BF16(SwiGLU_FP32) * BF16(route_weight) -> BF16.
        AscendC::SyncFunc<AscendC::HardEvent::V_S>();
        for (uint16_t row = 0; row < mSize; ++row) {
#if defined(DEEPEP_BF16_M192_FORWARD)
            // Dispatch can update a cache line previously read for another expert.
            union {
                uint16_t bits;
                bfloat16_t value;
            } freshWeight;
            freshWeight.bits =
                AscendC::ReadGmByPassDCache(const_cast<__gm__ uint16_t*>(reinterpret_cast<const __gm__ uint16_t*>(
                                                routedTopkWeightGlobal_.GetPhyAddr())) +
                                            tileRow + row);
            const bfloat16_t rowWeight = freshWeight.value;
#else
            const bfloat16_t rowWeight = routedTopkWeightGlobal_.GetValue(tileRow + row);
#endif
            AscendC::SyncFunc<AscendC::HardEvent::S_V>();
            AscendC::Muls(l0cOutUbFirst_[static_cast<uint32_t>(row) * rowStride],
                          l0cOutUbFirst_[static_cast<uint32_t>(row) * rowStride], rowWeight, nSize);
        }
    }

    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, kTileElements};
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond_{AscendC::TPosition::VECIN, kTileElements * sizeof(bfloat16_t),
                                                     kTileElements};
    AscendC::GlobalTensor<bfloat16_t> outputGlobal_;
    AscendC::GlobalTensor<bfloat16_t> savedGateGlobal_;
    AscendC::GlobalTensor<bfloat16_t> savedUpGlobal_;
    AscendC::GlobalTensor<bfloat16_t> routedTopkWeightGlobal_;
    __gm__ int32_t* groupFlagListGmAddr_{nullptr};
    Params params_{};
    int64_t outputN_{0};
    int64_t savedBaseOffset_{0};
    SavedGateUpMxFp8<kTileRows> quantSave_;
    static_assert(SavedGateUpMxFp8<kTileRows>::Layout::totalBytesWithClipMasks <= Catlass::Arch::Ascend950::UB_SIZE,
                  "Saved gate/up MXFP8 scratch must fit AIV0 UB");
    static_assert((kTileRows == 128 || kTileRows == 192) && kTileColumns == 256,
                  "Update saved quant UB for a new tile shape");
};

}  // namespace MegaMoeBF16Impl

#endif  // DEEPEP_BLOCK_EPILOGUE_SWIGLU_FORWARD_BF16_HPP
