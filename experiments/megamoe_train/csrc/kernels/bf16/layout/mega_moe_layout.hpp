#pragma once

#include <cstdint>
#include <cfloat>

#if defined(ASCEND_MEGAMOE_USE_NEW_ASCENDC_FRONTEND)
#include <kernel_operator.h>
#define MEGA_MOE_BF16_HOST_DEVICE __host_aicore__
#elif defined(__CCE__)
#include <kernel_operator.h>
#define MEGA_MOE_BF16_HOST_DEVICE __host__ __aicore__
#else
#define MEGA_MOE_BF16_HOST_DEVICE
#endif

namespace MegaMoeBF16Training {

constexpr uint64_t kAlignment128 = 128;
constexpr uint64_t kAlignment256 = 256;
constexpr uint64_t kAlignment512 = 512;
constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kInt32Bytes = 4;
constexpr uint64_t kIntCacheline = 16;
constexpr uint64_t kBackwardProgressElements = 256;
constexpr uint64_t kMaxEpRanks = 1024;
constexpr uint64_t kMaxLocalExperts = 32;
constexpr uint64_t kMaxTotalExperts = 2048;

// Ordinary BF16 always reserves 64 KiB for one 64-byte slot per supported
// rank and 8 KiB for per-AIV generation counters. The shared 72 KiB header
// keeps Host allocation, forward and backward offsets independent of EP.
constexpr uint64_t kPeerRankSyncSlotBytes = 64;
// CalcPeermemSize still includes the common 60 KiB legacy header; ordinary
// BF16 Host allocation adds the difference to the fixed data offset below.
constexpr uint64_t kPeerBaseDataOffset = 60 * 1024;
constexpr uint64_t kPeerSyncCountOffset = 64 * 1024;
constexpr uint64_t kPeerDataOffset = 72 * 1024;
constexpr uint64_t kCompactTokenCountAutoMinEp = 512;
// A 64-byte packet stores up to eight int32 counts in [0, 32) and the
// matching eight exclusive prefixes in [32, 64).
constexpr uint64_t kCompactTokenCountMaxLocalExperts = 8;
constexpr uint64_t kCompactTokenCountPacketBytes = 64;
constexpr uint64_t kCompactTokenCountPrefixOffset = 32;
constexpr uint64_t kCompactTokenCountMaxAivCores = 72;
static_assert(kCompactTokenCountMaxLocalExperts * kInt32Bytes <= kCompactTokenCountPrefixOffset);
static_assert(kCompactTokenCountPrefixOffset + kCompactTokenCountMaxLocalExperts * kInt32Bytes <=
              kCompactTokenCountPacketBytes);

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t PeerSyncCountOffset(uint64_t worldSize)
{
    (void)worldSize;
    return kPeerSyncCountOffset;
}

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t PeerDataOffset(uint64_t worldSize)
{
    (void)worldSize;
    return kPeerDataOffset;
}

static_assert(kMaxEpRanks * kPeerRankSyncSlotBytes <= kPeerSyncCountOffset);
// A5 exposes at most 72 AIVs (36 AICs in MIX 1:2); 8 KiB is ample for their
// 64-byte generation-counter slots.
static_assert(72 * kPeerRankSyncSlotBytes <= kPeerDataOffset - kPeerSyncCountOffset);

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

MEGA_MOE_BF16_HOST_DEVICE constexpr bool CanUseCompactTokenCountExchange(uint64_t worldSize, uint64_t expertPerRank)
{
    return worldSize >= 2 && worldSize <= kMaxEpRanks && expertPerRank > 0 &&
           expertPerRank <= kCompactTokenCountMaxLocalExperts;
}

MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseCompactTokenCountExchangeByDefault(uint64_t worldSize,
                                                                               uint64_t expertPerRank)
{
    return worldSize >= kCompactTokenCountAutoMinEp && CanUseCompactTokenCountExchange(worldSize, expertPerRank);
}

// Ordinary BF16 large-EP exchange sends one aligned packet per destination.
// Phase one keeps the existing dense local table and saved-state ABI: packets
// are unpacked into the same source-rank rows consumed by forward/backward.
// Experimental variants pass allowCompact=false and retain their old layout.
struct CompactTokenCountExchangeLayout {
    bool enabled;
    uint64_t packetOffset;
    uint64_t packetBytes;
    uint64_t blockSumsOffset;
    uint64_t blockSumsBytes;
    uint64_t tokenCountsOffset;
    uint64_t tokenCountStride;
    uint64_t tokenCountsBytes;
    uint64_t payloadOffset;

    MEGA_MOE_BF16_HOST_DEVICE constexpr CompactTokenCountExchangeLayout(uint64_t worldSize, uint64_t expertPerRank,
                                                                        bool allowCompact = true)
        : enabled(allowCompact && CanUseCompactTokenCountExchange(worldSize, expertPerRank)),
          packetOffset(kPeerDataOffset),
          packetBytes(enabled ? AlignUp(worldSize * kCompactTokenCountPacketBytes, kAlignment512) : 0),
          blockSumsOffset(packetOffset + packetBytes),
          blockSumsBytes(enabled ? AlignUp(kCompactTokenCountMaxAivCores * kInt32Bytes, kAlignment512) : 0),
          tokenCountsOffset(blockSumsOffset + blockSumsBytes),
          tokenCountStride(AlignUp(worldSize * expertPerRank, kAlignment128)),
          tokenCountsBytes(worldSize * tokenCountStride * kInt32Bytes),
          payloadOffset(tokenCountsOffset + tokenCountsBytes)
    {
    }
};

// Host normalizes the public swiglu_limit=0 to FLT_MAX in tiling.
enum SavedGateUpMode : uint32_t { kSaveAuto = 0, kSaveNone = 1, kSaveBf16 = 2, kSaveMxFp8 = 3 };

MEGA_MOE_BF16_HOST_DEVICE constexpr uint32_t ResolveSavedGateUpMode(uint32_t mode, float tilingLimit)
{
    return mode == kSaveAuto ? (tilingLimit == FLT_MAX ? kSaveMxFp8 : kSaveBf16) : mode;
}

// Ordinary forward only. UDMA/MGroup keep their existing UB path.
// This policy is compiled into the kernel SO and queried by Host allocation.
#ifndef DEEPEP_BF16_SWIGLU_DECOUPLE
#define DEEPEP_BF16_SWIGLU_DECOUPLE 0
#endif
static_assert(DEEPEP_BF16_SWIGLU_DECOUPLE >= 0 && DEEPEP_BF16_SWIGLU_DECOUPLE <= 2,
              "BF16 SwiGLU decoupling must be off(0), on(1), or auto(2)");
constexpr uint64_t kForwardSwigluAutoBoundary = 3584;

MEGA_MOE_BF16_HOST_DEVICE constexpr bool ForwardUsesGmSwiglu(uint64_t hidden, uint32_t savedMode,
                                                             uint32_t mode = DEEPEP_BF16_SWIGLU_DECOUPLE)
{
    return mode == 1 || (mode == 2 && (hidden < kForwardSwigluAutoBoundary ||
                                       (hidden == kForwardSwigluAutoBoundary && savedMode == kSaveMxFp8)));
}

MEGA_MOE_BF16_HOST_DEVICE constexpr uint64_t ForwardSwigluExtraBytes(uint64_t n, uint64_t maxRecvTokens,
                                                                     uint32_t savedMode, bool decoupled)
{
    // BF16 already owns raw Gate/Up. Other modes reuse Hp for raw Gate;
    // append only raw Up, after all existing payload and control flags.
    return decoupled && savedMode != kSaveBf16 ? AlignUp(maxRecvTokens * (n / 2) * kBf16Bytes, kAlignment512) : 0;
}

// Use the resolved saved mode in both host allocation and device execution.
// Saved modes use Hp/dO reuse and per-expert readiness; none retains UB.
MEGA_MOE_BF16_HOST_DEVICE constexpr bool BackwardUsesGm(uint32_t savedMode)
{
    return savedMode != kSaveNone;
}

// The caller supplies the effective (capacity-clipped) local expert rows.
// A single M tile has no inter-row-tile pipeline to overlap with Dispatch.
// ordinal still counts every non-empty expert; a fallback does not promote
// a later expert into the first k selected experts.
MEGA_MOE_BF16_HOST_DEVICE constexpr bool UseBackwardDispatchRowReady(uint32_t savedMode, uint32_t ordinal,
                                                                     uint32_t splitExperts, uint64_t effectiveRows,
                                                                     uint64_t computeRows)
{
    return BackwardUsesGm(savedMode) && ordinal < splitExperts && effectiveRows > computeRows;
}

MEGA_MOE_BF16_HOST_DEVICE constexpr bool SaveGateUpAsMxFp8(float tilingLimit, uint32_t mode = kSaveAuto)
{
    return ResolveSavedGateUpMode(mode, tilingLimit) == kSaveMxFp8;
}

// v2 appends three uint32 bit planes only for explicit MXFP8 with clipping.
// All v1 prefix/data/scale offsets are retained. No GM header or initialization.
enum SavedClipMaskKind : uint32_t { kClipMaskNone = 0, kClipMaskThreeBitplanesV1 = 1 };
MEGA_MOE_BF16_HOST_DEVICE constexpr bool SaveGateUpClipMasks(float tilingLimit, uint32_t mode)
{
    return SaveGateUpAsMxFp8(tilingLimit, mode) && tilingLimit != FLT_MAX;
}

// AIV0 epilogue only. Dispatch and GMM2 own UB in different phases.
template <uint32_t RawRows, uint32_t ChunkRows = RawRows>
struct SavedGateUpQuantUbLayoutFor {
    static constexpr uint32_t kRows = ChunkRows;
    static constexpr uint32_t kColumns = 256;
    static constexpr uint32_t kElements = kRows * kColumns;
    static constexpr uint32_t kGroups = kElements / 32;
    static constexpr uint32_t gateData = 2 * RawRows * kColumns * kBf16Bytes;
    static constexpr uint32_t upData = gateData + kElements;
    static constexpr uint32_t compactScale = upData + kElements;
    static constexpr uint32_t gateScale = compactScale + kGroups * 2;
    static constexpr uint32_t upScale = gateScale + kRows * 32;
    static constexpr uint32_t maxExp = upScale + kRows * 32;
    static constexpr uint32_t halfScale = maxExp + kGroups * 2;
    static constexpr uint32_t totalBytes = halfScale + kGroups * 2;
    static constexpr uint32_t maskPlaneBytes = kRows * 32;
    static constexpr uint32_t gateHigh = totalBytes;
    static constexpr uint32_t upLow = gateHigh + maskPlaneBytes;
    static constexpr uint32_t upHigh = upLow + maskPlaneBytes;
    static constexpr uint32_t totalBytesWithClipMasks = upHigh + maskPlaneBytes;
};

using SavedGateUpQuantUbLayout = SavedGateUpQuantUbLayoutFor<128>;

// Simultaneous UB allocations during count exchange and Dispatch. GM count
// rows contain all experts; Cumsum gathers only this rank's expert columns.
struct ForwardDispatchUbLayout {
    // The ordinary M192 entry opts in; Host, backward, MGroup and UDMA
    // retain their configured granularity and conservative UB capacity check.
    MEGA_MOE_BF16_HOST_DEVICE static constexpr uint64_t CopyRowsForForward(uint64_t configuredRows)
    {
#if defined(DEEPEP_BF16_M192_FORWARD)
        (void)configuredRows;
        return 1;
#else
        return configuredRows;
#endif
    }

    static constexpr uint64_t kTopkWeightBytes = 4 * 1024;
    static constexpr uint64_t kCountResetChunkBytes = 16 * 1024;
    uint64_t syncBytes;
    uint64_t prevSumBytes;
    uint64_t cumsumBytes;
    uint64_t copyElementsPerPing;
    uint64_t copyBytes;
    uint64_t resetFlagBytes;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr ForwardDispatchUbLayout(uint64_t worldSize, uint64_t expertPerRank, uint64_t k,
                                                                uint64_t dispatchRows)
        : syncBytes(AlignUp(worldSize * expertPerRank, kAlignment128) * kInt32Bytes),
          // CrossRankSyncCntPerExpert writes and copies exactly one int32
          // prefix per local expert.  It never indexes this buffer by rank or
          // total expert, so reserving syncBytes * expertPerRank was excessive.
          prevSumBytes(AlignUp(expertPerRank * kInt32Bytes, 32)),
          cumsumBytes(worldSize * AlignUp(expertPerRank * kInt32Bytes, 32)),
          copyElementsPerPing(AlignUp(dispatchRows * k, 32)),
          copyBytes(2 * copyElementsPerPing * kBf16Bytes),
          // Preserve the existing scratch reservation, rounded as InitBuffer does.
          resetFlagBytes(AlignUp((expertPerRank * kIntCacheline * kInt32Bytes * 2 + 31) / 32, 32)),
          totalBytes(syncBytes + prevSumBytes + cumsumBytes + copyBytes + kTopkWeightBytes + resetFlagBytes)
    {
    }
};

// Ordinary BF16 Dispatch output. Packed tokens and their packed TopK weights
// always share one lifetime because every row uses the same routing order.
// Keeping them in an independent allocation lets the framework retain both
// for the existing backward or release both for a redispatching backward.
struct DispatchOutputLayout {
    uint64_t tokensOffset;
    uint64_t tokensBytes;
    uint64_t topkWeightsOffset;
    uint64_t topkWeightsBytes;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr DispatchOutputLayout(uint64_t k, uint64_t maxRecvTokens)
        : tokensOffset(0),
          tokensBytes(AlignUp(maxRecvTokens * k * kBf16Bytes, kAlignment512)),
          topkWeightsOffset(tokensOffset + tokensBytes),
          topkWeightsBytes(AlignUp(maxRecvTokens * kBf16Bytes, kAlignment512)),
          totalBytes(topkWeightsOffset + topkWeightsBytes)
    {
    }
};

// State retained from forward until the matching backward completes.
struct TrainingSavedStateLayout {
    uint64_t ptrAOffset;
    uint64_t ptrABytes;
    uint64_t cumsumOffset;
    uint64_t expandedRowIdxOffset;
    uint64_t expandedRowIdxGatherOffset;
    uint64_t sumBeforeRankOffset;
    uint64_t savedTokenCountsOffset;
    uint64_t dispatchedTopkWeightsOffset;
    uint64_t dispatchedTopkWeightsBytes;
    uint64_t backwardProgressOffset;
    // Append-only extension: existing metadata/progress offsets stay fixed.
    uint64_t legacyBytes;
    uint64_t gateOffset;
    uint64_t upOffset;
    uint64_t gateUpPlaneBytes;
    uint64_t gateScaleOffset;
    uint64_t upScaleOffset;
    uint64_t gateUpScaleBytes;
    bool gateUpMxFp8;
    bool gateUpSaved;
    uint64_t gateHighOffset;
    uint64_t upLowOffset;
    uint64_t upHighOffset;
    uint64_t clipMaskPlaneBytes;
    uint32_t clipMaskKind;
    uint32_t layoutVersion;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr TrainingSavedStateLayout(uint64_t m, uint64_t k, uint64_t topk,
                                                                 uint64_t worldSize, uint64_t expertPerRank,
                                                                 uint64_t maxRecvTokens, uint64_t n, bool saveMxFp8,
                                                                 bool saveGateUp = true, bool saveClipMasks = false,
                                                                 bool embedDispatchOutput = true)
        : ptrAOffset(0),
          ptrABytes(embedDispatchOutput ? AlignUp(maxRecvTokens * k * kBf16Bytes, kAlignment512) : 0),
          cumsumOffset(ptrAOffset + ptrABytes),
          expandedRowIdxOffset(cumsumOffset + AlignUp(worldSize * expertPerRank * kInt32Bytes, kAlignment512)),
          expandedRowIdxGatherOffset(expandedRowIdxOffset + AlignUp(m, kAlignment256) * topk * kInt32Bytes),
          sumBeforeRankOffset(expandedRowIdxGatherOffset + AlignUp(m, kAlignment256) * topk * kInt32Bytes),
          savedTokenCountsOffset(sumBeforeRankOffset + AlignUp(worldSize * expertPerRank, kAlignment128) * kInt32Bytes),
          dispatchedTopkWeightsOffset(
              savedTokenCountsOffset +
              AlignUp(worldSize * AlignUp(worldSize * expertPerRank, kAlignment128) * kInt32Bytes, kAlignment512)),
          dispatchedTopkWeightsBytes(embedDispatchOutput ? AlignUp(maxRecvTokens * kBf16Bytes, kAlignment512) : 0),
          backwardProgressOffset(dispatchedTopkWeightsOffset + dispatchedTopkWeightsBytes),
          legacyBytes(backwardProgressOffset + AlignUp(kBackwardProgressElements * kInt32Bytes, kAlignment128)),
          gateOffset(AlignUp(legacyBytes, kAlignment512)),
          upOffset(gateOffset +
                   (saveGateUp ? AlignUp(maxRecvTokens * (n / 2) * (saveMxFp8 ? 1 : kBf16Bytes), kAlignment512) : 0)),
          gateUpPlaneBytes(saveGateUp ? AlignUp(maxRecvTokens * (n / 2) * (saveMxFp8 ? 1 : kBf16Bytes), kAlignment512)
                                      : 0),
          gateScaleOffset(upOffset + gateUpPlaneBytes),
          upScaleOffset(gateScaleOffset +
                        (saveGateUp && saveMxFp8 ? AlignUp(maxRecvTokens * ((n / 2 + 31) / 32), kAlignment512) : 0)),
          gateUpScaleBytes(saveGateUp && saveMxFp8 ? AlignUp(maxRecvTokens * ((n / 2 + 31) / 32), kAlignment512) : 0),
          gateUpMxFp8(saveGateUp && saveMxFp8),
          gateUpSaved(saveGateUp),
          gateHighOffset(upScaleOffset + gateUpScaleBytes),
          upLowOffset(gateHighOffset + (saveGateUp && saveMxFp8 && saveClipMasks
                                            ? AlignUp(maxRecvTokens * ((n / 2 + 31) / 32) * kInt32Bytes, kAlignment512)
                                            : 0)),
          upHighOffset(upLowOffset + (upLowOffset - gateHighOffset)),
          clipMaskPlaneBytes(upLowOffset - gateHighOffset),
          clipMaskKind(clipMaskPlaneBytes ? kClipMaskThreeBitplanesV1 : kClipMaskNone),
          layoutVersion(clipMaskPlaneBytes ? 2 : 1),
          totalBytes(saveGateUp ? upHighOffset + clipMaskPlaneBytes : legacyBytes)
    {
    }
};

// Scratch used only by the current forward call. Host allocation is
// uninitialized; the forward kernel resets all control flags per invocation.
struct ForwardTransientWorkspaceLayout {
    uint64_t ptrA2Offset;
    uint64_t swigluToGmm2FlagOffset;
    uint64_t dispatchToGmm1FlagOffset;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr ForwardTransientWorkspaceLayout(uint64_t n, uint64_t expertPerRank,
                                                                        uint64_t maxRecvTokens)
        : ptrA2Offset(0),
          swigluToGmm2FlagOffset(ptrA2Offset + AlignUp(maxRecvTokens * (n / 2) * kBf16Bytes, kAlignment512)),
          dispatchToGmm1FlagOffset(swigluToGmm2FlagOffset +
                                   AlignUp(expertPerRank * kIntCacheline * kInt32Bytes, kAlignment128)),
          totalBytes(dispatchToGmm1FlagOffset + AlignUp(expertPerRank * kIntCacheline * kInt32Bytes, kAlignment128))
    {
    }
};

// UDMA keeps packed X and routed TopK weights in SHMEM. Routing metadata
// lives in TrainingSavedStateLayout, leaving only GMM2 input and flags here.
struct UDMAForwardTransientWorkspaceLayout {
    uint64_t ptrA2Offset;
    uint64_t swigluToGmm2FlagOffset;
    uint64_t dispatchToGmm1FlagOffset;
    uint64_t combineReadyOffset;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr UDMAForwardTransientWorkspaceLayout(uint64_t n, uint64_t expertPerRank,
                                                                            uint64_t maxRecvTokens)
        : ptrA2Offset(0),
          swigluToGmm2FlagOffset(AlignUp(maxRecvTokens * (n / 2) * kBf16Bytes, kAlignment512)),
          dispatchToGmm1FlagOffset(swigluToGmm2FlagOffset +
                                   AlignUp(expertPerRank * kIntCacheline * kInt32Bytes, kAlignment128)),
          combineReadyOffset(dispatchToGmm1FlagOffset +
                             AlignUp(expertPerRank * kIntCacheline * kInt32Bytes, kAlignment128)),
          totalBytes(combineReadyOffset +
                     AlignUp(expertPerRank * ((maxRecvTokens + 127) / 128) * kInt32Bytes, kAlignment512))
    {
    }
};

// Ordinary BF16 only: append row-ready counters without moving any existing
// transient buffer or changing the saved-state / MGroup / UDMA layouts. The
// forward kernel resets every counter before Dispatch can publish readiness.
// Backward-only progress suffix; all previous offsets stay unchanged.
struct BackwardDispatchProgressLayout {
    static constexpr uint64_t kSlotBytes = 64;
    MEGA_MOE_BF16_HOST_DEVICE static constexpr uint64_t ExtraBytes(uint64_t experts, uint64_t sources,
                                                                   uint64_t producersPerSource, uint32_t savedMode)
    {
        return BackwardUsesGm(savedMode) ? experts * sources * producersPerSource * kSlotBytes : 0;
    }
    MEGA_MOE_BF16_HOST_DEVICE static constexpr uint64_t TotalBytes(uint64_t legacyBytes, uint64_t experts,
                                                                   uint64_t sources, uint64_t producersPerSource,
                                                                   uint32_t savedMode)
    {
        return BackwardUsesGm(savedMode) && experts != 0
                   ? AlignUp(legacyBytes, kSlotBytes) + ExtraBytes(experts, sources, producersPerSource, savedMode)
                   : legacyBytes;
    }
};

struct ForwardDispatchTileReadyLayout {
    // Only the ordinary forward entry opts into M192. Shared variants retain M128.
#if defined(DEEPEP_BF16_M192_FORWARD)
    static constexpr uint64_t kComputeRows = 192;
    static constexpr uint64_t kMaxReadyExperts = 7;
#else
    static constexpr uint64_t kComputeRows = 128;
    static constexpr uint64_t kMaxReadyExperts = 7;
#endif
    static constexpr uint64_t kReadyRows = 2 * kComputeRows;
    // Keep Host and backward allocation unchanged; Ready384 uses fewer slots.
    static constexpr uint64_t kReservedRows = 256;
    // tile readiness is used only for one startup expert, so one counter set is
    // shared by whichever local expert is selected at runtime.
    static constexpr uint64_t kMinReadyRowsPerExpert = 2 * kReadyRows;
    uint64_t flagSlotsPerExpert;
    uint64_t flagCount;
    uint64_t offset;
    uint64_t bytes;
    uint64_t totalBytes;

    MEGA_MOE_BF16_HOST_DEVICE constexpr ForwardDispatchTileReadyLayout(uint64_t n, uint64_t expertPerRank,
                                                                       uint64_t maxRecvTokens)
        : flagSlotsPerExpert((maxRecvTokens + kReservedRows - 1) / kReservedRows),
          flagCount(flagSlotsPerExpert),
          offset(ForwardTransientWorkspaceLayout(n, expertPerRank, maxRecvTokens).totalBytes),
          bytes(AlignUp(flagCount * kIntCacheline * kInt32Bytes, kAlignment512)),
          totalBytes(offset + bytes)
    {
    }

    // A source-rank/core range can begin inside a ready block. Publish only
    // its intersection with that block, never rows belonging to the next one.
    MEGA_MOE_BF16_HOST_DEVICE static constexpr uint32_t FragmentRows(uint32_t expertRow, uint32_t remainingRows)
    {
        const uint32_t toBoundary = static_cast<uint32_t>(kReadyRows) - expertRow % static_cast<uint32_t>(kReadyRows);
        return remainingRows < toBoundary ? remainingRows : toBoundary;
    }

    // Called for a block containing a valid GMM1 row. Both compute
    // tiles in the block wait for its full payload, including a partial tail.
    MEGA_MOE_BF16_HOST_DEVICE static constexpr uint32_t RowsInReadyBlock(uint32_t blockIndex, uint32_t expertRows)
    {
        const uint32_t remainingRows = expertRows - blockIndex * static_cast<uint32_t>(kReadyRows);
        return remainingRows < kReadyRows ? remainingRows : static_cast<uint32_t>(kReadyRows);
    }

    // Invocation-wide candidate selection. A rank still checks the actual row
    // count of its first non-empty expert before enabling tile readiness locally.
    MEGA_MOE_BF16_HOST_DEVICE static constexpr bool UseTileReadyForShape(uint64_t m, uint64_t topK,
                                                                         uint64_t expertPerRank)
    {
        if (m == 0 || topK == 0 || expertPerRank == 0 || expertPerRank > kMaxReadyExperts) {
            return false;
        }
        const uint64_t routeRows = m * topK;
        return routeRows >= expertPerRank * kMinReadyRowsPerExpert;
    }

    MEGA_MOE_BF16_HOST_DEVICE static constexpr bool UseTileReadyForStartupExpert(bool shapeCandidate,
                                                                                 bool startupExpertHandled,
                                                                                 uint64_t expertRows)
    {
        return shapeCandidate && !startupExpertHandled && expertRows >= kMinReadyRowsPerExpert;
    }

    // Estimate the number of complete ready blocks emitted by one balanced
    // source/core partition. Aligned multi-block partitions use phase-major
    // ordering; sub-block and irregular partitions use sequential ordering.
    MEGA_MOE_BF16_HOST_DEVICE static constexpr uint32_t ReadyBlocksPerProducerPartition(uint64_t m, uint64_t topK,
                                                                                        uint64_t expertPerRank,
                                                                                        uint64_t epWorldSize,
                                                                                        uint64_t dispatchBlockNumPerEP)
    {
        if (m == 0 || topK == 0 || expertPerRank == 0 || epWorldSize == 0 || dispatchBlockNumPerEP == 0) {
            return 1;
        }
        const uint64_t partitions = expertPerRank * epWorldSize * dispatchBlockNumPerEP;
        const uint64_t partitionRows = (m * topK + partitions - 1) / partitions;
        if (partitionRows < kMinReadyRowsPerExpert || partitionRows % kReadyRows != 0) {
            return 1;
        }
        return static_cast<uint32_t>(partitionRows / kReadyRows);
    }
};

// Share only the local-expert cutoff with forward. Actual-row guards,
// compute mapping and explicit diagnostic k overrides remain independent.
MEGA_MOE_BF16_HOST_DEVICE constexpr uint32_t DefaultBackwardDispatchSplitExperts(uint32_t savedMode,
                                                                                 uint32_t localExperts)
{
    return BackwardUsesGm(savedMode) && localExperts <= ForwardDispatchTileReadyLayout::kMaxReadyExperts ? localExperts
                                                                                                         : 0U;
}

}  // namespace MegaMoeBF16Training

#undef MEGA_MOE_BF16_HOST_DEVICE
