#pragma once

#include <cstdint>

#if defined(ASCEND_MEGAMOE_USE_NEW_ASCENDC_FRONTEND)
#include <kernel_operator.h>
#define MEGA_MOE_BF16_ROUTING_HOST_DEVICE __host_aicore__
#elif defined(__CCE__)
#include <kernel_operator.h>
#define MEGA_MOE_BF16_ROUTING_HOST_DEVICE __host__ __aicore__
#else
#define MEGA_MOE_BF16_ROUTING_HOST_DEVICE
#endif

namespace MegaMoeBF16Routing {

// A 16-bit Gather offset must remain below 2^17 bytes. Keep the full
// source window at UB offset zero in a dedicated, short-lived pipe.
struct TopkWeightGatherPlan {
    // Maximum rows per window; the opt-in two-window path doubles the total.
    static constexpr uint32_t kMaxRoutes = 65536;
    static constexpr uint32_t kTwoWindowMaxRoutes = 2 * kMaxRoutes;
    static constexpr uint32_t kBatchRoutes = 2048;
    static constexpr uint32_t kIndexBytes = kBatchRoutes * sizeof(uint32_t);
    static constexpr uint32_t kOutputBytes = kBatchRoutes * sizeof(uint16_t);
    static constexpr uint32_t kMaxUbBytes = kMaxRoutes * sizeof(uint16_t) + kIndexBytes + kOutputBytes;
    static constexpr uint32_t kTwoWindowUbBytes = kMaxUbBytes + kIndexBytes + kOutputBytes;

    bool enabled = false;
    uint32_t sourceBytes = 0;
    uint32_t begin = 0;
    uint32_t count = 0;
    bool twoWindows = false;

    MEGA_MOE_BF16_ROUTING_HOST_DEVICE constexpr TopkWeightGatherPlan(uint64_t totalRoutes, uint32_t coreIndex,
                                                                     uint32_t coreCount, bool allowTwoWindows = false)
    {
        if (totalRoutes == 0 || totalRoutes > (allowTwoWindows ? kTwoWindowMaxRoutes : kMaxRoutes) || coreCount == 0 ||
            coreIndex >= coreCount) {
            return;
        }
        enabled = true;
        twoWindows = totalRoutes > kMaxRoutes;
        sourceBytes =
            twoWindows ? kMaxRoutes * sizeof(uint16_t) : static_cast<uint32_t>((totalRoutes * 2 + 31) / 32 * 32);
        const uint64_t perCore = (totalRoutes + coreCount - 1) / coreCount;
        const uint64_t start = perCore * coreIndex;
        begin = static_cast<uint32_t>(start < totalRoutes ? start : totalRoutes);
        const uint64_t remaining = totalRoutes - begin;
        count = static_cast<uint32_t>(remaining < perCore ? remaining : perCore);
    }

    MEGA_MOE_BF16_ROUTING_HOST_DEVICE constexpr uint32_t BatchCount(uint32_t offset) const
    {
        if (offset >= count) {
            return 0;
        }
        return count - offset < kBatchRoutes ? count - offset : kBatchRoutes;
    }
};

static_assert(TopkWeightGatherPlan::kMaxUbBytes == 140 * 1024,
              "BF16 routed-weight gather must fit its independent A5 UB phase");
static_assert(TopkWeightGatherPlan::kTwoWindowUbBytes == 152 * 1024,
              "Two windows reuse one source table, with bounded offsets and merge buffers");

}  // namespace MegaMoeBF16Routing

#undef MEGA_MOE_BF16_ROUTING_HOST_DEVICE
