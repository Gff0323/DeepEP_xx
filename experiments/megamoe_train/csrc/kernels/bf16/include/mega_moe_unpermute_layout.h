#pragma once

#include <cstdint>

#if defined(ASCEND_MEGAMOE_USE_NEW_ASCENDC_FRONTEND)
#include <kernel_operator.h>
#define MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE __host_aicore__
#elif defined(__CCE__)
#include <kernel_operator.h>
#define MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE __host__ __aicore__
#else
#define MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE
#endif

namespace MegaMoeBF16Routing {

struct UnpermutePlan {
    static constexpr uint32_t kIndexElements = 1024;
    static constexpr uint32_t kIndexBytes = kIndexElements * sizeof(int32_t);
    bool enabled = false;
    uint32_t rowBytes = 0;
    uint32_t batchTokens = 0;

    MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE constexpr UnpermutePlan(uint32_t hidden, uint32_t topK, uint32_t ubBytes)
    {
        const uint64_t bytes = uint64_t{hidden} * sizeof(uint16_t);
        if (hidden == 0 || bytes % 32 != 0 || topK == 0 || topK > kIndexElements || 4 * bytes + kIndexBytes > ubBytes) {
            return;
        }
        enabled = true;
        rowBytes = static_cast<uint32_t>(bytes);
        batchTokens = kIndexElements / topK;
    }

    MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE constexpr uint32_t BatchCount(uint32_t remaining) const
    {
        return remaining < batchTokens ? remaining : batchTokens;
    }
};

}  // namespace MegaMoeBF16Routing

#undef MEGA_MOE_BF16_UNPERMUTE_HOST_DEVICE
