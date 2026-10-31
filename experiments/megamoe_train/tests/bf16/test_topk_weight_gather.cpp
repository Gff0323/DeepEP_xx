// Host-only planning/coverage tests. This does not emulate Ascend instructions
// or establish MTE visibility on a device.
#include "../../csrc/kernels/bf16/layout/mega_moe_topk_weight_gather_layout.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using MegaMoeBF16Routing::TopkWeightGatherPlan;

void Check(uint32_t routes, uint32_t cores, std::mt19937& rng)
{
    std::vector<uint16_t> weights(routes), output(routes);
    std::vector<uint32_t> permutation(routes), writes(routes, 0);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    // At 65536 routes, exercise every BF16 bit pattern (including NaNs, signed
    // zero and infinities), with no conversion or floating point comparison.
    std::iota(weights.begin(), weights.end(), uint16_t{0});
    for (uint32_t core = 0; core < cores; ++core) {
        const TopkWeightGatherPlan plan(routes, core, cores, true);
        assert(plan.enabled);
        assert(plan.sourceBytes % 32 == 0);
        assert(plan.sourceBytes >= std::min(routes, plan.kMaxRoutes) * sizeof(uint16_t));
        assert(plan.sourceBytes + plan.kIndexBytes + plan.kOutputBytes <= plan.kMaxUbBytes);
        assert(plan.begin + plan.count <= routes);
        assert(plan.BatchCount(plan.count) == 0);
        for (uint32_t offset = 0; offset < plan.count; offset += plan.kBatchRoutes) {
            const uint32_t count = plan.BatchCount(offset);
            assert(count > 0 && count <= plan.kBatchRoutes);
            assert(count * sizeof(uint32_t) <= plan.kIndexBytes);
            assert(count * sizeof(uint16_t) <= plan.kOutputBytes);
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t dst = plan.begin + offset + i;
                const uint32_t window = permutation[dst] >= plan.kMaxRoutes ? 1 : 0;
                const uint32_t localIndex = permutation[dst] - window * plan.kMaxRoutes;
                const uint32_t byteOffset = localIndex * sizeof(uint16_t);
                assert(byteOffset < (1U << 17));
                assert(byteOffset + sizeof(uint16_t) <=
                       std::min(routes - window * plan.kMaxRoutes, plan.kMaxRoutes) * sizeof(uint16_t));
                std::memcpy(
                    &output[dst],
                    reinterpret_cast<const unsigned char*>(weights.data() + window * plan.kMaxRoutes) + byteOffset,
                    sizeof(uint16_t));
                // Both window offsets are safe, including a one-element tail.
                for (uint32_t w = 0; w < (plan.twoWindows ? 2U : 1U); ++w) {
                    const uint32_t safe = window == w ? localIndex : 0;
                    assert(safe < std::min(routes - w * plan.kMaxRoutes, plan.kMaxRoutes));
                }
                ++writes[dst];
            }
        }
    }
    for (uint32_t row = 0; row < routes; ++row) {
        assert(writes[row] == 1);
        assert(output[row] == weights[permutation[row]]);
    }
}

int main()
{
    std::mt19937 rng(20260905);
    for (uint32_t routes :
         {1U, 7U, 17U, 63U, 65U, 2047U, 2048U, 2049U, 65535U, 65536U, 65537U, 65552U, 98305U, 131071U, 131072U}) {
        for (uint32_t cores : {1U, 2U, 32U, 64U, 128U}) {
            Check(routes, cores, rng);
        }
    }
    for (uint32_t trial = 0; trial < 100; ++trial) {
        Check(1 + rng() % 131072, 1 + rng() % 128, rng);
    }
    const TopkWeightGatherPlan typical(4096ULL * 16, 0, 64);
    assert(typical.sourceBytes == 128 * 1024 && typical.count == 1024);
    assert(TopkWeightGatherPlan(1, 63, 64).count == 0);
    assert(!TopkWeightGatherPlan(0, 0, 64).enabled);
    assert(!TopkWeightGatherPlan(65537, 0, 64).enabled);
    assert(TopkWeightGatherPlan(65537, 0, 64, true).twoWindows);
    assert(!TopkWeightGatherPlan(131073, 0, 64, true).enabled);
    assert(!TopkWeightGatherPlan(1ULL << 40, 0, 64).enabled);
    assert(!TopkWeightGatherPlan(16, 0, 0).enabled);
    assert(!TopkWeightGatherPlan(16, 64, 64).enabled);
    std::cout
        << "PASS: routed-weight gather capacity, per-core coverage, batches, tails, byte offsets and bit patterns\n";
}
