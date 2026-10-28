// Host-only selection/layout checks. NPU tests cover publication visibility.
#include "../../csrc/kernels/bf16/include/mega_moe_training_workspace.h"
#include <cassert>
#include <cstdint>
#include <iostream>

using namespace MegaMoeBF16Training;

int main()
{
    constexpr uint64_t tileM = 128;
    static_assert(ForwardDispatchTileReadyLayout::kMaxReadyExperts == 7);
    static_assert(DefaultBackwardDispatchSplitExperts(kSaveBf16, 7) == 7);
    static_assert(DefaultBackwardDispatchSplitExperts(kSaveBf16, 8) == 0);
    static_assert(DefaultBackwardDispatchSplitExperts(kSaveMxFp8, 16) == 0);
    static_assert(DefaultBackwardDispatchSplitExperts(kSaveNone, 7) == 0);
    for (uint32_t mode : {kSaveNone, kSaveBf16, kSaveMxFp8}) {
        for (uint32_t experts : {0U, 1U, 4U, 7U, 8U, 16U, 32U}) {
            const uint32_t k = DefaultBackwardDispatchSplitExperts(mode, experts);
            assert(k == (mode != kSaveNone && experts <= 7 ? experts : 0));
            assert(UseBackwardDispatchRowReady(mode, 0, k, 129, tileM) == (k > 0));
            assert(!UseBackwardDispatchRowReady(mode, 0, k, 128, tileM));
            for (uint64_t ep : {4ULL, 8ULL, 64ULL, 128ULL, 256ULL}) {
                constexpr uint64_t oldBytes = 4099;
                assert(BackwardDispatchProgressLayout::TotalBytes(oldBytes, k, ep, 1, mode) ==
                       (k ? AlignUp(oldBytes, 64) + k * ep * 64 : oldBytes));
            }
        }
    }
    // Explicit debug k remains usable above the default expert cutoff.
    assert(UseBackwardDispatchRowReady(kSaveBf16, 7, 8, 4096, tileM));
    for (uint32_t mode : {kSaveNone, kSaveBf16, kSaveMxFp8}) {
        for (uint64_t rows : {0ULL, 1ULL, 127ULL, 128ULL, 129ULL, 256ULL, 16384ULL, 18725ULL, 131072ULL}) {
            for (uint32_t k : {0U, 1U, 4U, 7U}) {
                for (uint32_t ordinal = 0; ordinal < 8; ++ordinal) {
                    const bool expected = mode != kSaveNone && ordinal < k && rows > tileM;
                    assert(UseBackwardDispatchRowReady(mode, ordinal, k, rows, tileM) == expected);
                }
            }
        }
    }
    // A protected first non-empty expert still consumes ordinal 0. The next
    // expert does not get promoted into k=1 merely because the first is small.
    assert(!UseBackwardDispatchRowReady(kSaveBf16, 0, 1, 128, tileM));
    assert(!UseBackwardDispatchRowReady(kSaveBf16, 1, 1, 4096, tileM));
    assert(UseBackwardDispatchRowReady(kSaveBf16, 1, 7, 4096, tileM));
    for (uint64_t ep : {1ULL, 4ULL, 8ULL, 16ULL, 32ULL, 64ULL, 128ULL, 256ULL}) {
        for (uint64_t producers : {1ULL, 4ULL, 8ULL}) {
            for (uint64_t k : {0ULL, 1ULL, 4ULL, 7ULL}) {
                constexpr uint64_t oldBytes = 4099;
                const auto base = k == 0 ? oldBytes : AlignUp(oldBytes, 64);
                assert(BackwardDispatchProgressLayout::TotalBytes(oldBytes, k, ep, producers, kSaveBf16) ==
                       base + k * ep * producers * 64);
                assert(BackwardDispatchProgressLayout::TotalBytes(oldBytes, k, ep, producers, kSaveMxFp8) ==
                       base + k * ep * producers * 64);
                assert(BackwardDispatchProgressLayout::TotalBytes(oldBytes, k, ep, producers, kSaveNone) == oldBytes);
            }
        }
    }
    std::cout << "[BACKWARD_DISPATCH_POLICY] PASS single-M, ordinal, modes, EP256 layout\n";
}
