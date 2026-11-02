// Host-only capacity/policy regression. NPU tests cover memory ordering.
#include "../../csrc/kernels/bf16/layout/mega_moe_layout.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>

using namespace MegaMoeBF16Training;

int main()
{
    using Ready = ForwardDispatchTileReadyLayout;
    static_assert(Ready::kComputeRows == 192);
    static_assert(Ready::kReadyRows == 384);
    static_assert(Ready::kMinReadyRowsPerExpert == 768);
    // Fragments are intersections with a target expert's Ready384 block.
    assert(Ready::FragmentRows(0, 900) == 384);
    assert(Ready::FragmentRows(300, 500) == 84);
    assert(Ready::FragmentRows(384, 300) == 300);
    assert(Ready::FragmentRows(767, 10) == 1);
    assert(Ready::FragmentRows(768, 0) == 0);
    for (uint32_t row = 0; row < 3 * Ready::kReadyRows; ++row) {
        for (uint32_t remaining : {0U, 1U, 83U, 192U, 383U, 384U, 385U, 900U}) {
            const uint32_t fragment = Ready::FragmentRows(row, remaining);
            assert(fragment <= remaining);
            assert(row / Ready::kReadyRows == (row + fragment - (fragment != 0)) / Ready::kReadyRows);
            if (fragment < remaining) {
                assert((row + fragment) % Ready::kReadyRows == 0);
            }
        }
    }
    for (uint64_t hidden : {256ULL, 3072ULL, 3583ULL, 3584ULL, 3585ULL, 4096ULL, 5120ULL, 6144ULL, 7168ULL}) {
        for (uint32_t saved : {kSaveNone, kSaveBf16, kSaveMxFp8}) {
            assert(!ForwardUsesGmSwiglu(hidden, saved, 0));
            assert(ForwardUsesGmSwiglu(hidden, saved, 1));
            const bool expected = hidden < 3584 || (hidden == 3584 && saved == kSaveMxFp8);
            assert(ForwardUsesGmSwiglu(hidden, saved, 2) == expected);
            assert(ForwardUsesGmSwiglu(hidden, saved) ==
                   ForwardUsesGmSwiglu(hidden, saved, DEEPEP_BF16_SWIGLU_DECOUPLE));
        }
    }
    for (uint64_t capacity : {0ULL, 1ULL, 193ULL, 65536ULL, 131072ULL}) {
        for (uint64_t n : {512ULL, 1024ULL, 6144ULL, 8192ULL}) {
            for (uint64_t experts : {1ULL, 4ULL, 7ULL, 16ULL, 32ULL}) {
                const ForwardTransientWorkspaceLayout payload(n, experts, capacity);
                const ForwardDispatchTileReadyLayout ready(n, experts, capacity);
                const uint64_t plane = AlignUp(capacity * (n / 2) * 2, 512);
                // Existing Hp/flags/Ready counters retain their exact prefix.
                assert(payload.ptrA2Offset == 0);
                assert(payload.swigluToGmm2FlagOffset == plane);
                assert(ready.offset == payload.totalBytes);
                assert(ready.totalBytes >= ready.offset + ready.bytes);
                for (uint32_t saved : {kSaveNone, kSaveBf16, kSaveMxFp8}) {
                    assert(ForwardSwigluExtraBytes(n, capacity, saved, false) == 0);
                    const auto extra = ForwardSwigluExtraBytes(n, capacity, saved, true);
                    assert(extra == (saved == kSaveBf16 ? 0 : plane));
                    assert((ready.totalBytes + extra) % 128 == 0);
                    if (extra != 0) {
                        // Appended raw Up is disjoint from Hp, every control
                        // line, and the entire reserved tile-ready region.
                        assert(ready.totalBytes >= payload.dispatchToGmm1FlagOffset);
                        assert(ready.totalBytes >= ready.offset + ready.bytes);
                        assert(ready.totalBytes + extra >= ready.totalBytes + capacity * (n / 2) * 2);
                    }
                }
            }
        }
    }
    assert(ForwardSwigluExtraBytes(6144, 131072, kSaveNone, true) == 768ULL * 1024 * 1024);
    assert(ForwardSwigluExtraBytes(8192, 65536, kSaveMxFp8, true) == 512ULL * 1024 * 1024);
    // Same raw-Gate alias/Up layout whether public saved mode is explicit or auto.
    assert(ResolveSavedGateUpMode(kSaveAuto, FLT_MAX) == kSaveMxFp8);
    assert(ResolveSavedGateUpMode(kSaveAuto, .05F) == kSaveBf16);
    std::cout << "BF16 forward SwiGLU policy/capacity PASS\n";
}
