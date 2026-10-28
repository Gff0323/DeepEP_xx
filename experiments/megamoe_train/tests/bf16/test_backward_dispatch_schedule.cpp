#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <iostream>
#define __aicore__
#define __gm__
#define PRODUCER_PHASE_HOST_TEST 1
namespace AscendC {
enum class HardEvent { S_MTE2, MTE2_S };
template <HardEvent E>
void SetFlag(int)
{
}
template <HardEvent E>
void WaitFlag(int)
{
}
inline int32_t ReadGmByPassDCache(const int32_t* p)
{
    return *p;
}
inline void WriteGmByPassDCache(int32_t* p, int32_t x)
{
    *p = x;
}
inline void AtomicAdd(int32_t* p, int32_t x)
{
    *p += x;
}
}  // namespace AscendC
#include "backward_dispatch_ready.h"
#include "backward_dispatch_schedule.h"
using namespace MegaMoeBF16BackwardCatlassImpl;
int main()
{
    std::mt19937 rng(20261024);
    uint64_t tiles = 0, tails = 0;
    unsigned cases = 0;
    for (unsigned c : {28U, 32U, 36U})
        for (unsigned s : {1U, 4U, 8U, 16U, 32U, 64U, 128U, 256U})
            for (unsigned mode = 0; mode < 9; ++mode) {
                const unsigned cs = std::min(s, c);
                const unsigned p = std::max(1U, c / s), chunks = (s + cs - 1) / cs;
                std::vector<int32_t> ends(s), progress(s * p * 16), flags(16);
                unsigned total = 0;
                for (unsigned i = 0; i < s; ++i) {
                    unsigned n = 32768 / s;
                    if (mode) n = rng() % (2 * n + 1);
                    if (mode == 2 && i == 0) n += 32768;
                    if (mode == 3 && i % 7) n = 0;
                    if (mode == 5) n = i % 3 == 0 ? 1U : 0U;
                    if (mode == 6) n = i == s - 1 ? 32769U : 0U;
                    if (mode == 7) n = 1U + rng() % 31U;
                    if (mode == 8) n = 257U + ((i + 1) % cs == 0 ? 1U : 0U);
                    total += n;
                    ends[i] = total;
                }
                if (!total) {
                    total = 1;
                    ends.back() = 1;
                }
                const unsigned m = mode == 4 ? 1 + rng() % total : total;
                const unsigned cap = mode == 4 ? m + rng() % (total - m + 1) : total;
                BackwardDispatchReadyView v;
                v.progressBase = progress.data();
                v.cumsum = ends.data();
                v.sources = s;
                v.producersPerSource = p;
                v.experts = 1;
                v.expert = 0;
                v.rowLimit = cap;
                v.enabled = true;
                std::vector<unsigned> order(s);
                for (unsigned i = 0; i < s; ++i) order[i] = i;
                std::shuffle(order.begin(), order.end(), rng);
                for (auto i : order) PrepareProducerPhaseSource(v, c, m, i, flags.data());
                // Publish all data for the actual row-wait query checks (not a speed model).
                for (unsigned i = 0; i < s; ++i) {
                    unsigned begin = i ? v.End(i - 1) : 0U, z = v.End(i), band = (z - begin + p - 1) / p;
                    for (unsigned owner = 0; owner < p; ++owner) {
                        unsigned a = begin + owner * band;
                        *v.Slot(i * p + owner) = a < z ? std::min(band, z - a) : 0;
                    }
                }
                const unsigned mt = (m + 127) / 128;
                std::vector<std::vector<unsigned>> cls(chunks * 11);
                for (unsigned row = 0; row < mt; ++row) {
                    unsigned x = row * 128, i = 0;
                    while (v.End(i) <= x) ++i;
                    unsigned a = i ? v.End(i - 1) : 0, z = v.End(i), b = (z - a + p - 1) / p, owner = (x - a) / b;
                    unsigned stop = std::min(std::min(a + (owner + 1) * b, z), m), phase = 10;
                    if (x + 128 <= stop) {
                        unsigned needed = x + 128 - a - owner * b;
                        for (unsigned q = 0; q < 10; ++q)
                            if (needed <= (256U << q)) {
                                phase = q;
                                break;
                            }
                    }
                    cls[(i / cs) * 11 + phase].push_back(row);
                }
                for (unsigned nt : {1U, 3U, 12U, 16U, 17U}) {
                    std::vector<ProducerPhaseMath::Coord> expected;
                    std::vector<unsigned> crossTails;
                    auto emit = [&](const std::vector<unsigned>& rows) {
                        unsigned width = 3;

                        // Independent loop enumeration, rather than division/inverse mapping.
                        for (unsigned n = 0, stripe = 0; n < nt; n += width, ++stripe)
                            for (unsigned r = 0; r < rows.size(); ++r)
                                for (unsigned col = n; col < std::min(n + width, nt); ++col)
                                    expected.push_back({rows[stripe % 2 ? rows.size() - 1 - r : r], col});
                    };
                    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
                        unsigned lastSource = std::min(s, (chunk + 1) * cs) - 1,
                                 groupEnd = std::min(m, v.End(lastSource));
                        unsigned beginM = chunk ? (std::min(m, v.End(chunk * cs - 1)) + 127) / 128 : 0U,
                                 endM = (groupEnd + 127) / 128;
                        bool tail = groupEnd < m && groupEnd % 128 && endM > beginM;
                        std::vector<std::vector<unsigned>> rows(11);
                        unsigned nonempty = 0;
                        std::vector<unsigned> joined;
                        for (unsigned q = 0; q < 11; ++q) {
                            rows[q] = cls[chunk * 11 + q];
                            if (q == 10 && tail) {
                                assert(!rows[q].empty() && rows[q].back() == endM - 1);
                                rows[q].pop_back();
                            }
                            nonempty += !rows[q].empty();
                            joined.insert(joined.end(), rows[q].begin(), rows[q].end());
                        }
                        if (nonempty == 1) {
                            std::sort(joined.begin(), joined.end());
                            emit(joined);
                        } else
                            for (auto& r : rows)
                                if (!r.empty()) emit(r);
                        if (tail) {
                            for (unsigned n = 0; n < nt; ++n) expected.push_back({endM - 1, n});
                            crossTails.push_back(endM - 1);
                            ++tails;
                        }
                    }
                    assert(expected.size() == mt * nt);
                    std::vector<unsigned> seen(mt * nt);
                    unsigned start = rng() % c;
                    for (unsigned core = 0; core < c; ++core) {
                        ProducerPhaseSchedule schedule;
                        schedule.Init(v, flags.data(), c, m, nt * 256);
                        assert(schedule.Enabled());
                        unsigned first = (core + c - start) % c;
                        for (unsigned j = first; j < mt * nt; j += c) {
                            auto t = schedule.Get(j);
                            auto ex = expected[j];
                            assert(t.m == ex.m && t.n == ex.n);
                            assert(!seen[t.m * nt + t.n]++);
                            unsigned row = t.m * 128, rows = std::min(128U, m - row);
                            if (t.known) {
                                assert(row >= t.bandBegin && row < t.bandEnd);
                                WaitProducerPhaseRows(v, t, row, rows);
                            } else
                                v.WaitRows(row, rows);
                            ++tiles;
                        }
                    }
                    assert(std::all_of(seen.begin(), seen.end(), [](unsigned n) { return n == 1; }));
                    ++cases;
                }
            }
    std::cout << "[BACKWARD_DISPATCH_SCHEDULE] PASS cases=" << cases << " tiles=" << tiles << " tail_groups=" << tails
              << "\n";
}
