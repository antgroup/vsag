// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "datacell/sparse_read_execution.h"

#include <array>
#include <atomic>
#include <stdexcept>

#include "impl/searcher/search_execution_context.h"
#include "unittest.h"

namespace vsag {
TEST_CASE("Sparse read execution preserves range coverage and releases admission",
          "[ut][SparseReadExecution]") {
    const uint32_t background_workers = GENERATE(0U, 3U);
    SearchExecutionContext resources(background_workers);
    SparseReadExecution execution;
    execution.resources = &resources;
    execution.requested = 4;
    std::array<std::atomic<uint32_t>, 301> visits{};
    ReadSparseRanges(execution, visits.size(), [&](uint64_t first, uint64_t count) {
        for (uint64_t i = first; i < first + count; ++i) {
            ++visits[i];
        }
    });
    for (const auto& visit : visits) {
        REQUIRE(visit.load() == 1);
    }
    REQUIRE(resources.busy == 0);
    REQUIRE(resources.callers == 0);
    REQUIRE(execution.read_sum_ns == 0);
    REQUIRE(execution.worker_cpu_ns == 0);
#if HAVE_LIBAIO
    REQUIRE(execution.granted == std::min({4U, resources.limit + 1, resources.slots}));
#else
    REQUIRE(execution.granted == 1);
#endif
    SECTION("already concurrent single batch remains intact") {
        SparseReadExecution single;
        single.resources = &resources;
        single.requested = 4;
        uint64_t calls = 0;
        ReadSparseRanges(single, 100, [&](uint64_t first, uint64_t count) {
            REQUIRE(first == 0);
            REQUIRE(count == 100);
            ++calls;
        });
        REQUIRE(calls == 1);
        REQUIRE(single.granted == 1);
    }
    SECTION("failure drains and resource group can be reused") {
        std::atomic<uint32_t> active{0};
        REQUIRE_THROWS_AS(ReadSparseRanges(execution,
                                           301,
                                           [&](uint64_t, uint64_t) {
                                               struct Guard {
                                                   std::atomic<uint32_t>& value;
                                                   explicit Guard(std::atomic<uint32_t>& value)
                                                       : value(value) {
                                                       ++value;
                                                   }
                                                   ~Guard() {
                                                       --value;
                                                   }
                                               } guard(active);
                                               throw std::runtime_error("read failure");
                                           }),
                          std::runtime_error);
        REQUIRE(active.load() == 0);
        REQUIRE(resources.busy == 0);
        REQUIRE(resources.callers == 0);
        REQUIRE_NOTHROW(ReadSparseRanges(execution, 301, [](uint64_t, uint64_t) {}));
    }
    SECTION("nested pool call executes inline") {
        if (not resources.pool) {
            REQUIRE(execution.granted == 1);
            return;
        }
        resources.pool
            ->GeneralEnqueue([&] {
                SparseReadExecution nested;
                nested.resources = &resources;
                nested.requested = 4;
                uint64_t covered = 0;
                ReadSparseRanges(nested, 301, [&](uint64_t first, uint64_t count) {
                    if (first != 0 || count != 301)
                        throw std::logic_error("nested read split");
                    covered += count;
                });
                if (covered != 301 || nested.granted != 1)
                    throw std::logic_error("nested admission");
            })
            .get();
        REQUIRE(resources.busy == 0);
        REQUIRE(resources.callers == 0);
    }
}
}  // namespace vsag
