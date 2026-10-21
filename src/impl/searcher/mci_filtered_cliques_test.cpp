// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "impl/searcher/mci_filtered_cliques.h"

#include <numeric>
#include <random>
#include <thread>

#include "catch2/catch_test_macros.hpp"

using namespace vsag;

TEST_CASE("Filtered MCI packs exact intersections and drops singletons", "[ut][mci]") {
    std::mt19937 random(4242);
    MCIFilteredCliques<uint8_t> packed;
    for (uint64_t iteration = 0; iteration < 600; ++iteration) {
        const uint32_t total = 1 + random() % 100;
        const uint32_t count = 1 + random() % 140;
        std::vector<std::vector<uint32_t>> original(count), inverse(total);
        for (uint32_t c = 0; c < count; ++c) {
            for (uint32_t v = 0; v < total; ++v) {
                if (random() % 7 == 0) {
                    original[c].push_back(v);
                    inverse[v].push_back(c);
                }
            }
            std::shuffle(original[c].begin(), original[c].end(), random);
        }
        std::vector<uint32_t> offsets{0}, cids, valid;
        for (uint32_t v = 0; v < total; ++v) {
            cids.insert(cids.end(), inverse[v].begin(), inverse[v].end());
            offsets.push_back(cids.size());
            if (random() % 3 == 0) {
                valid.push_back(v);
            }
        }
        REQUIRE(
            packed.Build(offsets.data(), cids.data(), total, count, valid.data(), valid.size()));
        uint64_t incidences = 0, nonempty = 0, kept = 0, memberships = 0;
        const uint32_t* previous_end = packed.MemberStorage();
        // Groups are in first-discovery order; inspect that order to verify contiguous ranges.
        std::vector<uint32_t> group_order;
        std::vector<bool> seen(count, false);
        for (auto v : valid) {
            for (auto c : inverse[v]) {
                ++incidences;
                if (not seen[c]) {
                    seen[c] = true;
                    group_order.push_back(c);
                }
            }
        }
        for (auto c : group_order) {
            std::vector<uint32_t> expected;
            for (auto v : original[c]) {
                if (std::binary_search(valid.begin(), valid.end(), v)) {
                    expected.push_back(v);
                }
            }
            std::sort(expected.begin(), expected.end());
            nonempty += not expected.empty();
            auto span = packed.Get(c);
            if (expected.size() < 2) {
                CHECK(span.count == 0);
                continue;
            }
            ++kept;
            memberships += expected.size();
            CHECK(span.data == previous_end);
            REQUIRE(span.count == expected.size());
            CHECK(std::equal(expected.begin(), expected.end(), span.data));
            previous_end += span.count;
        }
        CHECK(packed.Incidences() == incidences);
        CHECK(packed.NonemptyCliques() == nonempty);
        CHECK(packed.KeptCliques() == kept);
        CHECK(packed.MemberCount() == memberships);
        CHECK(packed.Get(count + 1000).count == 0);
        for (uint32_t c = 0; c < count; ++c) {
            if (not seen[c]) {
                CHECK(packed.Get(c).count == 0);
            }
        }
    }
}

TEST_CASE("Filtered MCI reuses pools and rejects malformed input", "[ut][mci]") {
    MCIFilteredCliques<> packed;
    const uint32_t offsets[]{0, 2, 4, 5};
    const uint32_t cids[]{0, 1, 0, 1, 1};
    const uint32_t valid[]{0, 1, 2};
    REQUIRE(packed.Build(offsets, cids, 3, 2, valid, 3));
    const auto* storage = packed.MemberStorage();
    const auto capacity = packed.CapacityBytes();
    for (uint64_t q = 0; q < 100; ++q) {
        REQUIRE(packed.Build(offsets, cids, 3, 2, valid, 2));
        CHECK(packed.MemberStorage() == storage);
        CHECK(packed.CapacityBytes() == capacity);
        CHECK(packed.Get(1).count == 2);
    }
    REQUIRE(packed.Build(offsets, cids, 3, 2, nullptr, 0));
    CHECK(packed.Get(0).count == 0);
    CHECK(packed.MemberCount() == 0);
    CHECK_FALSE(packed.Build(nullptr, cids, 3, 2, valid, 3));
    CHECK_FALSE(packed.Build(offsets, cids, 3, 2, nullptr, 3));
    CHECK_FALSE(packed.Build(offsets, cids, 3, 2, valid, 4));
    CHECK_FALSE(packed.Build(offsets, cids, 3, uint64_t{1} << 32, valid, 3));
    const uint32_t invalid[]{0, 1, 3}, duplicate[]{0, 0}, unsorted[]{1, 0};
    CHECK_FALSE(packed.Build(offsets, cids, 3, 2, invalid, 3));
    CHECK_FALSE(packed.Build(offsets, cids, 3, 2, duplicate, 2));
    CHECK_FALSE(packed.Build(offsets, cids, 3, 2, unsorted, 2));
    CHECK_FALSE(packed.Build(offsets, nullptr, 3, 2, valid, 3));
    const uint32_t backwards[]{2, 0, 4, 5}, outside[]{0, 9, 4, 5};
    CHECK_FALSE(packed.Build(backwards, cids, 3, 2, valid, 3));
    CHECK_FALSE(packed.Build(outside, cids, 3, 2, valid, 3));
    const uint32_t bad_cids[]{0, 2, 0, 1, 1};
    CHECK_FALSE(packed.Build(offsets, bad_cids, 3, 2, valid, 3));
    CHECK(packed.Get(0).count == 0);
    REQUIRE(packed.Build(offsets, cids, 3, 2, valid, 3));
    CHECK(packed.Get(1).count == 3);
}

TEST_CASE("Filtered MCI query workers have independent pools", "[ut][mci]") {
    std::vector<std::thread> workers;
    std::vector<uint32_t> sizes(8);
    for (uint32_t t = 0; t < sizes.size(); ++t) {
        workers.emplace_back([&, t] {
            thread_local MCIFilteredCliques<> packed;
            const uint32_t offsets[]{0, 1, 2, 3};
            const uint32_t cids[]{0, 0, 0}, valid[]{0, 1, 2};
            for (uint64_t q = 0; q < 100; ++q) {
                const bool built = packed.Build(offsets, cids, 3, 1, valid, 2 + t % 2);
                sizes[t] = built ? packed.Get(0).count : 0;
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    for (uint32_t t = 0; t < sizes.size(); ++t) {
        CHECK(sizes[t] == 2 + t % 2);
    }
}
