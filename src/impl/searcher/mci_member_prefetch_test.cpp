// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "impl/searcher/mci_member_prefetch.h"

#include <numeric>
#include <random>
#include <vector>

#include "catch2/catch_test_macros.hpp"

using namespace vsag;

TEST_CASE("MCI member prefetch preserves filtered visitation order and duplicates", "[ut][mci]") {
    std::mt19937 random(42);
    for (uint64_t total : {0, 1, 7, 127}) {
        std::vector<uint8_t> bitmap(total);
        for (auto& value : bitmap) {
            value = random() % 5 == 0 ? 7 : 0;
        }
        for (uint64_t count : {0, 1, 7, 16, 31, 32, 63, 64, 65, 201, 257}) {
            std::vector<uint32_t> members(count);
            for (auto& id : members) {
                id = random() % (total + 10);
            }
            std::vector<uint32_t> expected;
            for (auto id : members) {
                if (id < total and bitmap[id] != 0) {
                    expected.push_back(id);
                }
            }
            for (uint32_t mode = 0; mode <= 4; ++mode) {
                for (uint32_t block : {0, 1, 4, 16, 32, 64, 80}) {
                    for (uint32_t distance : {1, 16, 64, 256}) {
                        for (uint32_t locality = 0; locality <= 3; ++locality) {
                            MCIMemberPrefetch config{mode, distance, block, locality};
                            std::vector<uint32_t> actual;
                            VisitMCIMembers(
                                members.data(),
                                members.size(),
                                bitmap.data(),
                                total,
                                config,
                                [&](auto id) {
                                    if (id < total and bitmap[id] != 0) {
                                        actual.push_back(id);
                                    }
                                },
                                [&](auto id) {
                                    REQUIRE(id < total);
                                    REQUIRE(bitmap[id] != 0);
                                    actual.push_back(id);
                                });
                            REQUIRE(actual == expected);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("MCI prefetch falls back to the original visitor without an inner-id bitmap",
          "[ut][mci]") {
    const std::vector<uint32_t> members{0, 4, 4, 1, 9000};
    for (uint32_t mode = 0; mode <= 4; ++mode) {
        std::vector<uint32_t> visited;
        MCIMemberPrefetch config;
        config.mode = mode;
        VisitMCIMembers(
            members.data(),
            members.size(),
            nullptr,
            5,
            config,
            [&](auto id) { visited.push_back(id); },
            [&](auto) { FAIL("without bitmap cannot bypass the original filter"); });
        REQUIRE(visited == members);
    }
}

TEST_CASE("MCI experiment prefetch options have defaults and bounded sizes", "[ut][mci]") {
    const char* names[]{"VSAG_MCI_MEMBER_MODE",
                        "VSAG_MCI_MEMBER_DISTANCE",
                        "VSAG_MCI_MEMBER_BLOCK",
                        "VSAG_MCI_MEMBER_LOCALITY"};
    for (const auto* name : names) {
        unsetenv(name);
    }
    auto config = GetMCIMemberPrefetch();
    CHECK(config.mode == 4);
    CHECK(config.distance == 96);
    CHECK(config.block == 32);
    CHECK(config.locality == 1);
    for (const auto* name : names) {
        setenv(name, "999", 1);
    }
    config = GetMCIMemberPrefetch();
    CHECK(config.mode == 4);
    CHECK(config.distance == 256);
    CHECK(config.block == 64);
    CHECK(config.locality == 3);
    for (const auto* name : names) {
        setenv(name, "0", 1);
    }
    config = GetMCIMemberPrefetch();
    CHECK(config.mode == 0);
    CHECK(config.distance == 1);
    CHECK(config.block == 1);
    CHECK(config.locality == 0);
    for (const auto* name : names) {
        unsetenv(name);
    }
}
