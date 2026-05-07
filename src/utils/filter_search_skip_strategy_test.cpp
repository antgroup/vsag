// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "filter_search_skip_strategy.h"

#include <vector>

#include "unittest.h"

using namespace vsag;

TEST_CASE("Filter search skip strategy parse", "[ut][filter_search_skip_strategy]") {
    REQUIRE(parse_filter_search_skip_strategy_type("random") ==
            FilterSearchSkipStrategyType::RANDOM);
    REQUIRE(parse_filter_search_skip_strategy_type("deterministic_accumulative") ==
            FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE);
    REQUIRE(std::string(filter_search_skip_strategy_type_to_string(
                FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE)) ==
            "deterministic_accumulative");
    REQUIRE_THROWS(parse_filter_search_skip_strategy_type("unknown"));
}

TEST_CASE("Accumulative filter search skip strategy is deterministic",
          "[ut][filter_search_skip_strategy]") {
    constexpr float valid_ratio = 0.5F;
    constexpr float skip_ratio = 0.8F;
    std::vector<bool> first_sequence;
    std::vector<bool> second_sequence;

    auto first_strategy = create_filter_search_skip_strategy(
        FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE, valid_ratio, skip_ratio);
    auto second_strategy = create_filter_search_skip_strategy(
        FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE, valid_ratio, skip_ratio);

    for (uint64_t i = 0; i < 20; ++i) {
        first_sequence.emplace_back(first_strategy->ShouldSkipFilterCheck());
        second_sequence.emplace_back(second_strategy->ShouldSkipFilterCheck());
    }

    REQUIRE(first_sequence == second_sequence);
    REQUIRE(first_sequence == std::vector<bool>{false, false, true,  false, true,  false, false,
                                                true,  false, true,  false, false, true,  false,
                                                true,  false, false, true,  false, true});
}

TEST_CASE("Accumulative filter search skip strategy edge cases",
          "[ut][filter_search_skip_strategy]") {
    SECTION("valid ratio one skips filter checks") {
        auto strategy = create_filter_search_skip_strategy(
            FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE, 1.0F, 0.8F);
        for (uint64_t i = 0; i < 10; ++i) {
            REQUIRE(strategy->ShouldSkipFilterCheck());
        }
    }

    SECTION("skip ratio zero skips invalid candidates") {
        auto strategy = create_filter_search_skip_strategy(
            FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE, 0.5F, 0.0F);
        for (uint64_t i = 0; i < 10; ++i) {
            REQUIRE_FALSE(strategy->ShouldSkipFilterCheck());
        }
    }
}
