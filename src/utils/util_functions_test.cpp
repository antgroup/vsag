
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

#include "util_functions.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "impl/allocator/default_allocator.h"

using namespace vsag;

TEST_CASE("UtilFunctions Basic", "[ut][UtilFunctions]") {
    SECTION("format map") {
        std::unordered_map<std::string, std::string> mappings{{"name", "vsag"}, {"v", "1.0"}};
        std::string formatted = format_map("{name}-{v}", mappings);
        REQUIRE(formatted == "vsag-1.0");
    }

    SECTION("split string") {
        auto parts = split_string("a,,b", ',');
        REQUIRE(parts.size() == 3);
        REQUIRE(parts[0] == "a");
        REQUIRE(parts[1].empty());
        REQUIRE(parts[2] == "b");
    }

    SECTION("align up") {
        REQUIRE(align_up(7, 4) == 8);
        REQUIRE(align_up(8, 4) == 8);
    }

    SECTION("approx zero") {
        REQUIRE(is_approx_zero(1e-6F));
        REQUIRE_FALSE(is_approx_zero(1e-3F));
    }

    SECTION("next multiple of power of two") {
        REQUIRE(next_multiple_of_power_of_two(5, 2) == 8);
        REQUIRE(next_multiple_of_power_of_two(16, 4) == 16);
        REQUIRE_THROWS(next_multiple_of_power_of_two(1, 64));
    }

    SECTION("select k numbers") {
        auto selected = select_k_numbers(100, 10);
        REQUIRE(selected.size() == 10);
        std::unordered_set<int> dedup(selected.begin(), selected.end());
        REQUIRE(dedup.size() == 10);
        for (int v : selected) {
            REQUIRE(v >= 0);
            REQUIRE(v < 100);
        }
        REQUIRE_THROWS(select_k_numbers(10, 0));
        REQUIRE_THROWS(select_k_numbers(10, 11));
    }

    SECTION("string stream equality") {
        std::stringstream s1;
        std::stringstream s2;
        s1 << "abc";
        s2 << "abc";
        REQUIRE(check_equal_on_string_stream(s1, s2));

        std::stringstream s3;
        std::stringstream s4;
        s3 << "abc";
        s4 << "abd";
        REQUIRE_FALSE(check_equal_on_string_stream(s3, s4));
    }

    SECTION("base64 roundtrip") {
        std::string input = "vsag-base64-roundtrip";
        std::string encoded = base64_encode(input);
        std::string decoded = base64_decode(encoded);
        REQUIRE(decoded == input);
    }

    SECTION("create fast dataset") {
        auto allocator = std::make_shared<DefaultAllocator>();
        auto [dataset, dists, ids] = create_fast_dataset(4, allocator.get());
        REQUIRE(dataset != nullptr);
        REQUIRE(dataset->GetDim() == 4);
        REQUIRE(dataset->GetNumElements() == 1);
        REQUIRE(dists != nullptr);
        REQUIRE(ids != nullptr);
    }

    SECTION("current time string") {
        auto t = get_current_time();
        REQUIRE_FALSE(t.empty());
    }
}
