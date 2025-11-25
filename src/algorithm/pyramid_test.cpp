
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

#include "algorithm/pyramid.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Split function tests", "[ut][pyramid]") {
    SECTION("Empty input string") {
        auto result = vsag::split("", ',');
        REQUIRE(result.empty());
    }

    SECTION("No delimiters in string") {
        auto result = vsag::split("hello", ',');
        REQUIRE(result == std::vector<std::string>{"hello"});
    }

    SECTION("Delimiter at start") {
        auto result = vsag::split(",hello,world", ',');
        REQUIRE(result == std::vector<std::string>{"hello", "world"});
    }

    SECTION("Delimiter at end") {
        auto result = vsag::split("hello,world,", ',');
        REQUIRE(result == std::vector<std::string>{"hello", "world"});
    }

    SECTION("Multiple consecutive delimiters") {
        auto result = vsag::split("a,,b,,,c", ',');
        REQUIRE(result == std::vector<std::string>{"a", "b", "c"});
    }

    SECTION("Normal split with multiple tokens") {
        auto result = vsag::split("one,two,three", ',');
        REQUIRE(result == std::vector<std::string>{"one", "two", "three"});
    }

    SECTION("All delimiters") {
        auto result = vsag::split(",,,", ',');
        REQUIRE(result.empty());
    }

    SECTION("Mixed delimiters and spaces") {
        auto result = vsag::split("  , hello,  world  ", ',');
        REQUIRE(result == std::vector<std::string>{"  ", " hello", "  world  "});
    }
}
