
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

#include "slow_task_timer.h"

#include <catch2/catch_test_macros.hpp>

using namespace vsag;

TEST_CASE("SlowTaskTimer Basic", "[ut][SlowTaskTimer]") {
    SECTION("construct and destruct with default threshold") {
        SlowTaskTimer timer("default-threshold");
        REQUIRE(timer.name == "default-threshold");
        REQUIRE(timer.threshold == 0);
    }

    SECTION("construct and destruct with custom threshold") {
        SlowTaskTimer timer("custom-threshold", 1000);
        REQUIRE(timer.name == "custom-threshold");
        REQUIRE(timer.threshold == 1000);
    }

    SECTION("construct and destruct without triggering logging") {
        SlowTaskTimer timer("high-threshold", 1000);
        REQUIRE(timer.threshold == 1000);
    }
}
