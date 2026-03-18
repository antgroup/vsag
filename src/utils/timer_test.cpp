
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

#include "timer.h"

#include <catch2/catch_test_macros.hpp>
#include <thread>

using namespace vsag;

TEST_CASE("Timer Basic", "[ut][Timer]") {
    constexpr auto sleep_duration = std::chrono::milliseconds(20);

    SECTION("record elapsed time") {
        Timer timer;
        std::this_thread::sleep_for(sleep_duration);
        auto elapsed_ms = timer.Record();
        REQUIRE(elapsed_ms >= 5.0);
    }

    SECTION("threshold and overtime") {
        Timer timer;
        timer.SetThreshold(5.0);
        std::this_thread::sleep_for(sleep_duration);
        REQUIRE(timer.CheckOvertime());
    }

    SECTION("write back to reference in destructor") {
        double cost = 0.0;
        {
            Timer timer(cost);
            std::this_thread::sleep_for(sleep_duration);
        }
        REQUIRE(cost >= 5.0);
    }
}
