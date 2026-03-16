
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

#include "window_result_queue.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace vsag;

TEST_CASE("WindowResultQueue Basic", "[ut][WindowResultQueue]") {
    WindowResultQueue queue;

    SECTION("empty queue returns zero") {
        REQUIRE(std::abs(queue.GetAvgResult()) < 1e-6F);
    }

    SECTION("average over inserted values") {
        queue.Push(1.0F);
        queue.Push(2.0F);
        queue.Push(3.0F);
        REQUIRE(std::abs(queue.GetAvgResult() - 2.0F) < 1e-6F);
    }

    SECTION("new pushes eventually replace old values") {
        queue.Push(1.0F);
        REQUIRE(std::abs(queue.GetAvgResult() - 1.0F) < 1e-6F);

        for (int i = 0; i < 200; ++i) {
            queue.Push(100.0F);
        }
        REQUIRE(queue.GetAvgResult() > 90.0F);
    }
}
