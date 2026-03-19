

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

#include "spsc_queue.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("SPSCQueue Basic Operations", "[ut][spsc_queue]") {
    vsag::SPSCQueue<int, 4> queue;

    SECTION("Push and Pop") {
        int value;
        REQUIRE_FALSE(queue.Pop(value));

        REQUIRE(queue.Push(1));
        REQUIRE(queue.Push(2));
        REQUIRE(queue.Push(3));

        REQUIRE(queue.Pop(value));
        REQUIRE(value == 1);
        REQUIRE(queue.Pop(value));
        REQUIRE(value == 2);
        REQUIRE(queue.Pop(value));
        REQUIRE(value == 3);
        REQUIRE_FALSE(queue.Pop(value));
    }

    SECTION("Push with Move") {
        std::string str = "test";
        vsag::SPSCQueue<std::string, 4> str_queue;
        REQUIRE(str_queue.Push(std::move(str)));

        std::string out;
        REQUIRE(str_queue.Pop(out));
        REQUIRE(out == "test");
    }

    SECTION("Queue Full") {
        vsag::SPSCQueue<int, 4> small_queue;
        REQUIRE(small_queue.Push(1));
        REQUIRE(small_queue.Push(2));
        REQUIRE(small_queue.Push(3));
        REQUIRE_FALSE(small_queue.Push(4));

        int value;
        REQUIRE(small_queue.Pop(value));
        REQUIRE(small_queue.Push(4));
    }

    SECTION("Push Const Reference") {
        const int val = 42;
        REQUIRE(queue.Push(val));

        int out;
        REQUIRE(queue.Pop(out));
        REQUIRE(out == 42);
    }
}
