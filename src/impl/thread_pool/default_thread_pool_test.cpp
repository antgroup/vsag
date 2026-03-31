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

#include "default_thread_pool.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>

using namespace vsag;

TEST_CASE("DefaultThreadPool Basic Test", "[ut][DefaultThreadPool]") {
    DefaultThreadPool pool(4);

    SECTION("Enqueue and Execute") {
        std::atomic<int> counter{0};

        for (int i = 0; i < 10; ++i) {
            pool.Enqueue([&counter]() { counter++; });
        }

        pool.WaitUntilEmpty();
        REQUIRE(counter == 10);
    }
}

TEST_CASE("DefaultThreadPool SetPoolSize Test", "[ut][DefaultThreadPool]") {
    DefaultThreadPool pool(2);

    std::atomic<int> counter{0};
    for (int i = 0; i < 5; ++i) {
        pool.Enqueue([&counter]() { counter++; });
    }
    pool.WaitUntilEmpty();
    REQUIRE(counter == 5);

    pool.SetPoolSize(4);

    counter = 0;
    for (int i = 0; i < 10; ++i) {
        pool.Enqueue([&counter]() { counter++; });
    }
    pool.WaitUntilEmpty();
    REQUIRE(counter == 10);
}

TEST_CASE("DefaultThreadPool SetQueueSizeLimit Test", "[ut][DefaultThreadPool]") {
    DefaultThreadPool pool(2);
    pool.SetQueueSizeLimit(100);

    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        pool.Enqueue([&counter]() { counter++; });
    }
    pool.WaitUntilEmpty();
    REQUIRE(counter == 50);
}
