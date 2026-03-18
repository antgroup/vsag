
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

#include "lock_strategy.h"

#include <catch2/catch_test_macros.hpp>
#include <thread>

#include "impl/allocator/default_allocator.h"

using namespace vsag;

TEST_CASE("LockStrategy Basic", "[ut][LockStrategy]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("points mutex lock and shared lock") {
        PointsMutex mutex_array(8, allocator.get());
        mutex_array.Lock(0);
        mutex_array.Unlock(0);
        mutex_array.SharedLock(0);
        mutex_array.SharedUnlock(0);
        REQUIRE(mutex_array.GetMemoryUsage() > 0);
    }

    SECTION("resize updates memory usage") {
        PointsMutex mutex_array(4, allocator.get());
        auto before = mutex_array.GetMemoryUsage();
        mutex_array.Resize(10);
        auto after = mutex_array.GetMemoryUsage();
        REQUIRE(after > before);
        mutex_array.Resize(2);
        REQUIRE(mutex_array.GetMemoryUsage() < after);
    }

    SECTION("lock guards protect concurrent increment") {
        auto mutex_impl = std::make_shared<PointsMutex>(1, allocator.get());
        int counter = 0;
        constexpr int thread_num = 8;
        constexpr int loops = 1000;
        std::vector<std::thread> threads;
        threads.reserve(thread_num);
        for (int i = 0; i < thread_num; ++i) {
            threads.emplace_back([&]() {
                for (int j = 0; j < loops; ++j) {
                    LockGuard guard(mutex_impl, 0);
                    ++counter;
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        REQUIRE(counter == thread_num * loops);
    }

    SECTION("empty mutex no-op") {
        EmptyMutex mutex_array;
        mutex_array.Lock(0);
        mutex_array.Unlock(0);
        mutex_array.SharedLock(0);
        mutex_array.SharedUnlock(0);
        mutex_array.Resize(100);
        REQUIRE(mutex_array.GetMemoryUsage() == 0);
    }
}
