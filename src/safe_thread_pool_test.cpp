
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

#include "safe_thread_pool.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("SafeThreadPool Basic Test", "[ut][SafeThreadPool]") {
    auto thread_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    int data = 0;
    std::mutex m;
    thread_pool->SetPoolSize(4);
    thread_pool->SetQueueSizeLimit(6);
    int round = 10;
    for (int i = 0; i < round; ++i) {
        thread_pool->GeneralEnqueue(
            [&data, &m](int i) -> int {
                std::lock_guard lock(m);
                vsag::logger::info("current data:{}", data);
                data++;
                return i * i;
            },
            i);
        thread_pool->GeneralEnqueue(
            []() { throw std::runtime_error("throw a error in thread pool"); });
    }
    thread_pool->WaitUntilEmpty();
    REQUIRE(data == round);
}
