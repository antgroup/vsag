
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

#include "black_list_filter.h"

#include <catch2/catch_test_macros.hpp>

#include "impl/allocator/safe_allocator.h"
#include "impl/bitset/fast_bitset.h"

using namespace vsag;

TEST_CASE("BlackListFilter Basic Test For Bitset", "[ut][BlackListFilter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto bitset = std::make_shared<FastBitset>(allocator.get());
    int64_t max_count = 100;
    for (int64_t i = 0; i < max_count; i++) {
        if (i % 3 == 0) {
            bitset->Set(i, true);
        }
    }

    auto test_func = [&](std::shared_ptr<BlackListFilter>& black, int value = 0) {
        for (int64_t i = 0; i < max_count; i++) {
            if (i % 3 == value) {
                REQUIRE_FALSE(black->CheckValid(i));
            } else {
                REQUIRE(black->CheckValid(i));
            }
        }
    };

    SECTION("shared ptr") {
        auto black = std::make_shared<BlackListFilter>(bitset);
        test_func(black);
    }

    SECTION("raw ptr") {
        auto black = std::make_shared<BlackListFilter>(bitset.get());
        test_func(black);
    }
}

TEST_CASE("BlackListFilter Basic Test For IdFilterFuncType", "[ut][BlackListFilter]") {
    int64_t max_count = 100;

    auto func = [](int64_t id) -> bool { return id % 3 == 0; };

    auto test_func = [&](std::shared_ptr<BlackListFilter>& black, int value = 0) {
        for (int64_t i = 0; i < max_count; i++) {
            if (i % 3 == value) {
                REQUIRE_FALSE(black->CheckValid(i));
            } else {
                REQUIRE(black->CheckValid(i));
            }
        }
    };

    auto black = std::make_shared<BlackListFilter>(func);
    test_func(black);
}
