
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

#include "white_list_filter.h"

#include <catch2/catch_test_macros.hpp>

#include "impl/allocator/safe_allocator.h"
#include "impl/bitset/fast_bitset.h"

using namespace vsag;

TEST_CASE("WhiteListFilter Basic Test For Bitset", "[ut][WhiteListFilter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t max_count = 100;

    auto bitset = std::make_shared<FastBitset>(allocator.get());
    for (int64_t i = 0; i < max_count; i++) {
        if (i % 3 == 0) {
            bitset->Set(i, true);
        }
    }

    auto bitset2 = std::make_shared<FastBitset>(allocator.get());
    for (int64_t i = 0; i < max_count; i++) {
        if (i % 3 == 1) {
            bitset2->Set(i, true);
        }
    }

    auto test_func = [&](const Filter* white, int value = 0) {
        for (int64_t i = 0; i < max_count; i++) {
            if (i % 3 == value) {
                REQUIRE(white->CheckValid(i));
            } else {
                REQUIRE_FALSE(white->CheckValid(i));
            }
        }
    };

    SECTION("shared ptr") {
        Filter* white = new WhiteListFilter(bitset);
        test_func(white);

        // TestUpdate from nullptr
        delete white;
        white = nullptr;
        WhiteListFilter::TryToUpdate(white, bitset.get());
        test_func(white);

        WhiteListFilter::TryToUpdate(white, bitset2.get());
        test_func(white, 1);

        delete white;
    }

    SECTION("raw ptr") {
        Filter* white = new WhiteListFilter(bitset.get());
        test_func(white);

        // TestUpdate from nullptr
        delete white;
        white = nullptr;
        WhiteListFilter::TryToUpdate(white, bitset.get());
        test_func(white);

        WhiteListFilter::TryToUpdate(white, bitset2.get());
        test_func(white, 1);

        delete white;
    }
}

TEST_CASE("WhiteListFilter Basic Test For IdFilterFuncType", "[ut][WhiteListFilter]") {
    int64_t max_count = 100;

    auto func = [](int64_t id) -> bool { return id % 3 == 0; };

    auto func2 = [](int64_t id) -> bool { return id % 3 == 1; };

    auto test_func = [&](const Filter* white, int value = 0) {
        for (int64_t i = 0; i < max_count; i++) {
            if (i % 3 == value) {
                REQUIRE(white->CheckValid(i));
            } else {
                REQUIRE_FALSE(white->CheckValid(i));
            }
        }
    };

    Filter* white = new WhiteListFilter(func);
    test_func(white);

    delete white;
    white = nullptr;
    WhiteListFilter::TryToUpdate(white, func);
    test_func(white);

    WhiteListFilter::TryToUpdate(white, func2);
    test_func(white, 1);

    REQUIRE(white->FilterDistribution() == Filter::Distribution::NONE);
    delete white;
}
