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

#include "combined_filter.h"

#include <memory>

#include "impl/allocator/safe_allocator.h"
#include "impl/bitset/fast_bitset.h"
#include "unittest.h"
#include "white_list_filter.h"
using namespace vsag;

TEST_CASE("CombinedFilter Basic Test", "[ut][CombinedFilter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t max_count = 100;

    auto bitset1 = std::make_shared<FastBitset>(allocator.get());
    auto bitset2 = std::make_shared<FastBitset>(allocator.get());

    for (int64_t i = 0; i < max_count; i++) {
        if (i % 2 == 0) {
            bitset1->Set(i, true);
        }
        if (i % 3 == 0) {
            bitset2->Set(i, true);
        }
    }

    auto filter1 = std::make_shared<WhiteListFilter>(bitset1);
    auto filter2 = std::make_shared<WhiteListFilter>(bitset2);

    SECTION("Empty CombinedFilter") {
        CombinedFilter combined;
        REQUIRE(combined.IsEmpty());

        for (int64_t i = 0; i < max_count; i++) {
            REQUIRE(combined.CheckValid(i));
        }
    }

    SECTION("Single Filter") {
        CombinedFilter combined;
        combined.AppendFilter(filter1);
        REQUIRE_FALSE(combined.IsEmpty());

        for (int64_t i = 0; i < max_count; i++) {
            if (i % 2 == 0) {
                REQUIRE(combined.CheckValid(i));
            } else {
                REQUIRE_FALSE(combined.CheckValid(i));
            }
        }
    }

    SECTION("Two Filters AND") {
        CombinedFilter combined;
        combined.AppendFilter(filter1);
        combined.AppendFilter(filter2);

        for (int64_t i = 0; i < max_count; i++) {
            if (i % 2 == 0 && i % 3 == 0) {
                REQUIRE(combined.CheckValid(i));
            } else {
                REQUIRE_FALSE(combined.CheckValid(i));
            }
        }
    }

    SECTION("Append nullptr") {
        CombinedFilter combined;
        combined.AppendFilter(nullptr);
        REQUIRE(combined.IsEmpty());
    }

    SECTION("ValidRatio") {
        CombinedFilter combined;
        combined.AppendFilter(filter1);
        combined.AppendFilter(filter2);

        float ratio = combined.ValidRatio();
        REQUIRE(ratio > 0.0F);
        REQUIRE(ratio <= 1.0F);
    }
}
