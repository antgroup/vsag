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

#include "inner_id_wrapper_filter.h"

#include <memory>

#include "impl/allocator/safe_allocator.h"
#include "impl/bitset/fast_bitset.h"
#include "impl/label_table.h"
#include "unittest.h"
#include "white_list_filter.h"
using namespace vsag;

TEST_CASE("InnerIdWrapperFilter Basic Test", "[ut][InnerIdWrapperFilter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t max_count = 10;

    LabelTable label_table(allocator.get());
    for (int64_t i = 0; i < max_count; i++) {
        label_table.Insert(i, i * 100);
    }

    auto bitset = std::make_shared<FastBitset>(allocator.get());
    for (int64_t i = 0; i < max_count * 100; i++) {
        if (i % 200 == 0) {
            bitset->Set(i, true);
        }
    }

    auto inner_filter = std::make_shared<WhiteListFilter>(bitset);
    InnerIdWrapperFilter wrapper(inner_filter, label_table);

    for (int64_t i = 0; i < max_count; i++) {
        bool expected = (i % 2 == 0);
        REQUIRE(wrapper.CheckValid(i) == expected);
    }
}

TEST_CASE("InnerIdWrapperFilter ValidRatio Test", "[ut][InnerIdWrapperFilter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t max_count = 10;

    LabelTable label_table(allocator.get());
    for (int64_t i = 0; i < max_count; i++) {
        label_table.Insert(i, i);
    }

    auto func = [](int64_t id) -> bool { return id % 3 == 0; };
    auto inner_filter = std::make_shared<WhiteListFilter>(func);
    InnerIdWrapperFilter wrapper(inner_filter, label_table);

    float ratio = wrapper.ValidRatio();
    REQUIRE(ratio > 0.0F);
    REQUIRE(ratio <= 1.0F);
}
