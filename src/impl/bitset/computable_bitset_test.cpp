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

#include "computable_bitset.h"

#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"
using namespace vsag;

TEST_CASE("ComputableBitset MakeInstance Test", "[ut][ComputableBitset]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    SECTION("Make SparseBitset") {
        auto bitset =
            ComputableBitset::MakeInstance(ComputableBitsetType::SparseBitset, allocator.get());
        REQUIRE(bitset != nullptr);
    }

    SECTION("Make FastBitset") {
        auto bitset =
            ComputableBitset::MakeInstance(ComputableBitsetType::FastBitset, allocator.get());
        REQUIRE(bitset != nullptr);
    }
}

TEST_CASE("ComputableBitset MakeRawInstance Test", "[ut][ComputableBitset]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    SECTION("Make Raw SparseBitset") {
        auto* bitset =
            ComputableBitset::MakeRawInstance(ComputableBitsetType::SparseBitset, allocator.get());
        REQUIRE(bitset != nullptr);
        delete bitset;
    }

    SECTION("Make Raw FastBitset") {
        auto* bitset =
            ComputableBitset::MakeRawInstance(ComputableBitsetType::FastBitset, allocator.get());
        REQUIRE(bitset != nullptr);
        delete bitset;
    }
}

TEST_CASE("ComputableBitset And with Vector Test", "[ut][ComputableBitset]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    auto bitset1 =
        ComputableBitset::MakeInstance(ComputableBitsetType::FastBitset, allocator.get());
    auto bitset2 =
        ComputableBitset::MakeInstance(ComputableBitsetType::FastBitset, allocator.get());
    auto bitset3 =
        ComputableBitset::MakeInstance(ComputableBitsetType::FastBitset, allocator.get());

    for (int64_t i = 0; i < 100; i++) {
        bitset1->Set(i, true);
        if (i % 2 == 0) {
            bitset2->Set(i, true);
        }
        if (i % 3 == 0) {
            bitset3->Set(i, true);
        }
    }

    std::vector<const ComputableBitset*> bitsets = {bitset2.get(), bitset3.get()};
    bitset1->And(bitsets);

    for (int64_t i = 0; i < 100; i++) {
        if (i % 2 == 0 && i % 3 == 0) {
            REQUIRE(bitset1->Test(i));
        } else {
            REQUIRE_FALSE(bitset1->Test(i));
        }
    }
}

TEST_CASE("ComputableBitset Or with Vector Test", "[ut][ComputableBitset]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    auto bitset1 =
        ComputableBitset::MakeInstance(ComputableBitsetType::FastBitset, allocator.get());
    auto bitset2 =
        ComputableBitset::MakeInstance(ComputableBitsetType::FastBitset, allocator.get());
    auto bitset3 =
        ComputableBitset::MakeInstance(ComputableBitsetType::FastBitset, allocator.get());

    for (int64_t i = 0; i < 100; i++) {
        if (i % 2 == 0) {
            bitset2->Set(i, true);
        }
        if (i % 3 == 0) {
            bitset3->Set(i, true);
        }
    }

    std::vector<const ComputableBitset*> bitsets = {bitset2.get(), bitset3.get()};
    bitset1->Or(bitsets);

    for (int64_t i = 0; i < 100; i++) {
        if (i % 2 == 0 || i % 3 == 0) {
            REQUIRE(bitset1->Test(i));
        } else {
            REQUIRE_FALSE(bitset1->Test(i));
        }
    }
}
