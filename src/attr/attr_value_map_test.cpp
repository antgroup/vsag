
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

#include "attr/attr_value_map.h"

#include <catch2/catch_all.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("AttrValueMap insert and retrieve int32", "[ut][AttrValueMap]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto type = GENERATE(ComputableBitsetType::SparseBitset, ComputableBitsetType::FastBitset);
    AttrValueMap map(allocator.get(), type);
    int32_t value = 42;
    InnerIdType id = 5;

    map.Insert(value, id);
    BitsetPtr bitset = map.GetBitsetByValue(value);
    REQUIRE(bitset != nullptr);
    REQUIRE(bitset->Test(id) == true);
    REQUIRE(nullptr == map.GetBitsetByValue(999));
}

TEST_CASE("AttrValueMap serialize & deserialize", "[ut][AttrValueMap]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto type = GENERATE(ComputableBitsetType::SparseBitset, ComputableBitsetType::FastBitset);
    AttrValueMap map(allocator.get(), type);
    int32_t value = 42;
    InnerIdType id = 5;

    map.Insert(value, id);
    {
        auto bitset = map.GetBitsetByValue(value);
        REQUIRE(bitset != nullptr);
        REQUIRE(bitset->Test(id) == true);
        REQUIRE(nullptr == map.GetBitsetByValue(999));
    }

    auto dir = fixtures::TempDir("value_map");
    auto path = dir.GenerateRandomFile();
    std::ofstream ofs(path, std::ios::binary);
    IOStreamWriter writer(ofs);

    map.Serialize(writer);
    ofs.close();
    std::ifstream ifs(path, std::ios::binary);
    IOStreamReader reader(ifs);
    AttrValueMap map2(allocator.get(), type);
    map2.Deserialize(reader);
    ifs.close();
    {
        auto bitset = map2.GetBitsetByValue(value);
        REQUIRE(bitset != nullptr);
        REQUIRE(bitset->Test(id) == true);
        REQUIRE(nullptr == map2.GetBitsetByValue(999));
    }
}
