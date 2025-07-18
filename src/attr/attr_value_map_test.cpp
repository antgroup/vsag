
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

template <class T>
T
GetRandomValue() {
    if constexpr (std::is_integral<T>::value) {
        return random() % std::numeric_limits<T>::max();
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "abc";
    }
}

template <class T>
void
TestAttrValueMap() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto type = GENERATE(ComputableBitsetType::SparseBitset, ComputableBitsetType::FastBitset);
    AttrValueMap map(allocator.get(), type);
    T value = GetRandomValue<T>();
    InnerIdType id = random() % 10 + 1;

    map.Insert(value, id);
    auto manager = map.GetBitsetByValue(value);
    REQUIRE(manager != nullptr);
    REQUIRE(manager->GetOneBitset(0)->Test(id) == true);
    REQUIRE(nullptr == map.GetBitsetByValue(999));

    map.Insert(value, id, 3);
    manager = map.GetBitsetByValue(value);
    REQUIRE(manager != nullptr);
    REQUIRE(manager->GetOneBitset(2) == nullptr);
    REQUIRE(manager->GetOneBitset(3)->Test(id) == true);

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

    manager = map2.GetBitsetByValue(value);
    REQUIRE(manager != nullptr);
    REQUIRE(manager->GetOneBitset(2) == nullptr);
    REQUIRE(manager->GetOneBitset(3)->Test(id) == true);
    REQUIRE(nullptr == map2.GetBitsetByValue(999));
}

TEST_CASE("AttrValueMap Basic Test", "[ut][AttrValueMap]") {
    TestAttrValueMap<int64_t>();
    TestAttrValueMap<int32_t>();
    TestAttrValueMap<int16_t>();
    TestAttrValueMap<int8_t>();
    TestAttrValueMap<uint64_t>();
    TestAttrValueMap<uint32_t>();
    TestAttrValueMap<uint16_t>();
    TestAttrValueMap<uint8_t>();
    TestAttrValueMap<std::string>();
}
