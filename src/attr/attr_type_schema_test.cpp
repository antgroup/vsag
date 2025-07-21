
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

#include "attr_type_schema.h"

#include <catch2/catch_all.hpp>

#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("AttrTypeSchema Basic Test", "[ut][AttrTypeSchema]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto map = std::make_unique<AttrTypeSchema>(allocator.get());

    map->SetTypeOfField("field_str", AttrValueType::STRING);
    map->SetTypeOfField("field_int64", AttrValueType::INT64);
    map->SetTypeOfField("field_uint64_t", AttrValueType::UINT64);
    map->SetTypeOfField("field_int32", AttrValueType::INT32);
    map->SetTypeOfField("field_uint32", AttrValueType::UINT32);
    map->SetTypeOfField("field_int16", AttrValueType::INT16);
    map->SetTypeOfField("field_uint16", AttrValueType::UINT16);
    map->SetTypeOfField("field_int8", AttrValueType::INT8);
    map->SetTypeOfField("field_uint8", AttrValueType::UINT8);

    REQUIRE(map->GetTypeOfField("field_str") == AttrValueType::STRING);
    REQUIRE(map->GetTypeOfField("field_int64") == AttrValueType::INT64);
    REQUIRE(map->GetTypeOfField("field_uint64_t") == AttrValueType::UINT64);
    REQUIRE(map->GetTypeOfField("field_int32") == AttrValueType::INT32);
    REQUIRE(map->GetTypeOfField("field_uint32") == AttrValueType::UINT32);
    REQUIRE(map->GetTypeOfField("field_int16") == AttrValueType::INT16);
    REQUIRE(map->GetTypeOfField("field_uint16") == AttrValueType::UINT16);
    REQUIRE(map->GetTypeOfField("field_int8") == AttrValueType::INT8);
    REQUIRE(map->GetTypeOfField("field_uint8") == AttrValueType::UINT8);

    REQUIRE_THROWS(map->GetTypeOfField("field_float"));

    map->SetTypeOfField("field_str", AttrValueType::INT8);
    REQUIRE(map->GetTypeOfField("field_str") == AttrValueType::INT8);
}
