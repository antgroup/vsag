
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

#include "attribute_bucket_inverted_datacell.h"

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("AttributeBucketInvertedDataCell insert single attribute",
          "[ut][AttributeBucketInvertedDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    AttributeBucketInvertedDataCell cell(allocator.get());

    AttributeValue<int32_t>* attr = new AttributeValue<int32_t>();
    attr->name_ = "age";
    attr->GetValue().emplace_back(30);

    AttributeSet attrSet;
    attrSet.attrs_.emplace_back(attr);

    InnerIdType inner_id = 100;

    BucketIdType bucket_id = 29;

    cell.InsertWithBucket(attrSet, inner_id, bucket_id);

    auto bitsets = cell.GetBitsetsByAttrAndBucketId(*attr, bucket_id);
    REQUIRE(bitsets.size() == 1);
    REQUIRE(bitsets[0]->Test(inner_id) == true);
    REQUIRE(bitsets[0]->Test(inner_id + 1) == false);

    // invalid bucket id
    bitsets = cell.GetBitsetsByAttrAndBucketId(*attr, bucket_id + 1);
    REQUIRE(bitsets.size() == 1);
    REQUIRE(bitsets[0] == nullptr);

    // invalid attr name
    AttributeValue<int32_t> attr2;
    attr2.name_ = "age2";
    attr2.GetValue().emplace_back(30);
    bitsets = cell.GetBitsetsByAttrAndBucketId(attr2, bucket_id);
    REQUIRE(bitsets.size() == 1);
    REQUIRE(bitsets[0] == nullptr);

    // invalid value
    attr->GetValue().emplace_back(47);
    bitsets = cell.GetBitsetsByAttrAndBucketId(*attr, bucket_id);
    REQUIRE(bitsets.size() == 2);
    REQUIRE(bitsets[0]->Test(inner_id) == true);
    REQUIRE(bitsets[1] == nullptr);

    delete attr;
}

TEST_CASE("AttributeBucketInvertedDataCell insert multiple values",
          "[ut][AttributeBucketInvertedDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    AttributeBucketInvertedDataCell cell(allocator.get());

    AttributeValue<int32_t>* attr = new AttributeValue<int32_t>();
    attr->name_ = "scores";
    attr->GetValue() = {85, 90, 95};

    AttributeSet attrSet;
    attrSet.attrs_.emplace_back(attr);

    InnerIdType inner_id = 5;
    BucketIdType bucket_id = 11;
    cell.InsertWithBucket(attrSet, inner_id, bucket_id);

    auto bitsets = cell.GetBitsetsByAttrAndBucketId(*attr, bucket_id);
    REQUIRE(bitsets.size() == 3);
    for (auto& bs : bitsets) {
        REQUIRE(bs->Test(inner_id) == true);
    }

    delete attr;
}

TEST_CASE("AttributeBucketInvertedDataCell insert various types",
          "[ut][AttributeBucketInvertedDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    AttributeBucketInvertedDataCell cell(allocator.get());

    auto attr_i8 = std::make_unique<AttributeValue<int8_t>>();
    attr_i8->name_ = "i8";
    attr_i8->GetValue().emplace_back(-10);

    auto attr_u16 = std::make_unique<AttributeValue<uint16_t>>();
    attr_u16->name_ = "u16";
    attr_u16->GetValue().emplace_back(1000);

    auto attr_str = std::make_unique<AttributeValue<std::string>>();
    attr_str->name_ = "str";
    attr_str->GetValue().emplace_back("test");

    AttributeSet attrSet;
    attrSet.attrs_.emplace_back(attr_i8.get());
    attrSet.attrs_.emplace_back(attr_u16.get());
    attrSet.attrs_.emplace_back(attr_str.get());

    InnerIdType inner_id = 99;
    BucketIdType bucket_id = 53;
    cell.InsertWithBucket(attrSet, inner_id, bucket_id);
    REQUIRE(cell.GetTypeOfField("str") == AttrValueType::STRING);

    for (auto* attr : attrSet.attrs_) {
        auto bitsets = cell.GetBitsetsByAttrAndBucketId(*attr, bucket_id);
        REQUIRE(bitsets.size() == 1);
        REQUIRE(bitsets[0]->Test(inner_id) == true);
        REQUIRE(bitsets[0]->Test(inner_id - 1) == false);
    }

    auto dir = fixtures::TempDir("attr");
    auto path = dir.GenerateRandomFile();
    std::ofstream ofs(path, std::ios::binary);
    IOStreamWriter writer(ofs);

    cell.Serialize(writer);
    ofs.close();
    std::ifstream ifs(path, std::ios::binary);
    IOStreamReader reader(ifs);
    AttributeBucketInvertedDataCell cell2(allocator.get());
    cell2.Deserialize(reader);
    ifs.close();
    for (auto* attr : attrSet.attrs_) {
        auto bitsets = cell2.GetBitsetsByAttrAndBucketId(*attr, bucket_id);
        REQUIRE(bitsets.size() == 1);
        REQUIRE(bitsets[0]->Test(inner_id) == true);
        REQUIRE(bitsets[0]->Test(inner_id - 1) == false);
    }
    REQUIRE(cell2.GetTypeOfField("str") == AttrValueType::STRING);
}
