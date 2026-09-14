
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

#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

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

    cell.Insert(attrSet, inner_id, bucket_id);

    auto managers = cell.GetBitsetsByAttr(*attr);
    REQUIRE(managers.size() == 1);
    REQUIRE(managers[0]->GetOneBitset(bucket_id)->Test(inner_id) == true);
    REQUIRE(managers[0]->GetOneBitset(bucket_id + 1) == nullptr);
    REQUIRE(managers[0]->GetOneBitset(bucket_id - 1) == nullptr);
    REQUIRE(managers[0]->GetOneBitset(bucket_id)->Test(inner_id + 1) == false);

    // invalid attr name
    AttributeValue<int32_t> attr2;
    attr2.name_ = "age2";
    attr2.GetValue().emplace_back(30);
    managers = cell.GetBitsetsByAttr(attr2);
    REQUIRE(managers.size() == 1);
    REQUIRE(managers[0] == nullptr);

    // invalid value
    attr->GetValue().emplace_back(47);
    managers = cell.GetBitsetsByAttr(*attr);
    REQUIRE(managers.size() == 2);
    REQUIRE(managers[0]->GetOneBitset(bucket_id)->Test(inner_id) == true);
    REQUIRE(managers[1] == nullptr);

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
    cell.Insert(attrSet, inner_id, bucket_id);

    auto managers = cell.GetBitsetsByAttr(*attr);
    REQUIRE(managers.size() == 3);
    for (auto& ms : managers) {
        REQUIRE(ms->GetOneBitset(bucket_id)->Test(inner_id) == true);
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
    cell.Insert(attrSet, inner_id, bucket_id);
    REQUIRE(cell.GetTypeOfField("str") == AttrValueType::STRING);

    for (auto* attr : attrSet.attrs_) {
        auto managers = cell.GetBitsetsByAttr(*attr);
        REQUIRE(managers.size() == 1);
        REQUIRE(managers[0]->GetOneBitset(bucket_id)->Test(inner_id) == true);
        REQUIRE(managers[0]->GetOneBitset(bucket_id)->Test(inner_id - 1) == false);
    }

    AttributeBucketInvertedDataCell cell2(allocator.get());
    test_serializion(cell, cell2);

    for (auto* attr : attrSet.attrs_) {
        auto managers = cell2.GetBitsetsByAttr(*attr);
        REQUIRE(managers.size() == 1);
        REQUIRE(managers[0]->GetOneBitset(bucket_id)->Test(inner_id) == true);
        REQUIRE(managers[0]->GetOneBitset(bucket_id)->Test(inner_id - 1) == false);
    }
    REQUIRE(cell2.GetTypeOfField("str") == AttrValueType::STRING);
}

TEST_CASE("Attribute update introduces a new field", "[ut][AttributeBucketInvertedDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    AttributeBucketInvertedDataCell cell(allocator.get());
    AttributeValue<int64_t> original;
    original.name_ = "region";
    original.GetValue() = {10};
    AttributeSet origin;
    origin.attrs_ = {&original};
    cell.Insert(origin, 3, 2);
    cell.Insert(origin, 7, 2);

    AttributeValue<int64_t> replacement;
    replacement.name_ = "region";
    replacement.GetValue() = {20};
    AttributeValue<std::string> introduced;
    introduced.name_ = "new_field";
    introduced.GetValue() = {"value"};
    AttributeSet updated;
    updated.attrs_ = {&replacement, &introduced};
    cell.UpdateBitsetsByAttr(updated, 3, 2, origin);

    REQUIRE(cell.GetTypeOfField("new_field") == AttrValueType::STRING);
    auto old = cell.GetBitsetsByAttr(original);
    REQUIRE_FALSE(old[0]->GetOneBitset(2)->Test(3));
    REQUIRE(old[0]->GetOneBitset(2)->Test(7));
    auto added = cell.GetBitsetsByAttr(introduced);
    REQUIRE(added[0]->GetOneBitset(2)->Test(3));
    REQUIRE_FALSE(added[0]->GetOneBitset(2)->Test(7));
    REQUIRE(added[0]->GetOneBitset(1) == nullptr);
    auto replaced = cell.GetBitsetsByAttr(replacement);
    REQUIRE(replaced[0]->GetOneBitset(2)->Test(3));

    AttributeBucketInvertedDataCell restored(allocator.get());
    test_serializion(cell, restored);
    REQUIRE(restored.GetTypeOfField("new_field") == AttrValueType::STRING);
    REQUIRE(restored.GetBitsetsByAttr(introduced)[0]->GetOneBitset(2)->Test(3));

    cell.UpdateBitsetsByAttr(updated, 3, 2, updated);
    REQUIRE(cell.GetBitsetsByAttr(introduced)[0]->GetOneBitset(2)->Test(3));

    AttributeValue<int32_t> wrong_type;
    wrong_type.name_ = "new_field";
    wrong_type.GetValue() = {42};
    AttributeSet invalid;
    invalid.attrs_ = {&replacement, &wrong_type};
    REQUIRE_THROWS_AS(cell.UpdateBitsetsByAttr(invalid, 3, 2, updated), VsagException);
    REQUIRE(cell.GetTypeOfField("new_field") == AttrValueType::STRING);
    REQUIRE(cell.GetBitsetsByAttr(introduced)[0]->GetOneBitset(2)->Test(3));
    REQUIRE(cell.GetBitsetsByAttr(replacement)[0]->GetOneBitset(2)->Test(3));
    REQUIRE_THROWS_AS(cell.UpdateBitsetsByAttr(updated, 3, 2, invalid), VsagException);
    REQUIRE(cell.GetBitsetsByAttr(introduced)[0]->GetOneBitset(2)->Test(3));

    wrong_type.name_ = "brand_new";
    AttributeValue<std::string> conflicting;
    conflicting.name_ = "brand_new";
    conflicting.GetValue() = {"different type"};
    invalid.attrs_ = {&wrong_type, &conflicting};
    REQUIRE_THROWS_AS(cell.UpdateBitsetsByAttr(invalid, 3, 2, updated), VsagException);
    REQUIRE_THROWS_AS(cell.GetTypeOfField("brand_new"), VsagException);
    REQUIRE(cell.GetBitsetsByAttr(introduced)[0]->GetOneBitset(2)->Test(3));
}
