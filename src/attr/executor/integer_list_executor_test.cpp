
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

#include "integer_list_executor.h"

#include <catch2/catch_test_macros.hpp>

#include "attr/expression_visitor.h"
#include "executor_test.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

template <typename T>
static void
TestAttributeWithoutBucket(const std::string& name,
                           const std::vector<T>& values,
                           int index,
                           Allocator* allocator,
                           AttrInvertedInterfacePtr sparse_attr_index) {
    auto none_interact_vec = GetNoneInteractValues(values, name);
    auto query = CreateMultiInString(name, none_interact_vec);
    auto expr = AstParse(query);
    auto executor = std::make_shared<IntegerListExecutor>(allocator, expr, sparse_attr_index);
    auto filter = executor->Run();
    REQUIRE(filter->CheckValid(index) == false);
    REQUIRE(executor->only_bitset_ == true);

    query = CreateMultiNotInString(name, none_interact_vec);
    expr = AstParse(query);
    executor = std::make_shared<IntegerListExecutor>(allocator, expr, sparse_attr_index);
    filter = executor->Run();
    REQUIRE(filter->CheckValid(index) == true);
    REQUIRE(executor->only_bitset_ == false);

    auto interact_vec = GetInteractValues(values, name);
    query = CreateMultiNotInString(name, interact_vec);
    expr = AstParse(query);
    executor = std::make_shared<IntegerListExecutor>(allocator, expr, sparse_attr_index);
    filter = executor->Run();
    REQUIRE(filter->CheckValid(index) == false);
    REQUIRE(executor->only_bitset_ == false);

    query = CreateMultiInString(name, interact_vec);
    expr = AstParse(query);
    executor = std::make_shared<IntegerListExecutor>(allocator, expr, sparse_attr_index);
    filter = executor->Run();
    REQUIRE(filter->CheckValid(index) == true);
    REQUIRE(executor->only_bitset_ == true);
}

TEST_CASE("IntegerListExecutor Normal Without Bucket", "[ut][IntegerListExecutor]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto sparse_attr_index = AttributeInvertedInterface::MakeInstance(allocator.get(), false);

    std::vector<AttributeSet> attr_sets;
    for (int i = 0; i < 20; ++i) {
        attr_sets.emplace_back(ExecutorTest::MockAttrSet());
    }
    int idx = 0;
    for (auto& attr_set : attr_sets) {
        sparse_attr_index->Insert(attr_set, idx);
        idx++;
    }

    for (int i = 0; i < 20; ++i) {
        auto& attr_set = attr_sets[i];
        for (auto& attr_ptr : attr_set.attrs_) {
            auto name = attr_ptr->name_;
            auto type_str = split_string(name, '_')[0];
            if (type_str == "i32") {
                auto vec = GetValues<int32_t>(attr_ptr);
                TestAttributeWithoutBucket<int32_t>(
                    name, vec, i, allocator.get(), sparse_attr_index);
            } else if (type_str == "i64") {
                auto vec = GetValues<int64_t>(attr_ptr);
                TestAttributeWithoutBucket<int64_t>(
                    name, vec, i, allocator.get(), sparse_attr_index);
            } else if (type_str == "i8") {
                auto vec = GetValues<int8_t>(attr_ptr);
                TestAttributeWithoutBucket<int8_t>(
                    name, vec, i, allocator.get(), sparse_attr_index);
            } else if (type_str == "i16") {
                auto vec = GetValues<int16_t>(attr_ptr);
                TestAttributeWithoutBucket<int16_t>(
                    name, vec, i, allocator.get(), sparse_attr_index);
            } else if (type_str == "u8") {
                auto vec = GetValues<uint8_t>(attr_ptr);
                TestAttributeWithoutBucket<uint8_t>(
                    name, vec, i, allocator.get(), sparse_attr_index);
            } else if (type_str == "u16") {
                auto vec = GetValues<uint16_t>(attr_ptr);
                TestAttributeWithoutBucket<uint16_t>(
                    name, vec, i, allocator.get(), sparse_attr_index);
            } else if (type_str == "u32") {
                auto vec = GetValues<uint32_t>(attr_ptr);
                TestAttributeWithoutBucket<uint32_t>(
                    name, vec, i, allocator.get(), sparse_attr_index);
            } else if (type_str == "u64") {
                auto vec = GetValues<uint64_t>(attr_ptr);
                TestAttributeWithoutBucket<uint64_t>(
                    name, vec, i, allocator.get(), sparse_attr_index);
            }
        }
    }
    for (auto& attr_set : attr_sets) {
        ExecutorTest::DeleteAttrSet(attr_set);
    }
}

template <typename T>
static void
TestAttributeWithBucket(const std::string& name,
                        const std::vector<T>& values,
                        int index,
                        Allocator* allocator,
                        AttrInvertedInterfacePtr sparse_attr_index) {
    auto none_interact_vec = GetNoneInteractValues(values, name);
    auto query = CreateMultiInString(name, none_interact_vec);
    auto expr = AstParse(query);
    auto executor = std::make_shared<IntegerListExecutor>(allocator, expr, sparse_attr_index);
    auto filter = executor->RunWithBucket(index % 2);
    REQUIRE(filter->CheckValid(index) == false);
    REQUIRE(executor->only_bitset_ == true);
    executor->Clear();
    auto filter_other_bucket = executor->RunWithBucket((index + 1) % 2);
    REQUIRE(filter_other_bucket->CheckValid(index) == false);
    REQUIRE(executor->only_bitset_ == true);

    query = CreateMultiNotInString(name, none_interact_vec);
    expr = AstParse(query);
    executor = std::make_shared<IntegerListExecutor>(allocator, expr, sparse_attr_index);
    filter = executor->RunWithBucket(index % 2);
    REQUIRE(filter->CheckValid(index) == true);

    REQUIRE(executor->only_bitset_ == true);
    executor->Clear();
    filter_other_bucket = executor->RunWithBucket((index + 1) % 2);
    REQUIRE(filter_other_bucket->CheckValid(index) == true);
    REQUIRE(executor->only_bitset_ == true);

    auto interact_vec = GetInteractValues(values, name);
    query = CreateMultiNotInString(name, interact_vec);
    expr = AstParse(query);
    executor = std::make_shared<IntegerListExecutor>(allocator, expr, sparse_attr_index);
    filter = executor->RunWithBucket(index % 2);
    REQUIRE(filter->CheckValid(index) == false);
    REQUIRE(executor->only_bitset_ == true);
    executor->Clear();
    filter_other_bucket = executor->RunWithBucket((index + 1) % 2);
    REQUIRE(filter_other_bucket->CheckValid(index) == true);
    REQUIRE(executor->only_bitset_ == true);

    query = CreateMultiInString(name, interact_vec);
    expr = AstParse(query);
    executor = std::make_shared<IntegerListExecutor>(allocator, expr, sparse_attr_index);
    filter = executor->RunWithBucket(index % 2);
    REQUIRE(filter->CheckValid(index) == true);
    REQUIRE(executor->only_bitset_ == true);
    executor->Clear();
    filter_other_bucket = executor->RunWithBucket((index + 1) % 2);
    REQUIRE(filter_other_bucket->CheckValid(index) == false);
    REQUIRE(executor->only_bitset_ == true);
}

TEST_CASE("IntegerListExecutor Normal With Bucket", "[ut][IntegerListExecutor]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto fast_attr_index = AttributeInvertedInterface::MakeInstance(allocator.get(), true);

    std::vector<AttributeSet> attr_sets;
    for (int i = 0; i < 20; ++i) {
        attr_sets.emplace_back(ExecutorTest::MockAttrSet());
    }
    int idx = 0;
    for (auto& attr_set : attr_sets) {
        fast_attr_index->InsertWithBucket(attr_set, idx, idx % 2);
        idx++;
    }

    for (int i = 0; i < 20; ++i) {
        auto& attr_set = attr_sets[i];
        for (auto& attr_ptr : attr_set.attrs_) {
            auto name = attr_ptr->name_;
            auto type_str = split_string(name, '_')[0];
            if (type_str == "i32") {
                auto vec = GetValues<int32_t>(attr_ptr);
                TestAttributeWithBucket<int32_t>(name, vec, i, allocator.get(), fast_attr_index);
            } else if (type_str == "i64") {
                auto vec = GetValues<int64_t>(attr_ptr);
                TestAttributeWithBucket<int64_t>(name, vec, i, allocator.get(), fast_attr_index);
            } else if (type_str == "i8") {
                auto vec = GetValues<int8_t>(attr_ptr);
                TestAttributeWithBucket<int8_t>(name, vec, i, allocator.get(), fast_attr_index);
            } else if (type_str == "i16") {
                auto vec = GetValues<int16_t>(attr_ptr);
                TestAttributeWithBucket<int16_t>(name, vec, i, allocator.get(), fast_attr_index);
            } else if (type_str == "u8") {
                auto vec = GetValues<uint8_t>(attr_ptr);
                TestAttributeWithBucket<uint8_t>(name, vec, i, allocator.get(), fast_attr_index);
            } else if (type_str == "u16") {
                auto vec = GetValues<uint16_t>(attr_ptr);
                TestAttributeWithBucket<uint16_t>(name, vec, i, allocator.get(), fast_attr_index);
            } else if (type_str == "u32") {
                auto vec = GetValues<uint32_t>(attr_ptr);
                TestAttributeWithBucket<uint32_t>(name, vec, i, allocator.get(), fast_attr_index);
            } else if (type_str == "u64") {
                auto vec = GetValues<uint64_t>(attr_ptr);
                TestAttributeWithBucket<uint64_t>(name, vec, i, allocator.get(), fast_attr_index);
            }
        }
    }

    for (auto& attr_set : attr_sets) {
        ExecutorTest::DeleteAttrSet(attr_set);
    }
}
