
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

#include "logical_executor.h"

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "attr/expression_visitor.h"
#include "executor_test.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

std::string
CreateAndString(const std::string& left, const std::string& right) {
    return left + " AND " + right;
}

std::string
CreateOrString(const std::string& left, const std::string& right) {
    return left + " OR " + right;
}

template <typename T1, typename T2>
static void
TestLogical(const std::string& name1,
            const std::vector<T1>& values1,
            const std::string& name2,
            const std::vector<T2>& values2,
            int index,
            Allocator* allocator,
            AttrInvertedInterfacePtr attr_index) {
    auto none_interact_vec1 = GetNoneInteractValues(values1, name1);
    auto none_interact_vec2 = GetNoneInteractValues(values2, name2);
    auto interact_vec1 = GetInteractValues(values1, name1);
    auto interact_vec2 = GetInteractValues(values2, name2);

    auto query_non_in_1 = CreateMultiInString(name1, none_interact_vec1);
    auto query_non_in_2 = CreateMultiInString(name2, none_interact_vec2);
    auto query_in_1 = CreateMultiInString(name1, interact_vec1);
    auto query_in_2 = CreateMultiInString(name2, interact_vec2);
    auto query_non_notin_1 = CreateMultiNotInString(name1, none_interact_vec1);
    auto query_non_notin_2 = CreateMultiNotInString(name2, none_interact_vec2);
    auto query_notin_1 = CreateMultiNotInString(name1, interact_vec1);
    auto query_notin_2 = CreateMultiNotInString(name2, interact_vec2);

    auto query = CreateAndString(query_non_in_1, query_in_2);
    auto expr = AstParse(query);
    ExecutorPtr executor = std::make_shared<LogicalExecutor>(allocator, expr, attr_index);
    executor->Init();
    auto filter = executor->Run(5);
    REQUIRE(filter->CheckValid(index) == false);
    REQUIRE(executor->only_bitset_ == true);

    query = CreateOrString(query_non_in_1, query_in_2);
    expr = AstParse(query);
    executor = std::make_shared<LogicalExecutor>(allocator, expr, attr_index);
    executor->Init();
    filter = executor->Run(5);
    REQUIRE(filter->CheckValid(index) == true);

    query = CreateAndString(query_in_1, query_in_2);
    expr = AstParse(query);
    executor = std::make_shared<LogicalExecutor>(allocator, expr, attr_index);
    executor->Init();
    filter = executor->Run(5);
    REQUIRE(filter->CheckValid(index) == true);

    query = CreateOrString(query_non_notin_1, query_non_in_2);
    expr = AstParse(query);
    executor = std::make_shared<LogicalExecutor>(allocator, expr, attr_index);
    executor->Init();
    filter = executor->Run(5);
    REQUIRE(filter->CheckValid(index) == true);

    query = CreateAndString(query_non_notin_1, query_non_in_2);
    expr = AstParse(query);
    executor = std::make_shared<LogicalExecutor>(allocator, expr, attr_index);
    executor->Init();
    filter = executor->Run(5);
    REQUIRE(filter->CheckValid(index) == false);

    query = CreateAndString(query_non_notin_1, query_non_notin_2);
    expr = AstParse(query);
    executor = std::make_shared<LogicalExecutor>(allocator, expr, attr_index);
    executor->Init();
    filter = executor->Run(5);
    REQUIRE(filter->CheckValid(index) == true);
}

template <typename T1>
static void
TestLogical(const std::string& name1,
            const std::vector<T1>& values1,
            const std::string& name2,
            Attribute* attr2,
            int index,
            Allocator* allocator,
            AttrInvertedInterfacePtr attr_index) {
    auto type_str = split_string(name2, '_')[0];
    if (type_str == "str") {
        auto vec = GetValues<std::string>(attr2);
        TestLogical<T1, std::string>(name1, values1, name2, vec, index, allocator, attr_index);
    } else if (type_str == "i32") {
        auto vec = GetValues<int32_t>(attr2);
        TestLogical<T1, int32_t>(name1, values1, name2, vec, index, allocator, attr_index);
    } else if (type_str == "i64") {
        auto vec = GetValues<int64_t>(attr2);
        TestLogical<T1, int64_t>(name1, values1, name2, vec, index, allocator, attr_index);
    } else if (type_str == "i8") {
        auto vec = GetValues<int8_t>(attr2);
        TestLogical<T1, int8_t>(name1, values1, name2, vec, index, allocator, attr_index);
    } else if (type_str == "i16") {
        auto vec = GetValues<int16_t>(attr2);
        TestLogical<T1, int16_t>(name1, values1, name2, vec, index, allocator, attr_index);
    } else if (type_str == "u8") {
        auto vec = GetValues<uint8_t>(attr2);
        TestLogical<T1, uint8_t>(name1, values1, name2, vec, index, allocator, attr_index);
    } else if (type_str == "u16") {
        auto vec = GetValues<uint16_t>(attr2);
        TestLogical<T1, uint16_t>(name1, values1, name2, vec, index, allocator, attr_index);
    } else if (type_str == "u32") {
        auto vec = GetValues<uint32_t>(attr2);
        TestLogical<T1, uint32_t>(name1, values1, name2, vec, index, allocator, attr_index);
    } else if (type_str == "u64") {
        auto vec = GetValues<uint64_t>(attr2);
        TestLogical<T1, uint64_t>(name1, values1, name2, vec, index, allocator, attr_index);
    }
}

static void
TestLogical(const std::string& name1,
            Attribute* attr1,
            const std::string& name2,
            Attribute* attr2,
            int index,
            Allocator* allocator,
            AttrInvertedInterfacePtr attr_index) {
    auto type_str = split_string(name1, '_')[0];
    if (type_str == "str") {
        auto vec = GetValues<std::string>(attr1);
        TestLogical<std::string>(name1, vec, name2, attr2, index, allocator, attr_index);
    } else if (type_str == "i32") {
        auto vec = GetValues<int32_t>(attr1);
        TestLogical<int32_t>(name1, vec, name2, attr2, index, allocator, attr_index);
    } else if (type_str == "i64") {
        auto vec = GetValues<int64_t>(attr1);
        TestLogical<int64_t>(name1, vec, name2, attr2, index, allocator, attr_index);
    } else if (type_str == "i8") {
        auto vec = GetValues<int8_t>(attr1);
        TestLogical<int8_t>(name1, vec, name2, attr2, index, allocator, attr_index);
    } else if (type_str == "i16") {
        auto vec = GetValues<int16_t>(attr1);
        TestLogical<int16_t>(name1, vec, name2, attr2, index, allocator, attr_index);
    } else if (type_str == "u8") {
        auto vec = GetValues<uint8_t>(attr1);
        TestLogical<uint8_t>(name1, vec, name2, attr2, index, allocator, attr_index);
    } else if (type_str == "u16") {
        auto vec = GetValues<uint16_t>(attr1);
        TestLogical<uint16_t>(name1, vec, name2, attr2, index, allocator, attr_index);
    } else if (type_str == "u32") {
        auto vec = GetValues<uint32_t>(attr1);
        TestLogical<uint32_t>(name1, vec, name2, attr2, index, allocator, attr_index);
    } else if (type_str == "u64") {
        auto vec = GetValues<uint64_t>(attr1);
        TestLogical<uint64_t>(name1, vec, name2, attr2, index, allocator, attr_index);
    }
}

TEST_CASE("LogicalExecutor Normal Test", "[ut][LogicalExecutor]") {
    bool has_bucket = GENERATE(true, false);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto attr_index = AttributeInvertedInterface::MakeInstance(allocator.get(), has_bucket);

    std::vector<AttributeSet> attr_sets;
    for (int i = 0; i < 20; ++i) {
        attr_sets.emplace_back(ExecutorTest::MockAttrSet());
    }
    int idx = 0;
    for (auto& attr_set : attr_sets) {
        attr_index->Insert(attr_set, idx, 5);
        idx++;
    }

    for (int i = 0; i < 20; ++i) {
        auto& attr_set = attr_sets[i];
        auto attr_size = attr_set.attrs_.size();
        for (int j = 0; j < 5; ++j) {
            auto selected_idx = select_k_numbers(attr_size, 2);
            auto* attr1 = attr_set.attrs_[selected_idx[0]];
            auto* attr2 = attr_set.attrs_[selected_idx[1]];
            TestLogical(attr1->name_, attr1, attr2->name_, attr2, i, allocator.get(), attr_index);
        }
    }
    for (auto& attr_set : attr_sets) {
        ExecutorTest::DeleteAttrSet(attr_set);
    }
}
