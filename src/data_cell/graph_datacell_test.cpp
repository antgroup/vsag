
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

#include <fmt/format.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "graph_interface_parameter.h"
#include "graph_interface_test.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

void
TestGraphDataCell(const GraphInterfaceParamPtr& param,
                  const IndexCommonParam& common_param,
                  bool test_delete) {
    auto count = GENERATE(1000, 2000);
    auto max_id = 10000;

    auto graph = GraphInterface::MakeInstance(param, common_param);
    GraphInterfaceTest test(graph);
    auto other = GraphInterface::MakeInstance(param, common_param);
    test.BasicTest(max_id, count, other, test_delete);
}

TEST_CASE("GraphDataCell Basic Test", "[ut][GraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 32, 64);
    auto max_capacity = GENERATE(100);
    auto io_type = GENERATE("memory_io", "block_memory_io");
    auto is_support_delete = GENERATE(true, false);
    constexpr const char* graph_param_temp =
        R"(
        {{
            "io_params": {{
                "type": "{}"
            }},
            "max_degree": {},
            "init_capacity": {},
            "support_remove": {}
        }}
        )";

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto param_str =
        fmt::format(graph_param_temp, io_type, max_degree, max_capacity, is_support_delete);
    auto param_json = JsonType::parse(param_str);
    auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_FLAT, param_json);
    TestGraphDataCell(graph_param, common_param, is_support_delete);
}

TEST_CASE("GraphDataCell Remove Test", "[ut][GraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 32);
    auto io_type = GENERATE("block_memory_io");
    auto is_support_delete = GENERATE(true);
    auto remove_flag_bit = GENERATE(4, 8);
    constexpr const char* graph_param_temp =
        R"(
        {{
            "io_params": {{
                "type": "{}"
            }},
            "max_degree": {},
            "support_remove": true,
            "remove_flag_bit": {}
        }}
        )";

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto param_str = fmt::format(graph_param_temp, io_type, max_degree, remove_flag_bit);
    auto param_json = JsonType::parse(param_str);
    auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_FLAT, param_json);
    TestGraphDataCell(graph_param, common_param, is_support_delete);
}

TEST_CASE("GraphDataCell Merge", "[ut][GraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32);
    auto max_degree = GENERATE(5, 32, 64);
    auto max_capacity = GENERATE(100);
    auto io_type = GENERATE("memory_io", "block_memory_io");
    auto is_support_delete = GENERATE(true, false);
    constexpr const char* graph_param_temp =
        R"(
    {{
        "io_params": {{
            "type": "{}"
        }},
        "max_degree": {},
        "init_capacity": {},
        "support_remove": {}
    }}
    )";

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto param_str =
        fmt::format(graph_param_temp, io_type, max_degree, max_capacity, is_support_delete);
    auto param_json = JsonType::parse(param_str);
    auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_FLAT, param_json);

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    auto other = GraphInterface::MakeInstance(graph_param, common_param);
    test.MergeTest(other, 1000);
}
