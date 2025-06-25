
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

#include "compressed_graph_datacell.h"

#include <fmt/format-inl.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "graph_datacell_parameter.h"
#include "graph_interface_test.h"
#include "safe_allocator.h"
using namespace vsag;

void
TestCompressedGraphDataCell(const GraphInterfaceParamPtr& param,
                            const IndexCommonParam& common_param) {
    auto count = GENERATE(1000, 2000);
    auto max_id = 10000;

    auto graph = GraphInterface::MakeInstance(param, common_param);
    GraphInterfaceTest test(graph, true);
    auto other = GraphInterface::MakeInstance(param, common_param);
    test.BasicTest(max_id, count, other, false);
}

TEST_CASE("CompressedGraphDataCell Basic Test", "[ut][CompressedGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 12, 32, 64, 128);

    // raw construction
    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;

    auto graph_param = std::make_shared<CompressedGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->graph_storage_type_ = GraphStorageTypes::GRAPH_STORAGE_TYPE_COMPRESSED;
    TestCompressedGraphDataCell(graph_param, common_param);

    // parameter construction
    constexpr const char* graph_param_temp =
        R"(
        {{
            "max_degree": {},
            "graph_storage_type": "{}"
        }}
        )";

    auto param_str = fmt::format(graph_param_temp, max_degree, GRAPH_STORAGE_TYPE_COMPRESSED);
    auto param_json = JsonType::parse(param_str);
    graph_param->FromJson(param_json);
    REQUIRE(graph_param->graph_storage_type_ == GraphStorageTypes::GRAPH_STORAGE_TYPE_COMPRESSED);
    TestCompressedGraphDataCell(graph_param, common_param);
}

void
TestBuildCompressedFromGraphDataCell(const GraphInterfaceParamPtr& param,
                                     const IndexCommonParam& common_param) {
    auto count = 2000;
    auto max_id = 10000;

    auto graph = GraphInterface::MakeInstance(param, common_param);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto max_degree = graph->MaximumDegree();
    graph->Resize(max_id);
    UnorderedMap<InnerIdType, std::shared_ptr<Vector<InnerIdType>>> maps(allocator.get());
    std::unordered_set<InnerIdType> unique_keys;
    while (unique_keys.size() < count) {
        InnerIdType new_key = random() % max_id;
        unique_keys.insert(new_key);
    }

    std::vector<InnerIdType> keys(unique_keys.begin(), unique_keys.end());
    for (auto key : keys) {
        maps[key] = std::make_shared<Vector<InnerIdType>>(allocator.get());
    }

    std::random_device rd;
    std::mt19937 rng(rd());

    for (auto& pair : maps) {
        auto& vec_ptr = pair.second;
        int max_possible_length = keys.size();
        int length = random() % (max_degree - 1) + 2;
        length = std::min(length, max_possible_length);
        std::vector<InnerIdType> temp_keys = keys;
        std::shuffle(temp_keys.begin(), temp_keys.end(), rng);

        vec_ptr->resize(length);
        for (int i = 0; i < length; ++i) {
            (*vec_ptr)[i] = temp_keys[i];
        }
    }

    for (auto& [key, value] : maps) {
        std::sort(value->begin(), value->end());
    }

    for (auto& [key, value] : maps) {
        graph->InsertNeighborsById(key, *value);
    }

    auto compressed_graph = CompressedGraphDataCell::MakeCompressedGraph(graph, allocator.get());
    // Test GetNeighbors
    SECTION("Test GetNeighbors") {
        for (InnerIdType id = 0; id < graph->MaxCapacity(); ++id) {
            Vector<InnerIdType> neighbors(allocator.get());
            graph->GetNeighbors(id, neighbors);
            Vector<InnerIdType> compressed_neighbors(allocator.get());
            compressed_graph->GetNeighbors(id, compressed_neighbors);
            REQUIRE(memcmp(neighbors.data(),
                           compressed_neighbors.data(),
                           neighbors.size() * sizeof(InnerIdType)) == 0);
        }
    }
}

// TEST_CASE("CompressedGraphDataCell Build From GraphDataCell Test",
//           "[ut][CompressedGraphDataCell]") {
//     // copy from graph_datacell_test.cpp
//     auto allocator = SafeAllocator::FactoryDefaultAllocator();
//     auto dim = GENERATE(32, 64);
//     auto max_degree = GENERATE(5, 32, 64);
//     auto max_capacity = GENERATE(100);
//     auto io_type = GENERATE("block_memory_io", "memory_io");
//     auto is_support_delete = false;
//     constexpr const char* graph_param_temp =
//         R"(
//         {{
//             "io_params": {{
//                 "type": "{}"
//             }},
//             "max_degree": {},
//             "init_capacity": {},
//             "support_remove": {}
//         }}
//         )";
//
//     IndexCommonParam common_param;
//     common_param.dim_ = dim;
//     common_param.allocator_ = allocator;
//     auto param_str =
//         fmt::format(graph_param_temp, io_type, max_degree, max_capacity, is_support_delete);
//     auto param_json = JsonType::parse(param_str);
//     auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
//         GraphStorageTypes::GRAPH_STORAGE_TYPE_FLAT, param_json);
//     TestBuildCompressedFromGraphDataCell(graph_param, common_param);
// }
