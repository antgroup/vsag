
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

#include "graph_datacell_parameter.h"
#include "graph_interface_parameter.h"
#include "graph_interface_test.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io_parameter.h"
#include "sparse_graph_datacell_parameter.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("Graph deserialization restores incoming edges before tail moves",
          "[ut][GraphDataCell][mci][reload_remove]") {
    const bool sparse = GENERATE(false, true);
    const bool versioned = GENERATE(false, true);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 4;
    GraphInterfaceParamPtr param;
    if (sparse) {
        auto sparse_param = std::make_shared<SparseGraphDatacellParameter>();
        sparse_param->support_delete_ = versioned;
        param = sparse_param;
    } else {
        auto flat_param = std::make_shared<GraphDataCellParameter>();
        flat_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
        flat_param->support_remove_ = versioned;
        param = flat_param;
    }
    param->use_reverse_edges_ = true;
    param->max_degree_ = 4;
    auto graph = GraphInterface::MakeInstance(param, common);
    auto restored = GraphInterface::MakeInstance(param, common);
    const Vector<InnerIdType> ids(sparse ? std::initializer_list<InnerIdType>{10, 25, 100}
                                         : std::initializer_list<InnerIdType>{0, 1, 2},
                                  allocator.get());
    graph->Resize(ids.back() + 1);
    Vector<InnerIdType> empty(allocator.get());
    for (auto id : ids) {
        graph->InsertNeighborsById(id, empty);
    }
    Vector<InnerIdType> first({ids[1], ids[2]}, allocator.get());
    Vector<InnerIdType> second({ids[2]}, allocator.get());
    graph->InsertNeighborsById(ids[0], first);
    graph->InsertNeighborsById(ids[1], second);
    if (versioned) {
        graph->DeleteNeighborsById(ids[1]);
    }
    test_serializion(*graph, *restored);
    Vector<InnerIdType> incoming(allocator.get());
    restored->GetIncomingNeighbors(ids[1], incoming);
    REQUIRE(incoming.size() == (versioned ? 0 : 1));
    restored->GetIncomingNeighbors(ids[2], incoming);
    std::sort(incoming.begin(), incoming.end());
    REQUIRE(incoming == Vector<InnerIdType>({ids[0], ids[1]}, allocator.get()));
    restored->InsertNeighborsById(ids[0], empty);
    restored->Move(ids[2], ids[0]);
    Vector<InnerIdType> neighbors(allocator.get());
    restored->GetNeighbors(ids[1], neighbors);
    REQUIRE(neighbors == Vector<InnerIdType>({ids[0]}, allocator.get()));
    restored->GetIncomingNeighbors(ids[2], incoming);
    REQUIRE(incoming.empty());
}

void
test_graph_data_cell(const GraphInterfaceParamPtr& param,
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
    const auto* io_type = GENERATE("memory_io", "block_memory_io");
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
    auto param_json = JsonType::Parse(param_str);
    auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, param_json);
    test_graph_data_cell(graph_param, common_param, is_support_delete);
}

TEST_CASE("GraphDataCell Remove Test", "[ut][GraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 32);
    const auto* io_type = GENERATE("block_memory_io");
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
    auto param_json = JsonType::Parse(param_str);
    auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, param_json);
    test_graph_data_cell(graph_param, common_param, is_support_delete);
}

TEST_CASE("GraphDataCell Merge", "[ut][GraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32);
    auto max_degree = GENERATE(5, 32, 64);
    auto max_capacity = GENERATE(100);
    const auto* io_type = GENERATE("memory_io", "block_memory_io");
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
    auto param_json = JsonType::Parse(param_str);
    auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, param_json);

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    auto other = GraphInterface::MakeInstance(graph_param, common_param);
    test.MergeTest(other, 1000);
}

TEST_CASE("GraphDataCell duplicate tracker follows parameter", "[ut][GraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = allocator;

    auto disabled_param = std::make_shared<GraphDataCellParameter>();
    disabled_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    disabled_param->support_duplicate_ = false;
    auto disabled_graph = GraphInterface::MakeInstance(disabled_param, common_param);
    REQUIRE(disabled_graph->GetDuplicateTracker() == nullptr);
    REQUIRE(disabled_graph->GetDuplicateIds(0).empty());

    auto enabled_param = std::make_shared<GraphDataCellParameter>();
    enabled_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    enabled_param->support_duplicate_ = true;
    auto enabled_graph = GraphInterface::MakeInstance(enabled_param, common_param);
    REQUIRE(enabled_graph->GetDuplicateTracker() != nullptr);

    enabled_graph->Resize(4);
    enabled_graph->SetDuplicateId(0, 1);
    REQUIRE(disabled_graph->GetGroupId(5) == 5);
    REQUIRE(enabled_graph->GetGroupId(0) == 0);
    REQUIRE(enabled_graph->GetGroupId(1) == 0);
    REQUIRE(enabled_graph->GetDuplicateIds(0) == std::vector<InnerIdType>{1});
    REQUIRE(enabled_graph->GetDuplicateIds(1) == std::vector<InnerIdType>{0});
}

TEST_CASE("GraphDataCell Reverse Edges", "[ut][GraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = 32;
    auto max_degree = 16;
    auto io_type = "block_memory_io";
    constexpr const char* graph_param_temp =
        R"(
        {{
            "io_params": {{
                "type": "{}"
            }},
            "max_degree": {},
            "use_reverse_edges": true
        }}
        )";

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto param_str = fmt::format(graph_param_temp, io_type, max_degree);
    auto param_json = JsonType::Parse(param_str);
    auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, param_json);

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    test.ReverseEdgeTest(100);
}

TEST_CASE("GraphDataCell Move", "[ut][GraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = 8;
    auto max_degree = 8;
    auto io_type = "block_memory_io";
    constexpr const char* graph_param_temp =
        R"(
        {{
            "io_params": {{
                "type": "{}"
            }},
            "max_degree": {},
            "use_reverse_edges": true
        }}
        )";

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto param_str = fmt::format(graph_param_temp, io_type, max_degree);
    auto param_json = JsonType::Parse(param_str);
    auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, param_json);
    auto origin_size = vsag::Options::Instance().block_size_limit();
    const uint64_t size = uint64_t{2} * 1024 * 1024;
    vsag::Options::Instance().set_block_size_limit(size);
    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    test.MoveTest(100);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE("GraphDataCell Move same id keeps neighbors", "[ut][GraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int dim = 32;
    constexpr int max_degree = 8;
    constexpr const char* graph_param_temp =
        R"(
        {{
            "io_params": {{
                "type": "block_memory_io"
            }},
            "max_degree": {},
            "use_reverse_edges": true
        }}
        )";

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto param_str = fmt::format(graph_param_temp, max_degree);
    auto param_json = JsonType::Parse(param_str);
    auto graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, param_json);

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    graph->Resize(4);

    Vector<InnerIdType> empty(allocator.get());
    Vector<InnerIdType> neighbors(allocator.get());
    neighbors.emplace_back(1);
    neighbors.emplace_back(2);

    graph->InsertNeighborsById(0, neighbors);
    graph->InsertNeighborsById(1, empty);
    graph->InsertNeighborsById(2, empty);

    graph->Move(0, 0);

    Vector<InnerIdType> moved_neighbors(allocator.get());
    graph->GetNeighbors(0, moved_neighbors);
    REQUIRE(moved_neighbors.size() == 2);
    REQUIRE(moved_neighbors[0] == 1);
    REQUIRE(moved_neighbors[1] == 2);

    Vector<InnerIdType> incoming(allocator.get());
    graph->GetIncomingNeighbors(1, incoming);
    REQUIRE(incoming.size() == 1);
    REQUIRE(incoming[0] == 0);
}
