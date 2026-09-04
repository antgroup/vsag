// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pipnn_graph_builder.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include "datacell/graph_interface.h"
#include "datacell/graph_interface_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index_common_param.h"
#include "unittest.h"

namespace {

vsag::IndexCommonParam
MakeCommonParam(uint64_t dimensions) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = static_cast<int64_t>(dimensions);
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    return common_param;
}

std::vector<float>
MakeVectors(uint64_t count, uint64_t dimensions) {
    std::vector<float> vectors(count * dimensions);
    for (uint64_t point = 0; point < count; ++point) {
        for (uint64_t dim = 0; dim < dimensions; ++dim) {
            vectors[point * dimensions + dim] = std::sin(static_cast<float>(point * 17 + dim * 3)) +
                                                static_cast<float>(point) * 0.01F;
        }
    }
    return vectors;
}

vsag::Vector<const float*>
MakeRows(const std::vector<float>& vectors,
         const vsag::Vector<vsag::InnerIdType>& ids,
         uint64_t dimensions,
         vsag::Allocator* allocator) {
    vsag::Vector<const float*> rows(allocator);
    rows.reserve(ids.size());
    for (const auto id : ids) {
        rows.emplace_back(vectors.data() + static_cast<uint64_t>(id) * dimensions);
    }
    return rows;
}

vsag::GraphInterfacePtr
MakeGraph(const vsag::IndexCommonParam& common_param,
          uint64_t count,
          uint64_t max_degree,
          bool support_duplicate = false) {
    const auto graph_json = vsag::JsonType::Parse(fmt::format(
        R"({{"io_params": {{"type": "block_memory_io"}}, "max_degree": {}, "init_capacity": {}, "support_duplicate": {}}})",
        max_degree,
        std::max<uint64_t>(count, 1),
        support_duplicate));
    const auto graph_param = vsag::GraphInterfaceParameter::GetGraphParameterByJson(
        vsag::GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, graph_json);
    auto graph = vsag::GraphInterface::MakeInstance(graph_param, common_param);
    graph->Resize(static_cast<vsag::InnerIdType>(std::max<uint64_t>(count, 1)));
    return graph;
}

void
RequireGraphInvariants(const vsag::GraphInterfacePtr& graph,
                       const vsag::Vector<vsag::InnerIdType>& ids,
                       uint64_t max_degree,
                       vsag::Allocator* allocator) {
    std::set<vsag::InnerIdType> valid(ids.begin(), ids.end());
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> neighbors(allocator);
        graph->GetNeighbors(id, neighbors);
        REQUIRE(neighbors.size() <= max_degree);
        REQUIRE(std::set<vsag::InnerIdType>(neighbors.begin(), neighbors.end()).size() ==
                neighbors.size());
        for (const auto neighbor : neighbors) {
            REQUIRE(neighbor != id);
            REQUIRE(valid.find(neighbor) != valid.end());
        }
        if (ids.size() > 1) {
            REQUIRE_FALSE(neighbors.empty());
        }
    }
}

}  // namespace

TEST_CASE("PiPNN graph builder preserves adjacency invariants and determinism", "[ut][pipnn]") {
    constexpr uint64_t dimensions = 8;
    constexpr uint64_t count = 40;
    constexpr uint64_t max_degree = 8;
    auto common_param = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(count, dimensions);

    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    for (uint64_t id = 0; id < count; id += 2) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());

    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = 6;
    parameter.min_leaf_size = 2;
    parameter.leader_sample_rate = 0.25F;
    parameter.fanout = {2, 1};
    parameter.reservoir_size = 16;

    auto first = MakeGraph(common_param, count, max_degree);
    auto second = MakeGraph(common_param, count, max_degree);
    vsag::PiPNNGraphBuilder builder(parameter, dimensions, common_param.allocator_.get());
    builder.Build(first, ids, rows);
    builder.Build(second, ids, rows);

    RequireGraphInvariants(first, ids, max_degree, common_param.allocator_.get());
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> first_neighbors(common_param.allocator_.get());
        vsag::Vector<vsag::InnerIdType> second_neighbors(common_param.allocator_.get());
        first->GetNeighbors(id, first_neighbors);
        second->GetNeighbors(id, second_neighbors);
        REQUIRE(first_neighbors == second_neighbors);
    }
}

TEST_CASE("PiPNN graph builder writes an empty row for one point", "[ut][pipnn]") {
    constexpr uint64_t dimensions = 4;
    auto common_param = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(1, dimensions);
    auto graph = MakeGraph(common_param, 1, 4);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    ids.emplace_back(0);
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());

    vsag::PiPNNGraphBuilder({}, dimensions, common_param.allocator_.get()).Build(graph, ids, rows);

    vsag::Vector<vsag::InnerIdType> neighbors(common_param.allocator_.get());
    graph->GetNeighbors(0, neighbors);
    REQUIRE(neighbors.empty());
}

TEST_CASE("PiPNN graph builder runs overlapping leaves in parallel", "[ut][pipnn]") {
    constexpr uint64_t dimensions = 8;
    constexpr uint64_t count = 96;
    constexpr uint64_t max_degree = 8;
    auto common_param = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(count, dimensions);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    for (uint64_t id = 0; id < count; ++id) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());
    auto serial = MakeGraph(common_param, count, max_degree);
    auto first = MakeGraph(common_param, count, max_degree);
    auto second = MakeGraph(common_param, count, max_degree);

    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = 12;
    parameter.min_leaf_size = 3;
    parameter.leader_sample_rate = 0.25F;
    parameter.fanout = {3, 2};
    parameter.reservoir_size = 16;
    vsag::PiPNNGraphBuilder(parameter, dimensions, common_param.allocator_.get())
        .Build(serial, ids, rows);
    auto thread_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    thread_pool->SetPoolSize(4);
    vsag::PiPNNGraphBuilder builder(
        parameter, dimensions, common_param.allocator_.get(), thread_pool.get(), 4);
    builder.Build(first, ids, rows);
    builder.Build(second, ids, rows);

    RequireGraphInvariants(first, ids, max_degree, common_param.allocator_.get());
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> serial_neighbors(common_param.allocator_.get());
        vsag::Vector<vsag::InnerIdType> first_neighbors(common_param.allocator_.get());
        vsag::Vector<vsag::InnerIdType> second_neighbors(common_param.allocator_.get());
        serial->GetNeighbors(id, serial_neighbors);
        first->GetNeighbors(id, first_neighbors);
        second->GetNeighbors(id, second_neighbors);
        REQUIRE(serial_neighbors == first_neighbors);
        REQUIRE(first_neighbors == second_neighbors);
    }
}

TEST_CASE("PiPNN keeps identical vectors reachable across fallback leaves", "[ut][pipnn]") {
    constexpr uint64_t dimensions = 4;
    constexpr uint64_t max_leaf_size = 8;
    constexpr uint64_t count = max_leaf_size + 1;
    constexpr uint64_t max_degree = 4;
    auto common_param = MakeCommonParam(dimensions);
    std::vector<float> vectors(count * dimensions, 1.0F);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    for (uint64_t id = 0; id < count; ++id) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());
    auto graph = MakeGraph(common_param, count, max_degree);

    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = max_leaf_size;
    parameter.min_leaf_size = 2;
    parameter.leader_sample_rate = 0.25F;
    parameter.fanout = {2, 1};
    parameter.hash_plane_count = 3;
    parameter.reservoir_size = 8;
    vsag::PiPNNGraphBuilder(parameter, dimensions, common_param.allocator_.get())
        .Build(graph, ids, rows);

    RequireGraphInvariants(graph, ids, max_degree, common_param.allocator_.get());
    std::set<vsag::InnerIdType> visited{ids.front()};
    std::vector<vsag::InnerIdType> pending{ids.front()};
    for (uint64_t cursor = 0; cursor < pending.size(); ++cursor) {
        vsag::Vector<vsag::InnerIdType> neighbors(common_param.allocator_.get());
        graph->GetNeighbors(pending[cursor], neighbors);
        for (const auto neighbor : neighbors) {
            if (visited.emplace(neighbor).second) {
                pending.emplace_back(neighbor);
            }
        }
    }
    REQUIRE(visited.size() == count);
}

TEST_CASE("PiPNN registers exact duplicate rows instead of building duplicate vertices",
          "[ut][pipnn][duplicate]") {
    constexpr uint64_t dimensions = 4;
    constexpr uint64_t count = 6;
    auto common_param = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(count, dimensions);
    std::copy(vectors.begin() + dimensions,
              vectors.begin() + 2 * dimensions,
              vectors.begin() + 4 * dimensions);
    std::copy(vectors.begin() + dimensions,
              vectors.begin() + 2 * dimensions,
              vectors.begin() + 5 * dimensions);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    for (uint64_t id = 0; id < count; ++id) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());
    auto graph = MakeGraph(common_param, count, 4, true);

    vsag::PiPNNGraphBuilder({}, dimensions, common_param.allocator_.get()).Build(graph, ids, rows);

    REQUIRE(graph->GetGroupId(4) == 1);
    REQUIRE(graph->GetGroupId(5) == 1);
    auto duplicates = graph->GetDuplicateIds(1);
    std::sort(duplicates.begin(), duplicates.end());
    REQUIRE(duplicates == std::vector<vsag::InnerIdType>{4, 5});
    REQUIRE(graph->TotalCount() == 4);
}

TEST_CASE("PiPNN graph builder validates structural parameters", "[ut][pipnn]") {
    vsag::PiPNNGraphBuilderParameter parameter;
    REQUIRE_NOTHROW(parameter.Validate(32));

    SECTION("leaf size below two") {
        parameter.max_leaf_size = 1;
        parameter.min_leaf_size = 1;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("zero min leaf size") {
        parameter.min_leaf_size = 0;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("min leaf size exceeds max") {
        parameter.min_leaf_size = parameter.max_leaf_size + 1;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("empty fanout") {
        parameter.fanout.clear();
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("zero leaf neighbor count") {
        parameter.leaf_neighbor_count = 0;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("hash count uses the coincident marker bit") {
        parameter.hash_plane_count = 16;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("zero graph degree") {
        REQUIRE_THROWS(parameter.Validate(0));
    }
}
