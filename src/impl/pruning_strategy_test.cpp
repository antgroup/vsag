
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

#include "pruning_strategy.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <set>
#include <vector>

#include "data_cell/flatten_datacell_parameter.h"
#include "data_cell/graph_datacell_parameter.h"
#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/heap/standard_heap.h"
#include "io/memory_io_parameter.h"
#include "lock_strategy.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "typing.h"
#include "vsag/engine.h"

namespace vsag {
using DistanceRecord = std::pair<float, InnerIdType>;
#define distance first
#define id second

void
select_edges_by_heuristic(const DistHeapPtr& edges,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator,
                          float alpha = 1.0F) {
    if (edges->Size() <= max_size) {
        return;
    }

    // 1. Extract and scale all edges from the heap
    std::vector<DistanceRecord> all_edges;
    while (!edges->Empty()) {
        auto top = edges->Top();
        all_edges.emplace_back(top.distance * alpha, top.id);
        edges->Pop();
    }

    // 2. Sort by distance (descending) and then by id (ascending) for tie-breaker
    std::sort(
        all_edges.begin(), all_edges.end(), [](const DistanceRecord& a, const DistanceRecord& b) {
            if (a.distance != b.distance) {
                return a.distance > b.distance;
            }
            return a.id < b.id;
        });

    // 3. Select top max_size edges
    for (size_t i = 0; i < max_size && i < all_edges.size(); ++i) {
        edges->Push(all_edges[i].distance, all_edges[i].id);
    }
}

static FlattenInterfacePtr
CreateTestFlatten(Allocator* allocator) {
    auto flatten_param = std::make_shared<FlattenDataCellParameter>();
    flatten_param->quantizer_parameter = std::make_shared<FP32QuantizerParameter>();
    flatten_param->io_parameter = std::make_shared<MemoryIOParameter>();
    flatten_param->name = FLATTEN_DATA_CELL;

    IndexCommonParam common_param;
    common_param.allocator_ = std::shared_ptr<Allocator>(allocator, [](auto*) {});
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.dim_ = 128;

    auto flatten = FlattenInterface::MakeInstance(flatten_param, common_param);
    float vectors[4][3] = {
        {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, {0.5F, 0.5F, 0.5F}};
    // Node 0: [1,0,0] (query point)
    // Node 1: [0,1,0] (distance 1.414 from node 0)
    // Node 2: [0,0,1] (distance 1.414 from node 0)
    // Node 3: [0.5,0.5,0.5] (distance 0.866 from node 0)

    flatten->Train(vectors, 4);
    flatten->BatchInsertVector(vectors, 4);

    return flatten;
}

static GraphInterfacePtr
CreateTestGraph(Allocator* allocator) {
    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 32;
    graph_param->init_max_capacity_ = 100;
    graph_param->support_remove_ = false;

    IndexCommonParam common_param;
    common_param.allocator_ = std::shared_ptr<Allocator>(allocator, [](auto*) {});

    return GraphInterface::MakeInstance(graph_param, common_param);
}

TEST_CASE("Pruning Strategy Select Edges Test", "[ut][pruning_strategy]") {
    auto allocator = Engine::CreateDefaultAllocator();
    {
        auto flatten = CreateTestFlatten(allocator.get());

        SECTION("prunes to farthest nodes (alpha=1.0 default)") {
            // Initialize test heap with 3 edges:
            // [ (1.414,1), (1.414,2), (0.866,3) ]
            auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
            edges->Push(flatten->ComputePairVectors(0, 1), 1);
            edges->Push(flatten->ComputePairVectors(0, 2), 2);
            edges->Push(flatten->ComputePairVectors(0, 3), 3);

            // Pruning process (max_size=2):
            // 1. Extract all edges (no change with alpha=1.0)
            // 2. Sort: maintains order since distances equal
            // 3. Select top 2: keeps nodes 1 and 2
            select_edges_by_heuristic(edges, 2, flatten, allocator.get());

            REQUIRE(edges->Size() == 2);
            std::set<InnerIdType> kept_nodes;
            kept_nodes.insert(edges->Top().id);
            edges->Pop();
            kept_nodes.insert(edges->Top().id);
            REQUIRE(kept_nodes == std::set<InnerIdType>{1, 2});
        }

        SECTION("alpha=0.5 reduces distance differences") {
            auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
            edges->Push(flatten->ComputePairVectors(0, 1), 1);
            edges->Push(flatten->ComputePairVectors(0, 2), 2);
            edges->Push(flatten->ComputePairVectors(0, 3), 3);

            // Pruning process:
            // 1. Scaled distances: [0.707, 0.707, 0.433]
            // 2. Sort maintains order
            // 3. Still keeps nodes 1 and 2
            select_edges_by_heuristic(edges, 2, flatten, allocator.get(), 0.5F);

            REQUIRE(edges->Size() == 2);
            std::set<InnerIdType> kept_nodes;
            kept_nodes.insert(edges->Top().id);
            edges->Pop();
            kept_nodes.insert(edges->Top().id);
            REQUIRE(kept_nodes == std::set<InnerIdType>{1, 2});
        }

        SECTION("alpha=2.0 amplifies distance differences") {
            auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
            edges->Push(flatten->ComputePairVectors(0, 1), 1);
            edges->Push(flatten->ComputePairVectors(0, 2), 2);
            edges->Push(flatten->ComputePairVectors(0, 3), 3);

            select_edges_by_heuristic(edges, 2, flatten, allocator.get(), 2.0F);

            REQUIRE(edges->Size() == 2);
            std::set<InnerIdType> kept_nodes;
            kept_nodes.insert(edges->Top().id);
            edges->Pop();
            kept_nodes.insert(edges->Top().id);
            REQUIRE(kept_nodes == std::set<InnerIdType>{1, 2});
        }

        SECTION("alpha=0.5 with single selection") {
            auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
            edges->Push(flatten->ComputePairVectors(0, 1), 1);  // 0.707
            edges->Push(flatten->ComputePairVectors(0, 2), 2);  // 0.707
            edges->Push(flatten->ComputePairVectors(0, 3), 3);  // 0.433

            select_edges_by_heuristic(edges, 1, flatten, allocator.get(), 0.5F);

            REQUIRE(edges->Size() == 1);
            REQUIRE((edges->Top().id == 1 || edges->Top().id == 2));
        }
    }
}

TEST_CASE("Pruning Strategy Mutual Connection Test", "[ut][pruning_strategy]") {
    auto allocator = Engine::CreateDefaultAllocator();
    {
        auto flatten = CreateTestFlatten(allocator.get());
        auto graph = CreateTestGraph(allocator.get());
        auto mutexes = std::make_shared<EmptyMutex>();

        graph->SetMaximumDegree(1);

        SECTION("connects to farthest candidate") {
            auto candidates = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
            candidates->Push(flatten->ComputePairVectors(0, 1), 1);
            candidates->Push(flatten->ComputePairVectors(0, 3), 3);

            //verify connection was made
            auto entry = mutually_connect_new_element(
                0, candidates, graph, flatten, mutexes, allocator.get());
            REQUIRE(entry == 1);

            Vector<InnerIdType> neighbors(allocator.get());
            graph->GetNeighbors(0, neighbors);
            REQUIRE(neighbors.size() == 1);
            REQUIRE(neighbors[0] == 1);
        }

        SECTION("rejects invalid candidates") {
            auto candidates = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
            candidates->Push(-1.0f, 0);

            REQUIRE_THROWS_AS(mutually_connect_new_element(
                                  0, candidates, graph, flatten, mutexes, allocator.get()),
                              VsagException);
        }
    }
}
}  // namespace vsag
