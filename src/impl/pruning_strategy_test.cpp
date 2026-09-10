
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
#include <cmath>
#include <memory>
#include <vector>

#include "datacell/flatten_datacell.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/graph_datacell_parameter.h"
#include "datacell/graph_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/heap/standard_heap.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "typing.h"
#include "unittest.h"
#include "utils/lock_strategy.h"
#include "vsag/engine.h"

namespace vsag {

TEST_CASE("Pruning Strategy Select Edges With Heuristic", "[ut][pruning_strategy]") {
    // Initialize memory allocator for safe memory management
    auto allocator = Engine::CreateDefaultAllocator();

    // Configure flatten data cell parameters with FP32 quantization and memory I/O
    auto flatten_param = std::make_shared<FlattenDataCellParameter>();
    flatten_param->quantizer_parameter = std::make_shared<FP32QuantizerParameter>();
    flatten_param->io_parameter = std::make_shared<MemoryIOParameter>();

    // Set common index parameters: allocator, L2 squared metric, and 128-dimensional vectors
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.dim_ = 128;

    // Create flatten interface instance with configured parameters
    auto flatten = FlattenInterface::MakeInstance(flatten_param, common_param);
    REQUIRE(flatten != nullptr);

    float vectors[5][128] = {0};
    vectors[0][0] = 5.0F;
    vectors[1][0] = 4.0F;
    vectors[2][1] = 3.0F;
    vectors[3][2] = 2.0F;
    vectors[4][3] = 1.0F;

    flatten->Train(vectors, 5);
    flatten->BatchInsertVector(vectors, 5);

    // Pre-calculated L2 squared distances from base vector (ID 0) to other vectors
    const float d01 = 1.0F;
    const float d02 = 34.0F;
    const float d03 = 29.0F;
    const float d04 = 26.0F;

    SECTION("Vector overload preserves fixed-alpha geometry") {
        const float alpha = GENERATE(1.0F, 2.0F);
        Vector<InnerIdType> neighbors(allocator.get());
        neighbors = {2, 4, 1, 3};
        select_edges_by_heuristic(neighbors, 0, 2, flatten, allocator.get(), alpha);
        std::sort(neighbors.begin(), neighbors.end());
        REQUIRE(neighbors.size() == (alpha == 1.0F ? 1 : 2));
        CHECK(neighbors[0] == 1);
        if (alpha == 2.0F) {
            CHECK(neighbors[1] == 4);
        }
    }

    SECTION("Alpha=1.0 baseline behavior") {
        // Initial candidates in heap (distance from base: ID1 < ID4 < ID3 < ID2)
        // Candidates: [ID1(1.0), ID4(26.0), ID3(29.0), ID2(34.0)]
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        edges->Push(d01, 1);
        edges->Push(d02, 2);
        edges->Push(d03, 3);
        edges->Push(d04, 4);

        // Pruning process with alpha=1.0 (max_size=3)
        // Step 1: Process closest node (ID1, distance=1.0)
        //         - No existing nodes in return_list, so keep ID1
        //         - return_list = [ID1]
        // Step 2: Process next node (ID4, distance=26.0)
        //         - Compare with ID1: 1.0 * 17.0 (ID1-ID4 distance) = 17.0 < 26.0 -> PRUNE ID4
        //         - return_list remains [ID1]
        // Step 3: Process next node (ID3, distance=29.0)
        //         - Compare with ID1: 1.0 * 20.0 (ID1-ID3 distance) = 20.0 < 29.0 -> PRUNE ID3
        //         - return_list remains [ID1]
        // Step 4: Process next node (ID2, distance=34.0)
        //         - Compare with ID1: 1.0 * 25.0 (ID1-ID2 distance) = 25.0 < 34.0 -> PRUNE ID2
        //         - return_list remains [ID1]
        // Final return_list size: 1
        select_edges_by_heuristic(edges, 3, flatten, allocator.get(), 1.0F);

        REQUIRE(edges->Size() == 1);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1});
    }

    SECTION("Alpha=1.5 filters some neighbors") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        edges->Push(d01, 1);
        edges->Push(d02, 2);
        edges->Push(d03, 3);
        edges->Push(d04, 4);

        // Pruning process with alpha=1.5 (max_size=3)
        // Step 1: Keep ID1,so return_list = [ID1]
        // Step 2: Process ID4: 1.5 * 17.0 = 25.5 < 26.0-> PRUNE ID4
        // Step 3: Process ID3: 1.5 * 20.0 = 30.0 < 29.0-> NO, keep ID3
        //         - return_list = [ID1, ID3]
        // Step 4: Process ID2: 1.5 * 25.0 = 37.5 < 34.0-> NO, but check ID3
        //         - 1.5 * 13.0 (ID2-ID3 distance) = 19.5 < 34.0-> PRUNE ID2
        // Final return_list size: 2
        select_edges_by_heuristic(edges, 3, flatten, allocator.get(), 1.5F);

        REQUIRE(edges->Size() == 2);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1, 3});
    }

    SECTION("Alpha=2.0 enforces strict filtering") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        edges->Push(d01, 1);
        edges->Push(d02, 2);
        edges->Push(d03, 3);
        edges->Push(d04, 4);

        //similar process
        select_edges_by_heuristic(edges, 3, flatten, allocator.get(), 2.0F);

        REQUIRE(edges->Size() == 2);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1, 4});
    }

    SECTION("Alpha=3.5") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        edges->Push(d01, 1);
        edges->Push(d02, 2);
        edges->Push(d03, 3);
        edges->Push(d04, 4);

        select_edges_by_heuristic(edges, 3, flatten, allocator.get(), 3.5F);

        REQUIRE(edges->Size() == 3);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1, 2, 4});
    }

    SECTION("Disabled adaptive policy preserves legacy forward and reverse edges") {
        auto graph_param = std::make_shared<GraphDataCellParameter>();
        graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
        graph_param->max_degree_ = 3;
        auto legacy = GraphInterface::MakeInstance(graph_param, common_param);
        auto disabled = GraphInterface::MakeInstance(graph_param, common_param);
        AdaptivePruningParameter policy;
        policy.adjust_step = 99;  // Inactive tuning must not affect the legacy path.
        auto mutexes = std::make_shared<EmptyMutex>();
        auto connect = [&](const GraphInterfacePtr& graph, const AdaptivePruningParameter* p) {
            for (InnerIdType node = 0; node < 5; ++node) {
                auto heap = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
                for (InnerIdType other = 0; other < node; ++other) {
                    heap->Push(FlattenDistanceProvider(flatten, nullptr)
                                   .PairwiseDistance(node, other, nullptr),
                               other);
                }
                mutually_connect_new_element(
                    node, heap, graph, flatten, mutexes, allocator.get(), 1.5F, p);
            }
        };
        connect(legacy, nullptr);
        connect(disabled, &policy);
        for (InnerIdType node = 0; node < 5; ++node) {
            Vector<InnerIdType> old_neighbors(allocator.get()), new_neighbors(allocator.get());
            legacy->GetNeighbors(node, old_neighbors);
            disabled->GetNeighbors(node, new_neighbors);
            CHECK(old_neighbors == new_neighbors);
        }
    }

    SECTION("Adaptive forward pruning handles short lists and removes self edges") {
        auto graph_param = std::make_shared<GraphDataCellParameter>();
        graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
        graph_param->max_degree_ = 4;
        auto graph = GraphInterface::MakeInstance(graph_param, common_param);
        auto heap = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        heap->Push(0, 0);
        heap->Push(d01, 1);
        heap->Push(d02, 2);
        AdaptivePruningParameter policy;
        policy.enabled = true;
        auto mutexes = std::make_shared<EmptyMutex>();
        CHECK(mutually_connect_new_element(
                  0, heap, graph, flatten, mutexes, allocator.get(), 1.06F, &policy) == 1);
        Vector<InnerIdType> neighbors(allocator.get());
        graph->GetNeighbors(0, neighbors);
        REQUIRE(neighbors.size() == 1);
        CHECK(neighbors.front() == 1);
        graph->GetNeighbors(1, neighbors);
        REQUIRE(neighbors.size() == 1);
        CHECK(neighbors.front() == 0);
    }

    SECTION("Mutual connection returns farthest candidate") {
        auto graph_param = std::make_shared<GraphDataCellParameter>();
        graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
        graph_param->max_degree_ = 4;
        auto graph = GraphInterface::MakeInstance(graph_param, common_param);

        auto candidates = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        candidates->Push(d01, 1);
        candidates->Push(d02, 2);
        candidates->Push(d03, 3);
        candidates->Push(d04, 4);

        auto mutexes = std::make_shared<EmptyMutex>();
        MutexArrayPtr mutex_array = std::make_shared<EmptyMutex>();
        auto entry_point = mutually_connect_new_element(
            0, candidates, graph, flatten, mutexes, allocator.get(), 1.0F);

        REQUIRE(entry_point == 1);

        Vector<InnerIdType> neighbors_0(allocator.get());
        graph->GetNeighbors(0, neighbors_0);
        REQUIRE(neighbors_0.size() == 1);
    }
}

}  // namespace vsag

namespace vsag {

TEST_CASE("Adaptive reverse pruning relaxes around the existing neighbor",
          "[ut][pruning_strategy][adaptive_pruning][reverse]") {
    auto allocator = Engine::CreateDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common.dim_ = 1;
    auto flatten_param = std::make_shared<FlattenDataCellParameter>();
    flatten_param->quantizer_parameter = std::make_shared<FP32QuantizerParameter>();
    flatten_param->io_parameter = std::make_shared<MemoryIOParameter>();
    auto flatten = FlattenInterface::MakeInstance(flatten_param, common);
    // Center 0 has a close neighbor at 1 and a distant neighbor at 20/21.
    // Baseline 1.06 rejects both distant candidates; relaxation to 1.18 admits the nearer one.
    const bool incoming_is_nearer = GENERATE(false, true);
    float vectors[] = {
        0, 1, incoming_is_nearer ? 21.0F : 20.0F, incoming_is_nearer ? 20.0F : 21.0F};
    flatten->Train(vectors, 4);
    flatten->BatchInsertVector(vectors, 4);
    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 2;
    auto graph = GraphInterface::MakeInstance(graph_param, common);
    Vector<InnerIdType> old_neighbors(allocator.get());
    old_neighbors = {1, 2};
    graph->InsertNeighborsById(0, old_neighbors);
    auto candidates = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
    candidates->Push(vectors[3] * vectors[3], 0);
    AdaptivePruningParameter policy;
    policy.enabled = GENERATE(false, true);
    policy.apply_to_reverse = GENERATE(false, true);
    policy.fill_rejected = false;
    auto mutexes = std::make_shared<EmptyMutex>();
    CHECK(mutually_connect_new_element(
              3, candidates, graph, flatten, mutexes, allocator.get(), 1.06F, &policy) == 0);
    Vector<InnerIdType> actual(allocator.get());
    graph->GetNeighbors(0, actual);
    std::sort(actual.begin(), actual.end());
    if (policy.enabled && policy.apply_to_reverse) {
        REQUIRE(actual.size() == 2);
        CHECK(actual[0] == 1);
        CHECK(actual[1] == (incoming_is_nearer ? 3 : 2));
    } else {
        REQUIRE(actual.size() == 1);
        CHECK(actual.front() == 1);
    }
    graph->GetNeighbors(3, actual);
    REQUIRE(actual.size() == 1);
    CHECK(actual.front() == 0);
}

TEST_CASE("Adaptive reverse pruning preserves append and optional fill",
          "[ut][pruning_strategy][adaptive_pruning][reverse]") {
    auto allocator = Engine::CreateDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common.dim_ = 1;
    auto flatten_param = std::make_shared<FlattenDataCellParameter>();
    flatten_param->quantizer_parameter = std::make_shared<FP32QuantizerParameter>();
    flatten_param->io_parameter = std::make_shared<MemoryIOParameter>();
    auto flatten = FlattenInterface::MakeInstance(flatten_param, common);
    float vectors[] = {0, 1, 2, 3};
    flatten->Train(vectors, 4);
    flatten->BatchInsertVector(vectors, 4);
    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 2;
    auto graph = GraphInterface::MakeInstance(graph_param, common);
    const bool full = GENERATE(false, true);
    AdaptivePruningParameter policy;
    policy.enabled = true;
    policy.apply_to_reverse = true;
    policy.fill_rejected = GENERATE(false, true);
    Vector<InnerIdType> neighbors(allocator.get());
    neighbors.push_back(1);
    if (full) {
        neighbors.push_back(2);
    }
    graph->InsertNeighborsById(0, neighbors);
    auto candidates = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
    candidates->Push(9, 0);
    auto mutexes = std::make_shared<EmptyMutex>();
    mutually_connect_new_element(
        3, candidates, graph, flatten, mutexes, allocator.get(), 1.06F, &policy);
    graph->GetNeighbors(0, neighbors);
    std::sort(neighbors.begin(), neighbors.end());
    if (not full) {
        // Running the selector here with fill=false would incorrectly drop the new edge.
        REQUIRE(neighbors.size() == 2);
        CHECK(neighbors[0] == 1);
        CHECK(neighbors[1] == 3);
    } else if (policy.fill_rejected) {
        REQUIRE(neighbors.size() == 2);
        CHECK(neighbors[0] == 1);
        CHECK(neighbors[1] == 2);
    } else {
        REQUIRE(neighbors.size() == 1);
        CHECK(neighbors[0] == 1);
    }
}

}  // namespace vsag
