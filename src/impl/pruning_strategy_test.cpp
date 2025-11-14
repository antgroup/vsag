
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
#include <cmath>
#include <memory>
#include <vector>

#include "datacell/flatten_datacell.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/graph_datacell_parameter.h"
#include "datacell/graph_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/heap/standard_heap.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "typing.h"
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
        select_edges_by_heuristic<EdgeSelectionParam::ALPHA>(
            edges, 3, flatten, allocator.get(), 1.0F);

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
        select_edges_by_heuristic<EdgeSelectionParam::ALPHA>(
            edges, 3, flatten, allocator.get(), 1.5F);

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
        select_edges_by_heuristic<EdgeSelectionParam::ALPHA>(
            edges, 3, flatten, allocator.get(), 2.0F);

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

        select_edges_by_heuristic<EdgeSelectionParam::ALPHA>(
            edges, 3, flatten, allocator.get(), 3.5F);

        REQUIRE(edges->Size() == 3);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1, 2, 4});
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
        auto entry_point = mutually_connect_new_element<EdgeSelectionParam::ALPHA>(
            0, candidates, graph, flatten, mutexes, allocator.get(), 1.0F);

        REQUIRE(entry_point == 1);

        Vector<InnerIdType> neighbors_0(allocator.get());
        graph->GetNeighbors(0, neighbors_0);
        REQUIRE(neighbors_0.size() == 1);
    }
}

TEST_CASE("Pruning Strategy Select Edges With Tau-MNG", "[ut][pruning_strategy]") {
    auto allocator = Engine::CreateDefaultAllocator();
    auto flatten_param = std::make_shared<FlattenDataCellParameter>();
    flatten_param->quantizer_parameter = std::make_shared<FP32QuantizerParameter>();
    flatten_param->io_parameter = std::make_shared<MemoryIOParameter>();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.dim_ = 128;

    auto flatten = FlattenInterface::MakeInstance(flatten_param, common_param);
    REQUIRE(flatten != nullptr);

    float vectors[5][128] = {0};
    vectors[0][0] = 0.0F;
    vectors[1][0] = 1.0F;
    vectors[2][0] = 5.0F;
    vectors[3][0] = 10.0F;
    vectors[4][0] = 20.0F;

    flatten->Train(vectors, 5);
    flatten->BatchInsertVector(vectors, 5);

    const float d01 = 1.0F;    // distance to ID1
    const float d02 = 25.0F;   // distance to ID2
    const float d03 = 100.0F;  // distance to ID3
    const float d04 = 400.0F;  // distance to ID4

    SECTION("Tau-MNG with small tau value") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        edges->Push(d01, 1);
        edges->Push(d02, 2);
        edges->Push(d03, 3);
        edges->Push(d04, 4);

        // τ-MNG pruning with tau=2.0, alpha=1.0 (max_size=3)
        // τ-MNG rule: if curdist < (float_query - 3 * tau), prune
        // For tau=2.0, 3*tau=6.0
        // Step 1: Keep ID1 (distance=1.0), return_list = [ID1]
        // Step 2: Process ID2 (distance=25.0)
        //         - Compare with ID1: dist(ID1,ID2)=16.0
        //         - Check: 16.0 < (25.0 - 6.0)=19.0? YES -> prune ID2
        //         - return_list = [ID1, ID2]
        // Step 3: Process ID3 (distance=100.0)
        //         - Compare with ID1: dist(ID1,ID3)=99.0 < (100.0-6.0)=94.0? NO
        //         - Compare with ID2: dist(ID2,ID3)=75.0 < (100.0-6.0)=94.0? YES -> prune ID3
        // Step 4: Process ID4 (distance=400.0)
        //         - Compare with ID1: dist(ID1,ID4)=399.0 < (400.0-6.0)=394.0? NO
        //         - Compare with ID2: dist(ID2,ID4)=375.0 < (400.0-6.0)=394.0? YES -> prune ID4
        // Final return_list: [ID1, ID2]
        select_edges_by_heuristic<EdgeSelectionParam::TAU>(
            edges, 3, flatten, allocator.get(), 2.0F);

        REQUIRE(edges->Size() == 1);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1});
    }

    SECTION("Tau-MNG with larger tau value") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        edges->Push(d01, 1);
        edges->Push(d02, 2);
        edges->Push(d03, 3);
        edges->Push(d04, 4);

        // τ-MNG pruning with tau=5.0, alpha=1.0 (max_size=3)
        // For tau=5.0, 3*tau=15.0
        // Step 1: Keep ID1, return_list = [ID1]
        // Step 2: Process ID2 (distance=25.0)
        //         - dist(ID1,ID2)=24.0 < (25.0-15.0)=10.0? NO -> keep ID2
        //         - return_list = [ID1, ID2]
        // Step 3: Process ID3 (distance=100.0)
        //         - dist(ID1,ID3)=99.0 < (100.0-15.0)=85.0? NO
        //         - dist(ID2,ID3)=75.0 < (100.0-15.0)=85.0? YES -> prune ID3
        // Step 4: Process ID4 (distance=400.0)
        //         - dist(ID1,ID4)=399.0 < (400.0-15.0)=385.0? NO
        //         - dist(ID2,ID4)=375.0 < (400.0-15.0)=385.0? YES -> prune ID4
        // Final return_list: [ID1, ID2]
        select_edges_by_heuristic<EdgeSelectionParam::TAU>(
            edges, 3, flatten, allocator.get(), 5.0F);

        REQUIRE(edges->Size() == 2);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1, 2});
    }

    SECTION("Tau-MNG with very small tau (approaching original heuristic)") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        edges->Push(d01, 1);
        edges->Push(d02, 2);
        edges->Push(d03, 3);
        edges->Push(d04, 4);

        // τ-MNG with very small tau=0.1 should behave similarly to original heuristic
        // For tau=0.1, 3*tau=0.3
        // This will be very close to original heuristic behavior
        select_edges_by_heuristic<EdgeSelectionParam::TAU>(
            edges, 3, flatten, allocator.get(), 0.1F);

        REQUIRE(edges->Size() == 1);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1});
    }

    SECTION("Tau-MNG with short edge forced connection") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);

        // Add some very short edges that should be forced to connect
        const float short_dist_1 = 2.0F;  // <= 3*tau for tau=1.0
        const float short_dist_2 = 2.5F;  // <= 3*tau for tau=1.0

        edges->Push(short_dist_1, 1);
        edges->Push(short_dist_2, 2);
        edges->Push(d03, 3);
        edges->Push(d04, 4);

        // τ-MNG with tau=1.0, short edges should be forced to connect
        // For tau=1.0, 3*tau=3.0
        // ID1 (dist=2.0) and ID2 (dist=2.5) are both <= 3.0, so they should be kept
        // regardless of other conditions due to τ-MNG forced connection rule
        select_edges_by_heuristic<EdgeSelectionParam::TAU>(
            edges, 3, flatten, allocator.get(), 1.0F);

        REQUIRE(edges->Size() == 2);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1, 2});
    }

    SECTION("Mutual connection with Tau-MNG algorithm") {
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

        auto entry_point = mutually_connect_new_element<EdgeSelectionParam::TAU>(
            0, candidates, graph, flatten, mutex_array, allocator.get(), 2.0F);
        REQUIRE(entry_point == 1);
        Vector<InnerIdType> neighbors_0(allocator.get());
        graph->GetNeighbors(0, neighbors_0);
        REQUIRE(neighbors_0.size() == 1);  // ID1 should be kept
    }

    SECTION("Tau-MNG with zero tau falls back to original heuristic") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        edges->Push(d01, 1);
        edges->Push(d02, 2);
        edges->Push(d03, 3);
        edges->Push(d04, 4);

        // tau=0 should behave exactly like original heuristic with alpha=1.0
        select_edges_by_heuristic<EdgeSelectionParam::TAU>(
            edges, 3, flatten, allocator.get(), 0.0F);

        REQUIRE(edges->Size() == 1);
        std::vector<InnerIdType> kept;
        while (!edges->Empty()) {
            kept.push_back(edges->Top().second);
            edges->Pop();
        }
        std::sort(kept.begin(), kept.end());
        REQUIRE(kept == std::vector<InnerIdType>{1});
    }
}

}  // namespace vsag
