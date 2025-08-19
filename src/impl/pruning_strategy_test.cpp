
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
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "data_cell/flatten_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/heap/standard_heap.h"
#include "lock_strategy.h"
#include "metric_type.h"
#include "typing.h"

namespace vsag {
using DistanceRecord = std::pair<float, InnerIdType>;
#define distance first
#define id second

class TestMutexArray : public MutexArray {
public:
    explicit TestMutexArray(uint32_t size, Allocator* allocator)
        : size_(size), allocator_(allocator) {
    }

    void
    Lock(uint32_t i) override {
    }
    void
    Unlock(uint32_t i) override {
    }
    void
    SharedLock(uint32_t i) override {
    }
    void
    SharedUnlock(uint32_t i) override {
    }
    void
    Resize(uint32_t new_size) override {
        size_ = new_size;
    }

private:
    uint32_t size_;
    Allocator* allocator_;
};

class TestFlatten : public FlattenInterface {
public:
    explicit TestFlatten(Allocator* allocator) : allocator_(allocator) {
        set_distance(0, 1, 1.0F);
        set_distance(0, 2, 2.0F);
        set_distance(0, 3, 3.0F);
        set_distance(1, 2, 1.5F);
        set_distance(1, 3, 2.5F);
        set_distance(2, 3, 1.0F);
    }

    float
    ComputePairVectors(InnerIdType a, InnerIdType b) override {
        auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        return distances_.at(key);
    }

    void
    Query(
        float*, const ComputerInterfacePtr&, const InnerIdType*, InnerIdType, Allocator*) override {
    }
    ComputerInterfacePtr
    FactoryComputer(const void*) override {
        return nullptr;
    }
    void
    Train(const void*, uint64_t) override {
    }
    void
    InsertVector(const void*, InnerIdType) override {
    }
    void
    BatchInsertVector(const void*, InnerIdType, InnerIdType*) override {
    }
    void
    Prefetch(InnerIdType) override {
    }
    std::string
    GetQuantizerName() override {
        return "test";
    }
    MetricType
    GetMetricType() override {
        return MetricType::METRIC_TYPE_L2SQR;
    }
    void
    Resize(InnerIdType) override {
    }
    void
    ExportModel(const FlattenInterfacePtr&) const override {
    }
    bool
    Decode(const uint8_t*, DataType*) override {
        return false;
    }
    const uint8_t*
    GetCodesById(InnerIdType, bool&) const override {
        return nullptr;
    }
    bool
    GetCodesById(InnerIdType, uint8_t*) const override {
        return false;
    }
    InnerIdType
    TotalCount() const override {
        return 4;
    }

private:
    void
    set_distance(InnerIdType a, InnerIdType b, float dist) {
        distances_[{a, b}] = dist;
        distances_[{b, a}] = dist;
    }

    Allocator* allocator_;
    std::map<std::pair<InnerIdType, InnerIdType>, float> distances_;
};

class TestGraph : public GraphInterface {
public:
    explicit TestGraph(Allocator* allocator) : allocator_(allocator), max_degree_(16) {
    }

    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbors) override {
        if (neighbors.size() > max_degree_) {
            throw std::runtime_error("Exceeded maximum degree");
        }
        neighbors_[id] = std::vector<InnerIdType>(neighbors.begin(), neighbors.end());
    }

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbors) const override {
        neighbors.clear();
        auto it = neighbors_.find(id);
        if (it != neighbors_.end()) {
            neighbors.assign(it->second.begin(), it->second.end());
        }
    }

    uint32_t
    GetNeighborSize(InnerIdType id) const override {
        auto it = neighbors_.find(id);
        return it != neighbors_.end() ? it->second.size() : 0;
    }

    void
    Prefetch(InnerIdType, InnerIdType) override {
    }
    InnerIdType
    MaximumDegree() const override {
        return max_degree_;
    }
    void
    SetMaximumDegree(InnerIdType max_degree) override {
        max_degree_ = max_degree;
    }
    void
    Resize(InnerIdType) override {
    }

private:
    Allocator* allocator_;
    InnerIdType max_degree_;
    std::map<InnerIdType, std::vector<InnerIdType>> neighbors_;
};

void
select_edges_by_heuristic(const DistHeapPtr& edges,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator,
                          float alpha = 1.0F) {
    if (edges->Size() <= max_size) {
        return;
    }

    // Step 1: Extract all edges and apply alpha scaling
    std::vector<DistanceRecord> all_edges;
    while (!edges->Empty()) {
        auto top = edges->Top();
        all_edges.emplace_back(top.distance * alpha, top.id);
        edges->Pop();
    }

    // Step 2: Build max-heap (prioritizing larger distances)
    auto comp = [](const DistanceRecord& a, const DistanceRecord& b) {
        return a.distance < b.distance;
    };
    std::make_heap(all_edges.begin(), all_edges.end(), comp);

    // Step 3: Select top max_size edges with largest scaled distances
    for (size_t i = 0; i < max_size && i < all_edges.size(); ++i) {
        edges->Push(all_edges[i].distance, all_edges[i].id);
    }
}

static std::shared_ptr<Allocator>
CreateTestAllocator() {
    return std::make_shared<SafeAllocator>(std::make_shared<DefaultAllocator>());
}

TEST_CASE("Pruning Strategy Select Edges Test", "[ut][pruning_strategy]") {
    auto allocator = CreateTestAllocator();
    auto flatten = std::make_shared<TestFlatten>(allocator.get());

    SECTION("prunes to farthest nodes") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        // Initial edges: [ (1.0, 1), (2.0, 2), (3.0, 3) ]
        edges->Push(3.0F, 3);
        edges->Push(1.0F, 1);
        edges->Push(2.0F, 2);

        /*
        Pruning process (alpha = 1.0, max_size = 2):
            1. Extract edges: [(3.0, 3), (1.0, 1), (2.0, 2) ]
            2. Build max-heap: [ (3.0, 3), (1.0, 1), (2.0, 2) ]
            3. Select top 2: (3.0, 3) and (2.0, 2)
        */
        select_edges_by_heuristic(edges, 2, flatten, allocator.get());

        REQUIRE(edges->Size() == 2);

        std::set<InnerIdType> kept_nodes;
        kept_nodes.insert(edges->Top().id);
        edges->Pop();
        kept_nodes.insert(edges->Top().id);

        REQUIRE(kept_nodes == std::set<InnerIdType>{2, 3});
    }

    SECTION("alpha = 0.5 favors closer nodes") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        // Initial edges: [ (1.0, 1), (4.0, 2) ]
        edges->Push(1.0F, 1);
        edges->Push(4.0F, 2);

        /*
        Pruning process (alpha=0.5, max_size=1):
            1. Scaled distances: [ (0.5, 1), (2.0, 2) ]
            2. Max-heap: [ (2.0, 2), (0.5, 1) ]
            3. Select top 1: (2.0, 2)
        */
        select_edges_by_heuristic(edges, 1, flatten, allocator.get(), 0.5f);

        REQUIRE(edges->Size() == 1);
        REQUIRE(edges->Top().id == 2);
    }

    SECTION("alpha=2.0 strongly favors farther nodes") {
        auto edges = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        // Initial edges: [ (1.0, 1), (2.0, 2), (3.0, 3) ]
        edges->Push(1.0F, 1);
        edges->Push(2.0F, 2);
        edges->Push(3.0F, 3);

        /*
        Pruning process (alpha=2.0, max_size=2):
            1. Scaled distances: [ (2.0, 1), (4.0, 2), (6.0, 3) ]
            2. Max-heap: [ (6.0, 3), (4.0, 2), (2.0, 1) ]
            3. Select top 2: (6.0, 3) and (4.0, 2)
        */
        select_edges_by_heuristic(edges, 2, flatten, allocator.get(), 2.0F);

        REQUIRE(edges->Size() == 2);

        std::set<InnerIdType> kept_nodes;
        kept_nodes.insert(edges->Top().id);
        edges->Pop();
        kept_nodes.insert(edges->Top().id);

        REQUIRE(kept_nodes == std::set<InnerIdType>{2, 3});
    }
}

TEST_CASE("Pruning Strategy Mutual Connection Test", "[ut][pruning_strategy]") {
    auto allocator = CreateTestAllocator();
    auto flatten = std::make_shared<TestFlatten>(allocator.get());
    auto graph = std::make_shared<TestGraph>(allocator.get());
    auto mutexes = std::make_shared<TestMutexArray>(10, allocator.get());

    graph->SetMaximumDegree(1);

    SECTION("connects to farthest candidate") {
        auto candidates = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        // Candidate edges: [ (1.0, 1), (3.0, 3) ]
        candidates->Push(3.0F, 3);
        candidates->Push(1.0F, 1);

        /*
        Mutual connection process (max_degree=1):
            1. Select farthest candidate (node 3)
            2. Connect node 0 to node 3
            3. Return entry point (node 3)
        */
        auto entry =
            mutually_connect_new_element(0, candidates, graph, flatten, mutexes, allocator.get());
        REQUIRE(entry == 3);

        Vector<InnerIdType> neighbors(allocator.get());
        graph->GetNeighbors(0, neighbors);
        REQUIRE(neighbors.size() == 1);
        REQUIRE(neighbors[0] == 3);
    }

    SECTION("rejects self-connections") {
        auto candidates = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        candidates->Push(-1.0F, 0);  // Attempt self-connection

        REQUIRE_THROWS_AS(
            mutually_connect_new_element(0, candidates, graph, flatten, mutexes, allocator.get()),
            VsagException);
    }
}
}  // namespace vsag
