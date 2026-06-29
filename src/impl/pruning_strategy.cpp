
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

#include "datacell/flatten_datacell.h"
#include "datacell/graph_interface.h"
#include "hash_types.h"
#include "impl/heap/standard_heap.h"
#include "utils/lock_strategy.h"
namespace vsag {

namespace {

class PairwiseDistanceComputer {
public:
    PairwiseDistanceComputer(const DistanceProviderForGraph& distance_provider,
                             Allocator* allocator)
        : distance_provider_(distance_provider), computers_(allocator) {
    }

    PairwiseDistanceComputer(const PairwiseDistanceComputer&) = delete;
    PairwiseDistanceComputer&
    operator=(const PairwiseDistanceComputer&) = delete;
    PairwiseDistanceComputer(PairwiseDistanceComputer&&) = delete;
    PairwiseDistanceComputer&
    operator=(PairwiseDistanceComputer&&) = delete;

    float
    PairwiseDistance(InnerIdType id1, InnerIdType id2) {
        if (not distance_provider_.SupportsComputerById()) {
            return distance_provider_.PairwiseDistance(id1, id2);
        }
        auto& computer = computers_[id1];
        if (computer == nullptr) {
            computer = distance_provider_.FactoryComputerById(id1);
        }
        return distance_provider_.PairwiseDistance(id1, id2, computer);
    }

private:
    const DistanceProviderForGraph& distance_provider_;
    // The helper lives for one pruning pass; its source IDs are bounded by the selected-edge limit.
    UnorderedMap<InnerIdType, ComputerInterfacePtr> computers_;
};

}  // namespace

void
select_edges_by_heuristic(Vector<InnerIdType>& neighbors,
                          InnerIdType node_id,
                          uint64_t max_size,
                          const DistanceProviderForGraph& distance_provider,
                          Allocator* allocator,
                          float alpha) {
    PairwiseDistanceComputer pairwise_distance(distance_provider, allocator);
    auto edges = std::make_shared<StandardHeap<true, false>>(allocator, -1);
    for (const auto& neighbor : neighbors) {
        float dist = pairwise_distance.PairwiseDistance(node_id, neighbor);
        edges->Push(dist, neighbor);
    }

    select_edges_by_heuristic(edges, max_size, distance_provider, allocator, alpha);

    neighbors.clear();
    while (not edges->Empty()) {
        neighbors.emplace_back(edges->Top().second);
        edges->Pop();
    }
}

void
select_edges_by_heuristic(const DistHeapPtr& edges,
                          uint64_t max_size,
                          const DistanceProviderForGraph& distance_provider,
                          Allocator* allocator,
                          float alpha) {
    if (edges->Size() < max_size) {
        return;
    }

    auto queue_closest = std::make_shared<StandardHeap<true, false>>(allocator, -1);
    Vector<std::pair<float, InnerIdType>> return_list(allocator);
    PairwiseDistanceComputer pairwise_distance(distance_provider, allocator);
    while (not edges->Empty()) {
        queue_closest->Push(-edges->Top().first, edges->Top().second);
        edges->Pop();
    }

    while (not queue_closest->Empty()) {
        if (return_list.size() >= max_size) {
            break;
        }
        std::pair<float, InnerIdType> current_pair = queue_closest->Top();
        float float_query = -current_pair.first;
        queue_closest->Pop();
        bool good = true;

        for (const auto& second_pair : return_list) {
            float curdist =
                pairwise_distance.PairwiseDistance(second_pair.second, current_pair.second);
            if (alpha * curdist < float_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.emplace_back(current_pair);
        }
    }

    for (const auto& current_pair : return_list) {
        edges->Push(-current_pair.first, current_pair.second);
    }
}

InnerIdType
mutually_connect_new_element(InnerIdType cur_c,
                             const DistHeapPtr& top_candidates,
                             const GraphInterfacePtr& graph,
                             const DistanceProviderForGraph& distance_provider,
                             const MutexArrayPtr& neighbors_mutexes,
                             Allocator* allocator,
                             float alpha) {
    PairwiseDistanceComputer pairwise_distance(distance_provider, allocator);
    const uint64_t max_size = graph->MaximumDegree();
    select_edges_by_heuristic(top_candidates, max_size, distance_provider, allocator, alpha);
    if (top_candidates->Size() > max_size) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            "Should be not be more than max_size candidates returned by the heuristic");
    }

    Vector<InnerIdType> selected_neighbors(allocator);
    selected_neighbors.reserve(max_size);
    while (not top_candidates->Empty()) {
        selected_neighbors.emplace_back(top_candidates->Top().second);
        top_candidates->Pop();
    }

    InnerIdType next_closest_entry_point = selected_neighbors.back();

    graph->InsertNeighborsById(cur_c, selected_neighbors);

    for (auto selected_neighbor : selected_neighbors) {
        if (selected_neighbor == cur_c) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Trying to connect an element to itself");
        }

        LockGuard lock(neighbors_mutexes, selected_neighbor);

        Vector<InnerIdType> neighbors(allocator);
        graph->GetNeighbors(selected_neighbor, neighbors);

        uint64_t sz_link_list_other = neighbors.size();

        if (sz_link_list_other > max_size) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "Bad value of sz_link_list_other");
        }
        bool has_reverse_edge = false;
        for (auto neighbor : neighbors) {
            if (neighbor == cur_c) {
                has_reverse_edge = true;
                break;
            }
        }
        if (has_reverse_edge) {
            continue;
        }

        if (sz_link_list_other < max_size) {
            neighbors.emplace_back(cur_c);
            graph->InsertNeighborsById(selected_neighbor, neighbors);
        } else {
            // finding the "weakest" element to replace it with the new one
            float d_max = pairwise_distance.PairwiseDistance(selected_neighbor, cur_c);

            auto candidates = std::make_shared<StandardHeap<true, false>>(allocator, -1);
            candidates->Push(d_max, cur_c);

            for (uint64_t j = 0; j < sz_link_list_other; j++) {
                candidates->Push(
                    pairwise_distance.PairwiseDistance(selected_neighbor, neighbors[j]),
                    neighbors[j]);
            }

            select_edges_by_heuristic(candidates, max_size, distance_provider, allocator, alpha);

            Vector<InnerIdType> cand_neighbors(allocator);
            while (not candidates->Empty()) {
                cand_neighbors.emplace_back(candidates->Top().second);
                candidates->Pop();
            }

            graph->InsertNeighborsById(selected_neighbor, cand_neighbors);
        }
    }

    return next_closest_entry_point;
}

void
select_edges_by_heuristic(Vector<InnerIdType>& neighbors,
                          InnerIdType node_id,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator,
                          float alpha) {
    FlattenDistanceProvider distance_provider(flatten, nullptr);
    select_edges_by_heuristic(neighbors, node_id, max_size, distance_provider, allocator, alpha);
}

void
select_edges_by_heuristic(const DistHeapPtr& edges,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator,
                          float alpha) {
    FlattenDistanceProvider distance_provider(flatten, nullptr);
    select_edges_by_heuristic(edges, max_size, distance_provider, allocator, alpha);
}

InnerIdType
mutually_connect_new_element(InnerIdType cur_c,
                             const DistHeapPtr& top_candidates,
                             const GraphInterfacePtr& graph,
                             const FlattenInterfacePtr& flatten,
                             const MutexArrayPtr& neighbors_mutexes,
                             Allocator* allocator,
                             float alpha) {
    FlattenDistanceProvider distance_provider(flatten, nullptr);
    return mutually_connect_new_element(
        cur_c, top_candidates, graph, distance_provider, neighbors_mutexes, allocator, alpha);
}

}  // namespace vsag
