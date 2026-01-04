
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
#include "impl/heap/standard_heap.h"
#include "utils/lock_strategy.h"
namespace vsag {

template <EdgeSelectionParam param>
void
select_edges_by_heuristic(const DistHeapPtr& edges,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator,
                          float param_value) {
    if (edges->Size() < max_size) {
        return;
    }

    auto queue_closest = std::make_shared<StandardHeap<true, false>>(allocator, -1);
    Vector<std::pair<float, InnerIdType>> return_list(allocator);
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
            float curdist = flatten->ComputePairVectors(second_pair.second, current_pair.second);

            if constexpr (param == EdgeSelectionParam::ALPHA) {
                if (param_value * curdist < float_query) {
                    good = false;
                    break;
                }
            } else {
                if (curdist < (float_query - 3 * param_value)) {
                    good = false;
                    break;
                }
                if (float_query <= 3 * param_value) {
                    good = true;
                    break;
                }
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

template <EdgeSelectionParam param>
InnerIdType
mutually_connect_new_element(InnerIdType cur_c,
                             const DistHeapPtr& top_candidates,
                             const GraphInterfacePtr& graph,
                             const FlattenInterfacePtr& flatten,
                             const MutexArrayPtr& neighbors_mutexes,
                             Allocator* allocator,
                             float param_value) {
    const size_t max_size = graph->MaximumDegree();

    select_edges_by_heuristic<param>(top_candidates, max_size, flatten, allocator, param_value);

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

        size_t sz_link_list_other = neighbors.size();

        if (sz_link_list_other > max_size) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "Bad value of sz_link_list_other");
        }
        // If cur_c is already present in the neighboring connections of `selected_neighbors[idx]` then no need to modify any connections or run the heuristics.
        if (sz_link_list_other < max_size) {
            neighbors.emplace_back(cur_c);
            graph->InsertNeighborsById(selected_neighbor, neighbors);
        } else {
            // finding the "weakest" element to replace it with the new one
            float d_max = flatten->ComputePairVectors(cur_c, selected_neighbor);

            auto candidates = std::make_shared<StandardHeap<true, false>>(allocator, -1);
            candidates->Push(d_max, cur_c);

            for (size_t j = 0; j < sz_link_list_other; j++) {
                candidates->Push(flatten->ComputePairVectors(neighbors[j], selected_neighbor),
                                 neighbors[j]);
            }

            select_edges_by_heuristic<param>(candidates, max_size, flatten, allocator, param_value);

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

template void
select_edges_by_heuristic<EdgeSelectionParam::ALPHA>(
    const DistHeapPtr&, uint64_t, const FlattenInterfacePtr&, Allocator*, float);

template void
select_edges_by_heuristic<EdgeSelectionParam::TAU>(
    const DistHeapPtr&, uint64_t, const FlattenInterfacePtr&, Allocator*, float);

template InnerIdType
mutually_connect_new_element<EdgeSelectionParam::ALPHA>(InnerIdType,
                                                        const DistHeapPtr&,
                                                        const GraphInterfacePtr&,
                                                        const FlattenInterfacePtr&,
                                                        const MutexArrayPtr&,
                                                        Allocator*,
                                                        float);

template InnerIdType
mutually_connect_new_element<EdgeSelectionParam::TAU>(InnerIdType,
                                                      const DistHeapPtr&,
                                                      const GraphInterfacePtr&,
                                                      const FlattenInterfacePtr&,
                                                      const MutexArrayPtr&,
                                                      Allocator*,
                                                      float);

}  // namespace vsag
