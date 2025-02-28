
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

namespace vsag {

void
select_edges_by_heuristic(MaxHeap& edges,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator) {
    if (edges.size() < max_size) {
        return;
    }

    MaxHeap queue_closest(allocator);
    vsag::Vector<std::pair<float, InnerIdType>> return_list(allocator);
    while (not edges.empty()) {
        queue_closest.emplace(-edges.top().first, edges.top().second);
        edges.pop();
    }

    while (not queue_closest.empty()) {
        if (return_list.size() >= max_size) {
            break;
        }
        std::pair<float, InnerIdType> curent_pair = queue_closest.top();
        float float_query = -curent_pair.first;
        queue_closest.pop();
        bool good = true;

        for (const auto& second_pair : return_list) {
            float curdist = flatten->ComputePairVectors(second_pair.second, curent_pair.second);
            if (curdist < float_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.emplace_back(curent_pair);
        }
    }

    for (const auto& curent_pair : return_list) {
        edges.emplace(-curent_pair.first, curent_pair.second);
    }
}

InnerIdType
mutually_connect_new_element(InnerIdType cur_c,
                             MaxHeap& top_candidates,
                             const GraphInterfacePtr& graph,
                             const FlattenInterfacePtr& flatten,
                             const MutexArrayPtr& neighbors_mutexs,
                             Allocator* allocator) {
    const size_t max_size = graph->MaximumDegree();
    select_edges_by_heuristic(top_candidates, max_size, flatten, allocator);
    if (top_candidates.size() > max_size) {
        throw std::runtime_error(
            "Should be not be more than max_size candidates returned by the heuristic");
    }

    Vector<InnerIdType> selected_neighbors(allocator);
    selected_neighbors.reserve(max_size);
    while (not top_candidates.empty()) {
        selected_neighbors.emplace_back(top_candidates.top().second);
        top_candidates.pop();
    }

    InnerIdType next_closest_entry_point = selected_neighbors.back();

    graph->InsertNeighborsById(cur_c, selected_neighbors);

    for (auto selected_neighbor : selected_neighbors) {
        if (selected_neighbor == cur_c) {
            throw std::runtime_error("Trying to connect an element to itself");
        }

        LockGuard lock(neighbors_mutexs, selected_neighbor);

        Vector<InnerIdType> neighbors(allocator);
        graph->GetNeighbors(selected_neighbor, neighbors);

        size_t sz_link_list_other = neighbors.size();

        if (sz_link_list_other > max_size) {
            throw std::runtime_error("Bad value of sz_link_list_other");
        }
        // If cur_c is already present in the neighboring connections of `selected_neighbors[idx]` then no need to modify any connections or run the heuristics.
        if (sz_link_list_other < max_size) {
            neighbors.emplace_back(cur_c);
            graph->InsertNeighborsById(selected_neighbor, neighbors);
        } else {
            // finding the "weakest" element to replace it with the new one
            float d_max = flatten->ComputePairVectors(cur_c, selected_neighbor);

            MaxHeap candidates(allocator);
            candidates.emplace(d_max, cur_c);

            for (size_t j = 0; j < sz_link_list_other; j++) {
                candidates.emplace(flatten->ComputePairVectors(neighbors[j], selected_neighbor),
                                   neighbors[j]);
            }

            select_edges_by_heuristic(candidates, max_size, flatten, allocator);

            Vector<InnerIdType> cand_neighbors(allocator);
            while (not candidates.empty()) {
                cand_neighbors.emplace_back(candidates.top().second);
                candidates.pop();
            }
            graph->InsertNeighborsById(selected_neighbor, cand_neighbors);
        }
    }
    return next_closest_entry_point;
}

}  // namespace vsag
