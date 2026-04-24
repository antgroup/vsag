
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

#pragma once

#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER2(DistHeap, DistanceHeap);
DEFINE_POINTER(FlattenInterface);
DEFINE_POINTER(GraphInterface);
DEFINE_POINTER(MutexArray);

/**
 * @brief Selects edges using heuristic pruning on a distance heap.
 *
 * This function applies a diversity-based heuristic to select up to max_size edges
 * from the given heap. It prefers edges that are both close to the query point and
 * well-distributed in the vector space, avoiding redundant neighbors.
 *
 * @param edges Distance heap containing candidate edges (distance, id pairs).
 *              Modified in-place to contain selected edges.
 * @param max_size Maximum number of edges to select.
 * @param flatten Flatten interface for computing pairwise vector distances.
 * @param allocator Allocator for memory management.
 * @param alpha Diversity parameter controlling the trade-off between proximity
 *              and diversity. Higher values allow more diverse neighbors.
 */
void
select_edges_by_heuristic(const DistHeapPtr& edges,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator,
                          float alpha = 1.0F);

/**
 * @brief Selects edges using heuristic pruning on a neighbor vector.
 *
 * This overload computes distances from node_id to each neighbor and applies
 * the same diversity-based heuristic as the heap version. Selected neighbors
 * are stored back into the neighbors vector.
 *
 * @param neighbors Vector of neighbor IDs to be filtered. Modified in-place
 *                  to contain only the selected neighbors.
 * @param node_id The reference node ID for computing distances to neighbors.
 * @param max_size Maximum number of neighbors to select.
 * @param flatten Flatten interface for computing pairwise vector distances.
 * @param allocator Allocator for memory management.
 * @param alpha Diversity parameter controlling the trade-off between proximity
 *              and diversity. Higher values allow more diverse neighbors.
 */
void
select_edges_by_heuristic(Vector<InnerIdType>& neighbors,
                          InnerIdType node_id,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator,
                          float alpha = 1.0F);

/**
 * @brief Connects a new element to the graph using mutual edge selection.
 *
 * This function inserts a new element into the graph by selecting its neighbors
 * using the heuristic edge selection, and then updating all selected neighbors
 * to maintain bidirectional connections. If a neighbor's degree exceeds max_size,
 * it triggers re-selection of that neighbor's edges.
 *
 * @param cur_c The ID of the new element to be connected.
 * @param top_candidates Distance heap containing candidate neighbors for cur_c.
 * @param graph Graph interface for storing and retrieving neighbor connections.
 * @param flatten Flatten interface for computing pairwise vector distances.
 * @param neighbors_mutexes Mutex array for thread-safe neighbor updates.
 * @param allocator Allocator for memory management.
 * @param alpha Diversity parameter for heuristic edge selection.
 * @return InnerIdType The ID of the farthest selected neighbor, typically used
 *                     as an entry point for subsequent operations.
 */
InnerIdType
mutually_connect_new_element(InnerIdType cur_c,
                             const DistHeapPtr& top_candidates,
                             const GraphInterfacePtr& graph,
                             const FlattenInterfacePtr& flatten,
                             const MutexArrayPtr& neighbors_mutexes,
                             Allocator* allocator,
                             float alpha = 1.0F);

}  // namespace vsag
