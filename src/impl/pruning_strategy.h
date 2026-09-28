
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

#include "impl/distance_provider_for_graph.h"
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
                          const DistanceProviderForGraph& distance_provider,
                          Allocator* allocator,
                          float alpha = 1.0F);

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
                          const DistanceProviderForGraph& distance_provider,
                          Allocator* allocator,
                          float alpha = 1.0F);

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
/**
 * @brief Phase A of edge construction: select cur_c's own outgoing edges and write them.
 *
 * Prunes @p top_candidates down to at most MaximumDegree neighbours, copies them into
 * @p selected_neighbors_out, and writes them as cur_c's neighbour list. Only cur_c's own
 * slots are touched, and cur_c is not reachable from the graph until that write completes,
 * so this phase needs no neighbour lock. It is the parallel-safe half of
 * mutually_connect_new_element and is what a batched build drives in parallel.
 *
 * @param cur_c The new element whose outgoing edges are being built.
 * @param top_candidates Candidate neighbours; consumed by the heuristic.
 * @param graph Graph interface for storing neighbour connections.
 * @param distance_provider Supplies pairwise distances.
 * @param allocator Allocator for temporary storage.
 * @param alpha Diversity parameter for the heuristic.
 * @param selected_neighbors_out Receives the chosen neighbours, farthest first.
 * @return The farthest selected neighbour, used as the next entry point.
 */
InnerIdType
build_forward_links(InnerIdType cur_c,
                    const DistHeapPtr& top_candidates,
                    const GraphInterfacePtr& graph,
                    const DistanceProviderForGraph& distance_provider,
                    Allocator* allocator,
                    float alpha,
                    Vector<InnerIdType>& selected_neighbors_out);

/**
 * @brief Phase B of edge construction: add cur_c to each selected neighbour's list.
 *
 * For every neighbour in @p selected_neighbors, takes that neighbour's lock, merges cur_c
 * into its list, and re-runs the pruning heuristic when the list is already full. This is
 * the conflicting half of mutually_connect_new_element: it is the only part that needs a
 * lock, which is why it is separated out — a batched build can group edges by target
 * neighbour and call this once per neighbour instead of once per edge.
 *
 * @param cur_c The element being linked back from.
 * @param selected_neighbors cur_c's outgoing neighbours, as produced by build_forward_links.
 * @param graph Graph interface for reading and writing neighbour connections.
 * @param distance_provider Supplies pairwise distances.
 * @param neighbors_mutexes Per-node locks protecting each neighbour's list.
 * @param allocator Allocator for temporary storage.
 * @param alpha Diversity parameter for the heuristic.
 */
void
link_back_edges(InnerIdType cur_c,
                const Vector<InnerIdType>& selected_neighbors,
                const GraphInterfacePtr& graph,
                const DistanceProviderForGraph& distance_provider,
                const MutexArrayPtr& neighbors_mutexes,
                Allocator* allocator,
                float alpha);

InnerIdType
mutually_connect_new_element(InnerIdType cur_c,
                             const DistHeapPtr& top_candidates,
                             const GraphInterfacePtr& graph,
                             const DistanceProviderForGraph& distance_provider,
                             const MutexArrayPtr& neighbors_mutexes,
                             Allocator* allocator,
                             float alpha = 1.0F);

InnerIdType
mutually_connect_new_element(InnerIdType cur_c,
                             const DistHeapPtr& top_candidates,
                             const GraphInterfacePtr& graph,
                             const FlattenInterfacePtr& flatten,
                             const MutexArrayPtr& neighbors_mutexes,
                             Allocator* allocator,
                             float alpha = 1.0F);

}  // namespace vsag
