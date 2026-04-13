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

/// @file pruning_strategy.h
/// @brief Edge pruning strategies for graph-based vector indexes.

#pragma once

#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER2(DistHeap, DistanceHeap);
DEFINE_POINTER(FlattenInterface);
DEFINE_POINTER(GraphInterface);
DEFINE_POINTER(MutexArray);

/// @brief Selects edges using a heuristic algorithm for graph-based indexes.
///
/// This function implements a heuristic edge selection algorithm that chooses
/// the best neighbors while maintaining diversity in the edge set.
///
/// @param edges Distance heap containing candidate edges.
/// @param max_size Maximum number of edges to select.
/// @param flatten Flatten interface for distance computation.
/// @param allocator Allocator for memory management.
/// @param alpha Alpha parameter for controlling edge selection diversity.
void
select_edges_by_heuristic(const DistHeapPtr& edges,
                          uint64_t max_size,
                          const FlattenInterfacePtr& flatten,
                          Allocator* allocator,
                          float alpha = 1.0F);

/// @brief Connects a new element to the graph with mutual edge selection.
///
/// This function adds a new element to the graph by selecting its neighbors
/// and updating the reverse connections. It uses the heuristic selection
/// algorithm to ensure high-quality edges.
///
/// @param cur_c ID of the new element to connect.
/// @param top_candidates Heap of candidate neighbors.
/// @param graph Graph interface for edge management.
/// @param flatten Flatten interface for distance computation.
/// @param neighbors_mutexes Mutex array for thread-safe edge updates.
/// @param allocator Allocator for memory management.
/// @param alpha Alpha parameter for edge selection diversity.
/// @return ID of the selected entry point.
InnerIdType
mutually_connect_new_element(InnerIdType cur_c,
                             const DistHeapPtr& top_candidates,
                             const GraphInterfacePtr& graph,
                             const FlattenInterfacePtr& flatten,
                             const MutexArrayPtr& neighbors_mutexes,
                             Allocator* allocator,
                             float alpha = 1.0F);

}  // namespace vsag