// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "graph_tail_search_parameters.h"
#include "impl/heap/distance_heap.h"
#include "impl/inner_search_param.h"
#include "query_context.h"
#include "utils/lock_strategy.h"
#include "utils/visited_list.h"

namespace vsag {
// Filter-code traversal. The enclosing index retains its existing full-code reorder path.
DistHeapPtr
SearchBatchedGraph(const GraphInterfacePtr& graph,
                   const FlattenInterfacePtr& flatten,
                   const VisitedListPtr& visited,
                   const ComputerInterfacePtr& computer,
                   const InnerSearchParam& parameters,
                   Allocator* allocator,
                   const MutexArrayPtr& mutexes,
                   QueryContext* context,
                   DistanceRecordVector* lower_bound_candidates,
                   const GraphTailSearchParameters& tail,
                   uint64_t requested_topk);
}  // namespace vsag
