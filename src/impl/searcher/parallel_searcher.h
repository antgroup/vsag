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

#include "attr/executor/executor.h"
#include "basic_searcher.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "impl/heap/distance_heap.h"
#include "index_common_param.h"
#include "utils/lock_strategy.h"
#include "utils/visited_list.h"

namespace vsag {

/**
 * @file parallel_searcher.h
 * @brief Parallel graph-based search implementation using thread pool.
 */

/**
 * @brief Multi-threaded graph searcher for parallel nearest neighbor search.
 *
 * ParallelSearcher performs graph-based search using multiple threads to
 * accelerate the search process. It supports concurrent neighbor exploration
 * and distance computation.
 */
class ParallelSearcher {
public:
    /**
     * @brief Constructs a ParallelSearcher with the given parameters.
     *
     * @param common_param Common index parameters including allocator and metrics.
     * @param search_pool Thread pool for parallel execution.
     * @param mutex_array Optional mutex array for thread-safe graph access.
     */
    explicit ParallelSearcher(const IndexCommonParam& common_param,
                              std::shared_ptr<SafeThreadPool> search_pool,
                              MutexArrayPtr mutex_array = nullptr);

    /**
     * @brief Performs graph-based search for nearest neighbors.
     *
     * @param graph Graph interface for neighbor traversal.
     * @param flatten Flatten interface for vector data access.
     * @param vl Visited list for tracking explored nodes.
     * @param query Query vector pointer.
     * @param inner_search_param Search parameters including topk and ef.
     * @param label_table Optional label table for label filtering.
     * @param ctx Optional query context for attribute filtering.
     * @return DistHeapPtr Heap containing search results.
     */
    virtual DistHeapPtr
    Search(const GraphInterfacePtr& graph,
           const FlattenInterfacePtr& flatten,
           const VisitedListPtr& vl,
           const void* query,
           const InnerSearchParam& inner_search_param,
           const LabelTablePtr& label_table = nullptr,
           QueryContext* ctx = nullptr) const;

    /**
     * @brief Sets the mutex array for thread-safe graph access.
     *
     * @param new_mutex_array New mutex array to use.
     */
    void
    SetMutexArray(MutexArrayPtr new_mutex_array);

private:
    /**
     * @brief Visits neighbors and collects candidates for parallel exploration.
     *
     * @param graph Graph interface for neighbor retrieval.
     * @param vl Visited list for tracking explored nodes.
     * @param node_pair Current node with distance.
     * @param filter Optional filter for candidate pruning.
     * @param skip_ratio Ratio for skipping candidates.
     * @param to_be_visited_rid Output vector for neighbor ranks to visit.
     * @param to_be_visited_id Output vector for neighbor IDs to visit.
     * @param neighbors Output vectors for neighbor lists per thread.
     * @param point_visited_num Number of points already visited.
     * @return Number of candidates added to visit lists.
     */
    uint32_t
    visit(const GraphInterfacePtr& graph,
          const VisitedListPtr& vl,
          const Vector<std::pair<float, uint64_t>>& node_pair,
          const FilterPtr& filter,
          float skip_ratio,
          Vector<InnerIdType>& to_be_visited_rid,
          Vector<InnerIdType>& to_be_visited_id,
          std::vector<Vector<InnerIdType>>& neighbors,
          uint64_t point_visited_num) const;

    /**
     * @brief Internal implementation of parallel graph search.
     *
     * @tparam mode Search mode (KNN_SEARCH or RANGE_SEARCH).
     * @param graph Graph interface for neighbor traversal.
     * @param flatten Flatten interface for vector data access.
     * @param vl Visited list for tracking explored nodes.
     * @param query Query vector pointer.
     * @param inner_search_param Search parameters including topk and ef.
     * @param label_table Optional label table for label filtering.
     * @param ctx Optional query context for attribute filtering.
     * @return DistHeapPtr Heap containing search results.
     */
    template <InnerSearchMode mode = KNN_SEARCH>
    DistHeapPtr
    search_impl(const GraphInterfacePtr& graph,
                const FlattenInterfacePtr& flatten,
                const VisitedListPtr& vl,
                const void* query,
                const InnerSearchParam& inner_search_param,
                const LabelTablePtr& label_table = nullptr,
                QueryContext* ctx = nullptr) const;

private:
    /// Allocator for memory management.
    Allocator* allocator_{nullptr};

    /// Thread pool for parallel execution.
    std::shared_ptr<SafeThreadPool> pool{nullptr};

    /// Mutex array for thread-safe graph access.
    MutexArrayPtr mutex_array_{nullptr};

    /// Prefetch stride for visit operations.
    uint32_t prefetch_stride_visit_{3};
};

using ParallelSearcherPtr = std::shared_ptr<ParallelSearcher>;

}  // namespace vsag