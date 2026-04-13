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

#include "algorithm/inner_index_interface.h"
#include "attr/executor/executor.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "impl/heap/distance_heap.h"
#include "impl/inner_search_param.h"
#include "index/iterator_filter.h"
#include "index_common_param.h"
#include "utils/lock_strategy.h"
#include "utils/pointer_define.h"
#include "utils/timer.h"
#include "utils/visited_list.h"

namespace vsag {

/// Sample size used for optimizer search.
static constexpr uint32_t OPTIMIZE_SEARCHER_SAMPLE_SIZE = 10000;

/// Error threshold for floating-point comparisons.
constexpr float THRESHOLD_ERROR = 2e-6;
DEFINE_POINTER(BasicSearcher);

/**
 * @file basic_searcher.h
 * @brief Basic graph-based search implementation.
 */

/**
 * @brief Standard graph-based searcher for nearest neighbor search.
 *
 * BasicSearcher performs graph-based search using a greedy traversal strategy.
 * It supports both standard KNN search and iterator-based filtering search,
 * with configurable runtime parameters for performance tuning.
 */
class BasicSearcher {
public:
    /**
     * @brief Constructs a BasicSearcher with the given parameters.
     *
     * @param common_param Common index parameters including allocator and metrics.
     * @param mutex_array Optional mutex array for thread-safe graph access.
     */
    explicit BasicSearcher(const IndexCommonParam& common_param,
                           MutexArrayPtr mutex_array = nullptr);

    /**
     * @brief Performs graph-based search with label filtering.
     *
     * @param graph Graph interface for neighbor traversal.
     * @param flatten Flatten interface for vector data access.
     * @param vl Visited list for tracking explored nodes.
     * @param query Query vector pointer.
     * @param inner_search_param Search parameters including topk and ef.
     * @param label_table Label table for label filtering.
     * @param ctx Query context for attribute filtering.
     * @return DistHeapPtr Heap containing search results.
     */
    virtual DistHeapPtr
    Search(const GraphInterfacePtr& graph,
           const FlattenInterfacePtr& flatten,
           const VisitedListPtr& vl,
           const void* query,
           const InnerSearchParam& inner_search_param,
           const LabelTablePtr& label_table,
           QueryContext* ctx) const;

    /**
     * @brief Performs graph-based search with iterator filter context.
     *
     * @param graph Graph interface for neighbor traversal.
     * @param flatten Flatten interface for vector data access.
     * @param vl Visited list for tracking explored nodes.
     * @param query Query vector pointer.
     * @param inner_search_param Search parameters including topk and ef.
     * @param iter_ctx Iterator filter context for progressive filtering.
     * @param ctx Query context for attribute filtering.
     * @return DistHeapPtr Heap containing search results.
     */
    virtual DistHeapPtr
    Search(const GraphInterfacePtr& graph,
           const FlattenInterfacePtr& flatten,
           const VisitedListPtr& vl,
           const void* query,
           const InnerSearchParam& inner_search_param,
           IteratorFilterContext* iter_ctx,
           QueryContext* ctx) const;

    /**
     * @brief Sets runtime parameters for search optimization.
     *
     * @param new_params Map of parameter names to values.
     * @return true if parameters were successfully set, false otherwise.
     */
    virtual bool
    SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params);

    /**
     * @brief Sets mock parameters for performance benchmarking.
     *
     * @param graph Mock graph interface for benchmarking.
     * @param flatten Mock flatten interface for benchmarking.
     * @param vl_pool Mock visited list pool for benchmarking.
     * @param inner_search_param Mock search parameters for benchmarking.
     * @param dim Vector dimension for benchmarking.
     * @param n_trials Number of trial runs for benchmarking.
     */
    virtual void
    SetMockParameters(const GraphInterfacePtr& graph,
                      const FlattenInterfacePtr& flatten,
                      const std::shared_ptr<VisitedListPool>& vl_pool,
                      const InnerSearchParam& inner_search_param,
                      const uint64_t dim,
                      const uint32_t n_trials = OPTIMIZE_SEARCHER_SAMPLE_SIZE);

    /**
     * @brief Runs a mock search for performance benchmarking.
     *
     * @param stats Output statistics from the mock run.
     * @return Execution time in seconds.
     */
    virtual double
    MockRun(SearchStatistics& stats) const;

    /**
     * @brief Sets the mutex array for thread-safe graph access.
     *
     * @param new_mutex_array New mutex array to use.
     */
    void
    SetMutexArray(MutexArrayPtr new_mutex_array);

private:
    /**
     * @brief Visits neighbors and collects candidates for exploration.
     *
     * @param graph Graph interface for neighbor retrieval.
     * @param vl Visited list for tracking explored nodes.
     * @param current_node_pair Current node with distance.
     * @param filter Optional filter for candidate pruning.
     * @param skip_ratio Ratio for skipping candidates.
     * @param to_be_visited_rid Output vector for neighbor ranks to visit.
     * @param to_be_visited_id Output vector for neighbor IDs to visit.
     * @param neighbors Output vector for neighbor list.
     * @return Number of candidates added to visit lists.
     */
    uint32_t
    visit(const GraphInterfacePtr& graph,
          const VisitedListPtr& vl,
          const std::pair<float, uint64_t>& current_node_pair,
          const FilterPtr& filter,
          float skip_ratio,
          Vector<InnerIdType>& to_be_visited_rid,
          Vector<InnerIdType>& to_be_visited_id,
          Vector<InnerIdType>& neighbors) const;

    /**
     * @brief Internal implementation of graph search with label table.
     *
     * @tparam mode Search mode (KNN_SEARCH or RANGE_SEARCH).
     * @param graph Graph interface for neighbor traversal.
     * @param flatten Flatten interface for vector data access.
     * @param vl Visited list for tracking explored nodes.
     * @param query Query vector pointer.
     * @param inner_search_param Search parameters including topk and ef.
     * @param label_table Label table for label filtering.
     * @param ctx Query context for attribute filtering.
     * @return DistHeapPtr Heap containing search results.
     */
    template <InnerSearchMode mode = KNN_SEARCH>
    DistHeapPtr
    search_impl(const GraphInterfacePtr& graph,
                const FlattenInterfacePtr& flatten,
                const VisitedListPtr& vl,
                const void* query,
                const InnerSearchParam& inner_search_param,
                const LabelTablePtr& label_table,
                QueryContext* ctx) const;

    /**
     * @brief Internal implementation of graph search with iterator context.
     *
     * @tparam mode Search mode (KNN_SEARCH or RANGE_SEARCH).
     * @param graph Graph interface for neighbor traversal.
     * @param flatten Flatten interface for vector data access.
     * @param vl Visited list for tracking explored nodes.
     * @param query Query vector pointer.
     * @param inner_search_param Search parameters including topk and ef.
     * @param iter_ctx Iterator filter context for progressive filtering.
     * @param ctx Query context for attribute filtering.
     * @return DistHeapPtr Heap containing search results.
     */
    template <InnerSearchMode mode = KNN_SEARCH>
    DistHeapPtr
    search_impl(const GraphInterfacePtr& graph,
                const FlattenInterfacePtr& flatten,
                const VisitedListPtr& vl,
                const void* query,
                const InnerSearchParam& inner_search_param,
                IteratorFilterContext* iter_ctx,
                QueryContext* ctx) const;

private:
    /// Allocator for memory management.
    Allocator* allocator_{nullptr};

    /// Mutex array for thread-safe graph access.
    MutexArrayPtr mutex_array_{nullptr};

    /// Mock graph interface for benchmarking.
    GraphInterfacePtr mock_graph_{nullptr};

    /// Mock flatten interface for benchmarking.
    FlattenInterfacePtr mock_flatten_{nullptr};

    /// Mock visited list pool for benchmarking.
    std::shared_ptr<VisitedListPool> mock_vl_pool_{nullptr};

    /// Mock search parameters for benchmarking.
    InnerSearchParam mock_inner_search_param_;

    /// Mock dimension for benchmarking.
    uint64_t mock_dim_{0};

    /// Number of trials for mock benchmarking.
    uint32_t mock_n_trials_{1};

    /// Prefetch stride for visit operations.
    uint32_t prefetch_stride_visit_{3};
};
}  // namespace vsag