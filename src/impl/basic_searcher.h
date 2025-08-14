
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
#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "impl/heap/distance_heap.h"
#include "index/index_common_param.h"
#include "index/iterator_filter.h"
#include "inner_search_param.h"
#include "lock_strategy.h"
#include "utils/timer.h"
#include "utils/visited_list.h"

namespace vsag {

static constexpr uint32_t OPTIMIZE_SEARCHER_SAMPLE_SIZE = 10000;

constexpr float THRESHOLD_ERROR = 2e-6;

class BasicSearcher {
public:
    explicit BasicSearcher(const IndexCommonParam& common_param,
                           MutexArrayPtr mutex_array = nullptr);

    virtual DistHeapPtr
    Search(const GraphInterfacePtr& graph,
           const FlattenInterfacePtr& flatten,
           const VisitedListPtr& vl,
           const void* query,
           const InnerSearchParam& inner_search_param,
           const LabelTablePtr& label_table = nullptr) const;

    virtual DistHeapPtr
    Search(const GraphInterfacePtr& graph,
           const FlattenInterfacePtr& flatten,
           const VisitedListPtr& vl,
           const void* query,
           const InnerSearchParam& inner_search_param,
           IteratorFilterContext* iter_ctx) const;

    virtual bool
    SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params);

    virtual void
    SetMockParameters(const GraphInterfacePtr& graph,
                      const FlattenInterfacePtr& flatten,
                      const std::shared_ptr<VisitedListPool>& vl_pool,
                      const InnerSearchParam& inner_search_param,
                      const uint64_t dim,
                      const uint32_t n_trials = OPTIMIZE_SEARCHER_SAMPLE_SIZE);

    virtual double
    MockRun() const;

    void
    SetMutexArray(MutexArrayPtr new_mutex_array);

private:
    // rid means the neighbor's rank (e.g., the first neighbor's rid == 0)
    //  id means the neighbor's  id  (e.g., the first neighbor's  id == 12345)
    uint32_t
    visit(const GraphInterfacePtr& graph,
          const VisitedListPtr& vl,
          const std::pair<float, uint64_t>& current_node_pair,
          const FilterPtr& filter,
          float skip_ratio,
          Vector<InnerIdType>& to_be_visited_rid,
          Vector<InnerIdType>& to_be_visited_id,
          Vector<InnerIdType>& neighbors) const;

    template <InnerSearchMode mode = KNN_SEARCH>
    DistHeapPtr
    search_impl(const GraphInterfacePtr& graph,
                const FlattenInterfacePtr& flatten,
                const VisitedListPtr& vl,
                const void* query,
                const InnerSearchParam& inner_search_param,
                const LabelTablePtr& label_table) const;

    template <InnerSearchMode mode = KNN_SEARCH>
    DistHeapPtr
    search_impl(const GraphInterfacePtr& graph,
                const FlattenInterfacePtr& flatten,
                const VisitedListPtr& vl,
                const void* query,
                const InnerSearchParam& inner_search_param,
                IteratorFilterContext* iter_ctx) const;

private:
    Allocator* allocator_{nullptr};

    MutexArrayPtr mutex_array_{nullptr};

    // mock run parameters
    GraphInterfacePtr mock_graph_{nullptr};
    FlattenInterfacePtr mock_flatten_{nullptr};
    std::shared_ptr<VisitedListPool> mock_vl_pool_{nullptr};
    InnerSearchParam mock_inner_search_param_;
    uint64_t mock_dim_{0};
    uint32_t mock_n_trials_{1};

    // runtime parameters
    uint32_t prefetch_stride_visit_{3};
};

using BasicSearcherPtr = std::shared_ptr<BasicSearcher>;

}  // namespace vsag
