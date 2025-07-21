
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
#include "lock_strategy.h"
#include "utils/visited_list.h"

namespace vsag {

static constexpr uint32_t OPTIMIZE_SEARCHER_SAMPLE_SIZE = 10000;

enum InnerSearchMode { KNN_SEARCH = 1, RANGE_SEARCH = 2 };

class InnerSearchParam {
public:
    int64_t topk{0};
    float radius{0.0f};
    InnerIdType ep{0};
    uint64_t ef{10};
    FilterPtr is_inner_id_allowed{nullptr};
    float skip_ratio{0.8F};
    InnerSearchMode search_mode{KNN_SEARCH};
    int range_search_limit_size{-1};
    int64_t parallel_search_thread_count{1};

    // for ivf
    int scan_bucket_size{1};
    float factor{2.0F};
    float first_order_scan_ratio{1.0F};
    Allocator* search_alloc{nullptr};
    ExecutorPtr executor{nullptr};

    InnerSearchParam&
    operator=(const InnerSearchParam& other) {
        if (this != &other) {
            topk = other.topk;
            radius = other.radius;
            ep = other.ep;
            ef = other.ef;
            skip_ratio = other.skip_ratio;
            search_mode = other.search_mode;
            range_search_limit_size = other.range_search_limit_size;
            is_inner_id_allowed = other.is_inner_id_allowed;
            scan_bucket_size = other.scan_bucket_size;
            factor = other.factor;
            first_order_scan_ratio = other.first_order_scan_ratio;
        }
        return *this;
    }
};

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
           const InnerSearchParam& inner_search_param) const;

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
                const InnerSearchParam& inner_search_param) const;

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
