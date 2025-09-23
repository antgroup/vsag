
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

class ParallelSearcher {
public:
    explicit ParallelSearcher(const IndexCommonParam& common_param,
                              std::shared_ptr<SafeThreadPool> search_pool,
                              MutexArrayPtr mutex_array = nullptr);

    virtual DistHeapPtr
    Search(const GraphInterfacePtr& graph,
           const FlattenInterfacePtr& flatten,
           const VisitedListPtr& vl,
           const void* query,
           const InnerSearchParam& inner_search_param,
           const LabelTablePtr& label_table = nullptr) const;

    void
    SetMutexArray(MutexArrayPtr new_mutex_array);

private:
    // rid means the neighbor's rank (e.g., the first neighbor's rid == 0)
    //  id means the neighbor's  id  (e.g., the first neighbor's  id == 12345)
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

    template <InnerSearchMode mode = KNN_SEARCH>
    DistHeapPtr
    search_impl(const GraphInterfacePtr& graph,
                const FlattenInterfacePtr& flatten,
                const VisitedListPtr& vl,
                const void* query,
                const InnerSearchParam& inner_search_param,
                const LabelTablePtr& label_table = nullptr) const;

private:
    Allocator* allocator_{nullptr};

    std::shared_ptr<SafeThreadPool> pool{nullptr};

    MutexArrayPtr mutex_array_{nullptr};

    // runtime parameters
    uint32_t prefetch_stride_visit_{3};
};

using ParallelSearcherPtr = std::shared_ptr<ParallelSearcher>;

}  // namespace vsag
