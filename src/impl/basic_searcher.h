
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

#include "../utils.h"
#include "algorithm/hnswlib/algorithm_interface.h"
#include "common.h"
#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "index/index_common_param.h"
#include "utils/visited_list.h"

namespace vsag {

class InnerSearchParam {
public:
    int64_t topk_{0};
    float radius_{-1.0f};
    InnerIdType ep_{0};
    uint64_t ef_{10};
    BaseFilterFunctor* is_id_allowed_{nullptr};
};

class BasicSearcher {
public:
    BasicSearcher(const IndexCommonParam& common_param);

    virtual MaxHeap
    Search(const GraphInterfacePtr& graph_data_cell,
           const FlattenInterfacePtr& vector_data_cell,
           const std::shared_ptr<VisitedList>& vl,
           const float* query,
           const InnerSearchParam& inner_search_param) const;

private:
    // rid means the neighbor's rank (e.g., the first neighbor's rid == 0)
    //  id means the neighbor's  id  (e.g., the first neighbor's  id == 12345)
    uint32_t
    visit(const GraphInterfacePtr& graph_data_cell,
          const std::shared_ptr<VisitedList>& vl,
          const std::pair<float, uint64_t>& current_node_pair,
          Vector<InnerIdType>& to_be_visited_rid,
          Vector<InnerIdType>& to_be_visited_id) const;

private:
    Allocator* allocator_{nullptr};

    uint32_t prefetch_jump_visit_size_{1};
};

}  // namespace vsag
