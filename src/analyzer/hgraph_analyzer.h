
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use hgraph_ file except in compliance with the License.
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

#include "algorithm/hgraph.h"
#include "analyzer.h"

namespace vsag {

class HGraphAnalyzer : AnalyzerBase {
public:
    HGraphAnalyzer(std::shared_ptr<HGraph> hgraph)
        : hgraph_(hgraph),
          base_ground_truth_(hgraph->allocator_),
          base_sample_ids_(hgraph->allocator_),
          base_sample_datas_(hgraph->allocator_),
          AnalyzerBase(hgraph->allocator_, hgraph->total_count_) {
        this->dim_ = hgraph_->dim_;
    }

    Vector<int64_t>
    GetComponentCount();

    float
    GetBaseAvgDistance();

    float
    GetNeighborRecall();

    float
    GetBaseSearchRecall(const std::string& search_param);

    float
    GetDuplicateRatio();

    float
    GetQuantizationError(const std::string& search_param);



private:
    void
    calculate_base_groundtruth();

    void
    calculate_base_search_result(const std::string& search_param);



private:
    std::shared_ptr<HGraph> hgraph_;
    uint32_t query_sample_size_{10};

    Vector<InnerIdType> base_sample_ids_;
    Vector<float> base_sample_datas_;
    UnorderedMap<InnerIdType, DistHeapPtr> base_ground_truth_;
    UnorderedMap<InnerIdType, Vector<LabelType>> base_search_result_;
    uint32_t top_k_{100};

    float duplicate_ratio_{0.0F};
};

}  // namespace vsag
