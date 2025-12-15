
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
          base_search_result_(hgraph->allocator_),
          query_ground_truth_(hgraph->allocator_),
          query_sample_ids_(hgraph->allocator_),
          query_sample_datas_(hgraph->allocator_),
          query_search_result_(hgraph->allocator_),
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

    float
    GetQuantizationInversionRatio(const std::string& search_param);

    bool SetQuery(const DatasetPtr& query);


    float GetQueryQuantizationError(const std::string& search_param);

        float GetQueryQuantizationInversionRatio(const std::string& search_param);

        float
        GetQueryAvgDistance();





private:
    void
    calculate_base_groundtruth();

    void
    calculate_query_groundtruth();

    void
    calculate_base_search_result(const std::string& search_param);

    void
    calculate_query_search_result(const std::string& search_param);

    std::tuple<float, float>
    calculate_quantization_result(const Vector<float>& sample_datas,
                                 const Vector<InnerIdType>& sample_ids,
                                 const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
                                 int sample_size);

    void calculate_groundtruth(const Vector<float>& sample_datas,
                          const Vector<InnerIdType>& sample_ids,
                          UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
                          int sample_siz);

        void calculate_search_result(const Vector<float>& sample_datas,
                                         const Vector<InnerIdType>& sample_ids,
                                         UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
                                         const std::string& search_param,
                                         int sample_size);

        float get_avg_distance(Vector<InnerIdType> sample_ids, UnorderedMap<InnerIdType, DistHeapPtr> ground_truth);



private:
    std::shared_ptr<HGraph> hgraph_;

    uint32_t base_sample_size_{10};
    Vector<InnerIdType> base_sample_ids_;
    Vector<float> base_sample_datas_;
    UnorderedMap<InnerIdType, DistHeapPtr> base_ground_truth_;
    UnorderedMap<InnerIdType, Vector<LabelType>> base_search_result_;

    uint32_t query_sample_size_{0};
    Vector<InnerIdType> query_sample_ids_;
    Vector<float> query_sample_datas_;
    UnorderedMap<InnerIdType, DistHeapPtr> query_ground_truth_;
    UnorderedMap<InnerIdType, Vector<LabelType>> query_search_result_;


    uint32_t top_k_{100};

    float duplicate_ratio_{0.0F};
    float quantization_error_{0.0F};
    float quantization_inversion_count_rate_{0.0F};
};

}  // namespace vsag
