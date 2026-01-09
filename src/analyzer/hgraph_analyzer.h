
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

class HGraphAnalyzer : public AnalyzerBase {
public:
    HGraphAnalyzer(HGraph* hgraph, const AnalyzerParam& param)
        : hgraph_(hgraph),
          base_ground_truth_(hgraph->allocator_),
          base_sample_ids_(hgraph->allocator_),
          base_sample_datas_(hgraph->allocator_),
          base_search_result_(hgraph->allocator_),
          is_duplicate_ids_(hgraph->allocator_),
          query_ground_truth_(hgraph->allocator_),
          query_sample_ids_(hgraph->allocator_),
          query_sample_datas_(hgraph->allocator_),
          query_search_result_(hgraph->allocator_),
          AnalyzerBase(hgraph->allocator_, hgraph->total_count_) {
        this->dim_ = hgraph_->dim_;
        this->topk_ = param.topk;
        this->base_sample_size_ = param.base_sample_size;
        this->search_params_ = param.search_params;
    }

    JsonType
    AnalyzeIndexBySearch(const vsag::SearchRequest& request) override;

    Vector<int64_t>
    GetComponentCount();

    float
    GetBaseAvgDistance();

    float
    GetNeighborRecall();

    std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, float>
    GetDegreeDistribution();

    float
    GetBaseSearchRecall(const std::string& search_param);

    float
    GetDuplicateRatio();

    float
    GetQuantizationError(const std::string& search_param);

    float
    GetQuantizationInversionRatio(const std::string& search_param);

    bool
    SetQuery(const DatasetPtr& query);

    float
    GetQueryQuantizationError(const std::string& search_param);

    float
    GetQueryQuantizationInversionRatio(const std::string& search_param);

    float
    GetQueryAvgDistance();

    float
    GetQuerySearchRecall(const std::string& search_param);

    float
    GetQuerySearchTimeCost(const std::string& search_param);

    float
    GetBaseSearchTimeCost(const std::string& search_param);

    //        float GetQueryGroundTruthInDegree();

    JsonType
    GetStats() override;

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
                                  uint32_t sample_size);

    void
    calculate_groundtruth(const Vector<float>& sample_datas,
                          const Vector<InnerIdType>& sample_ids,
                          UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
                          uint32_t sample_siz);

    float
    calculate_search_result(const Vector<float>& sample_datas,
                            const Vector<InnerIdType>& sample_ids,
                            UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
                            const std::string& search_param,
                            uint32_t sample_size);

    static float
    get_avg_distance(const Vector<InnerIdType>& sample_ids,
                     const UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth) {
        float dist_sum = 0.0F;
        uint32_t dist_count = 0;
        for (const auto& id : sample_ids) {
            const auto& result = ground_truth.at(id);
            const auto* data = result->GetData();
            for (uint32_t i = 0; i < result->Size(); ++i) {
                dist_sum += data[i].first;
                dist_count++;
            }
        }
        return dist_sum / static_cast<float>(dist_count);
    }

    float
    get_search_recall(uint32_t sample_size,
                      const Vector<InnerIdType>& sample_ids,
                      const UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
                      const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result);

private:
    HGraph* hgraph_;

    uint32_t base_sample_size_{10};
    Vector<InnerIdType> base_sample_ids_;
    Vector<float> base_sample_datas_;
    UnorderedMap<InnerIdType, DistHeapPtr> base_ground_truth_;
    UnorderedMap<InnerIdType, Vector<LabelType>> base_search_result_;
    Vector<bool> is_duplicate_ids_;
    float base_search_time_ms_{0.0F};

    uint32_t query_sample_size_{0};
    Vector<InnerIdType> query_sample_ids_;
    Vector<float> query_sample_datas_;
    UnorderedMap<InnerIdType, DistHeapPtr> query_ground_truth_;
    UnorderedMap<InnerIdType, Vector<LabelType>> query_search_result_;
    float query_search_time_ms_{0.0F};

    uint32_t topk_{100};
    std::string search_params_;
};

}  // namespace vsag
