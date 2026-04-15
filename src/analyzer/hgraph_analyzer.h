
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

/**
 * @brief Analyzer for HGraph (Hierarchical Navigable Small World Graph) indexes.
 *
 * This class provides comprehensive analysis capabilities for HGraph indexes,
 * including graph structure analysis, recall evaluation, quantization error
 * measurement, and duplicate detection.
 */
class HGraphAnalyzer : public AnalyzerBase {
public:
    /**
     * @brief Constructs an HGraph analyzer with the given HGraph index and parameters.
     *
     * @param hgraph Pointer to the HGraph index to be analyzed.
     * @param param Analyzer parameters for configuration.
     */
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

    /**
     * @brief Analyzes the HGraph index by performing searches.
     *
     * @param request The search request parameters.
     * @return JsonType containing analysis results.
     */
    JsonType
    AnalyzeIndexBySearch(const vsag::SearchRequest& request) override;

    /**
     * @brief Gets the count of connected components in the graph.
     *
     * @return Vector of component sizes.
     */
    Vector<int64_t>
    GetComponentCount();

    /**
     * @brief Calculates the average distance in the base dataset.
     *
     * @return Average distance value.
     */
    float
    GetBaseAvgDistance();

    /**
     * @brief Calculates the neighbor recall metric.
     *
     * @return Neighbor recall value.
     */
    float
    GetNeighborRecall();

    /**
     * @brief Gets the degree distribution statistics of the graph.
     *
     * @return Tuple containing degree distribution data and statistics.
     */
    std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, float>
    GetDegreeDistribution();

    /**
     * @brief Calculates search recall for the base dataset.
     *
     * @param search_param Search parameters string.
     * @return Search recall value.
     */
    float
    GetBaseSearchRecall(const std::string& search_param);

    /**
     * @brief Calculates the duplicate ratio in the index.
     *
     * @return Duplicate ratio value.
     */
    float
    GetDuplicateRatio();

    /**
     * @brief Calculates quantization error for the given search parameters.
     *
     * @param search_param Search parameters string.
     * @return Quantization error value.
     */
    float
    GetQuantizationError(const std::string& search_param);

    /**
     * @brief Calculates quantization inversion ratio.
     *
     * @param search_param Search parameters string.
     * @return Quantization inversion ratio value.
     */
    float
    GetQuantizationInversionRatio(const std::string& search_param);

    /**
     * @brief Sets query dataset for analysis.
     *
     * @param query Dataset pointer containing query vectors.
     * @return True if successful, false otherwise.
     */
    bool
    SetQuery(const DatasetPtr& query);

    /**
     * @brief Calculates quantization error for query vectors.
     *
     * @param search_param Search parameters string.
     * @return Query quantization error value.
     */
    float
    GetQueryQuantizationError(const std::string& search_param);

    /**
     * @brief Calculates quantization inversion ratio for query vectors.
     *
     * @param search_param Search parameters string.
     * @return Query quantization inversion ratio value.
     */
    float
    GetQueryQuantizationInversionRatio(const std::string& search_param);

    /**
     * @brief Calculates average distance for query vectors.
     *
     * @return Average query distance value.
     */
    float
    GetQueryAvgDistance();

    /**
     * @brief Calculates search recall for query vectors.
     *
     * @param search_param Search parameters string.
     * @return Query search recall value.
     */
    float
    GetQuerySearchRecall(const std::string& search_param);

    /**
     * @brief Calculates search time cost for query vectors.
     *
     * @param search_param Search parameters string.
     * @return Query search time in milliseconds.
     */
    float
    GetQuerySearchTimeCost(const std::string& search_param);

    /**
     * @brief Calculates search time cost for base dataset.
     *
     * @param search_param Search parameters string.
     * @return Base search time in milliseconds.
     */
    float
    GetBaseSearchTimeCost(const std::string& search_param);

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
    /// Pointer to the HGraph index being analyzed.
    HGraph* hgraph_;

    /// Number of base vectors sampled for analysis.
    uint32_t base_sample_size_{10};
    /// IDs of sampled base vectors.
    Vector<InnerIdType> base_sample_ids_;
    /// Data of sampled base vectors.
    Vector<float> base_sample_datas_;
    /// Ground truth distances for base samples.
    UnorderedMap<InnerIdType, DistHeapPtr> base_ground_truth_;
    /// Search results for base samples.
    UnorderedMap<InnerIdType, Vector<LabelType>> base_search_result_;
    /// Flags indicating duplicate IDs in base dataset.
    Vector<bool> is_duplicate_ids_;
    /// Time cost for base search operations (milliseconds).
    float base_search_time_ms_{0.0F};

    /// Number of query vectors sampled for analysis.
    uint32_t query_sample_size_{0};
    /// IDs of sampled query vectors.
    Vector<InnerIdType> query_sample_ids_;
    /// Data of sampled query vectors.
    Vector<float> query_sample_datas_;
    /// Ground truth distances for query samples.
    UnorderedMap<InnerIdType, DistHeapPtr> query_ground_truth_;
    /// Search results for query samples.
    UnorderedMap<InnerIdType, Vector<LabelType>> query_search_result_;
    /// Time cost for query search operations (milliseconds).
    float query_search_time_ms_{0.0F};

    /// Number of top results to consider.
    uint32_t topk_{100};
    /// Search parameters string for analysis queries.
    std::string search_params_;
};

}  // namespace vsag
