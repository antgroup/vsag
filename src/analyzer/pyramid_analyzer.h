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

#include "algorithm/pyramid.h"
#include "analyzer.h"

namespace vsag {

/**
 * @brief Analyzer for Pyramid indexes with hierarchical structure.
 *
 * This class provides analysis capabilities specifically designed for Pyramid indexes,
 * including sub-index quality analysis, graph connectivity metrics, and recall evaluation.
 */
class PyramidAnalyzer : public AnalyzerBase {
public:
    /**
     * @brief Constructs a Pyramid analyzer with the given Pyramid index and parameters.
     *
     * @param pyramid Pointer to the Pyramid index to be analyzed.
     * @param param Analyzer parameters for configuration.
     */
    PyramidAnalyzer(Pyramid* pyramid, const AnalyzerParam& param)
        : pyramid_(pyramid),
          sample_ids_(pyramid->allocator_),
          sample_datas_(pyramid->allocator_),
          subindex_stats_(pyramid->allocator_),
          ground_truth_(pyramid->allocator_),
          search_result_(pyramid->allocator_),
          low_recall_nodes_(pyramid->allocator_),
          AnalyzerBase(pyramid->allocator_, pyramid->GetNumElements()) {
        this->dim_ = pyramid_->dim_;
        this->topk_ = param.topk;
        this->sample_size_ = param.base_sample_size;
        this->search_params_ = param.search_params;
    }

    /**
     * @brief Gets statistical information about the Pyramid index.
     *
     * @return JsonType containing various statistics and metrics.
     */
    JsonType
    GetStats() override;

    /**
     * @brief Analyzes the Pyramid index by performing searches.
     *
     * @param request The search request parameters.
     * @return JsonType containing analysis results.
     */
    JsonType
    AnalyzeIndexBySearch(const SearchRequest& request) override;

    /**
     * @brief Calculates the duplicate ratio in the Pyramid index.
     *
     * @return Duplicate ratio value.
     */
    float
    GetDuplicateRatio();

private:
    /**
     * @brief Statistics for a single sub-index within the Pyramid.
     */
    struct SubIndexStats {
        SubIndexStats(Allocator* allocator) : ids(allocator) {
        }
        /// Path identifier for the sub-index.
        std::string path;
        /// Number of elements in the sub-index.
        uint32_t size{0};
        /// Recall metric for the sub-index.
        float recall{0.0F};
        /// Flag indicating if the sub-index has quality issues.
        bool is_problematic{false};
        /// Status type of the index node.
        IndexNode::Status status{IndexNode::Status::FLAT};
        /// IDs of elements in the sub-index.
        Vector<InnerIdType> ids;
    };

    /**
     * @brief Information about nodes with low recall scores.
     */
    struct LowRecallNodeInfo {
        /// Path identifier for the node.
        std::string path;
        /// Number of elements in the node.
        uint32_t size;
        /// Recall score for the node.
        float recall;
        /// Duplicate ratio within the node.
        float duplicate_ratio{0.0F};
        /// Flag indicating if entry point is duplicated.
        bool entry_point_duplicate{false};
        /// Size of the duplicate group at entry point.
        uint32_t entry_point_group_size{0};
    };

    /**
     * @brief Analysis results for graph quality metrics.
     */
    struct GraphQualityAnalysis {
        /// Average degree of nodes in the graph.
        float avg_degree{0.0F};
        /// Count of nodes with zero outgoing edges.
        uint32_t zero_out_degree_count{0};
        /// Count of nodes with zero incoming edges.
        uint32_t zero_in_degree_count{0};
        /// Maximum outgoing degree.
        uint32_t max_out_degree{0};
        /// Maximum incoming degree.
        uint32_t max_in_degree{0};
        /// Neighbor recall metric.
        float neighbor_recall{0.0F};
        /// Count of connected components.
        uint32_t component_count{0};
        /// Size of the largest connected component.
        uint32_t max_component_size{0};
        /// Count of singleton nodes (isolated nodes).
        uint32_t singleton_count{0};
        /// Connectivity ratio of the graph.
        float connectivity_ratio{0.0F};
        /// Entry point ID for the graph.
        InnerIdType entry_point{0};
        /// Maximum distance from entry point.
        uint32_t max_distance_from_entry{0};
        /// Count of unreachable nodes from entry point.
        uint32_t unreachable_count{0};
        /// Average distance from entry point.
        float avg_distance_from_entry{0.0F};
    };

    JsonType
    get_index_node_structure();

    JsonType
    get_leaf_node_size_distribution();

    void
    collect_leaf_sizes(IndexNode* node, Vector<uint32_t>& sizes);

    JsonType
    get_subindex_quality();

    void
    analyze_subindexes(IndexNode* node, const std::string& path);

    static void
    collect_subindex_ids(IndexNode* node, Vector<InnerIdType>& ids);

    float
    calculate_subindex_recall(const Vector<InnerIdType>& subindex_ids);

    float
    calculate_weighted_recall();

    Vector<InnerIdType>
    collect_node_ids(const IndexNode* node);

    DistHeapPtr
    calculate_node_groundtruth(const IndexNode* node,
                               const float* query,
                               const Vector<InnerIdType>& node_ids);

    DistHeapPtr
    search_single_node(const IndexNode* node,
                       const float* query,
                       const std::string& search_param_str);

    float
    calculate_node_recall(const IndexNode* node,
                          const float* queries,
                          uint32_t query_count,
                          const std::string& search_param_str);

    JsonType
    analyze_node_graph_quality(const IndexNode* node,
                               const std::string& path,
                               float recall,
                               const Vector<InnerIdType>& node_ids);

    GraphQualityAnalysis
    get_node_degree_distribution(const IndexNode* node, const Vector<InnerIdType>& node_ids);

    float
    get_node_neighbor_recall(const IndexNode* node, const Vector<InnerIdType>& node_ids);

    GraphQualityAnalysis
    get_node_connectivity(const IndexNode* node, const Vector<InnerIdType>& node_ids);

    GraphQualityAnalysis
    analyze_entry_point(const IndexNode* node, const Vector<InnerIdType>& node_ids);

    JsonType
    get_graph_node_recall_stats(const std::string& search_param_str);

    void
    sample_global();

    void
    calculate_groundtruth(const Vector<float>& sample_datas,
                          const Vector<InnerIdType>& sample_ids,
                          UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
                          uint32_t sample_size);

    float
    get_avg_distance();

    float
    get_quantization_error(const std::string& search_param);

    float
    get_quantization_inversion_ratio(const std::string& search_param);

    std::tuple<float, float>
    calculate_quantization_result(const Vector<float>& sample_datas,
                                  const Vector<InnerIdType>& sample_ids,
                                  const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
                                  uint32_t sample_size);

    float
    calculate_search_result(const Vector<float>& sample_datas,
                            const Vector<InnerIdType>& sample_ids,
                            UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
                            const std::string& search_param,
                            uint32_t sample_size);

    float
    get_search_recall(uint32_t sample_size,
                      const Vector<InnerIdType>& sample_ids,
                      const UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
                      const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result);

    static float
    get_avg_distance_from_groundtruth(const Vector<InnerIdType>& sample_ids,
                                      const UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth);

    float
    get_node_duplicate_ratio(const IndexNode* node, const Vector<InnerIdType>& node_ids);

    bool
    check_entry_point_duplicate(const IndexNode* node,
                                const Vector<InnerIdType>& node_ids,
                                uint32_t& duplicate_group_size);

    /// Pointer to the Pyramid index being analyzed.
    Pyramid* pyramid_;

    /// IDs of sampled vectors for analysis.
    Vector<InnerIdType> sample_ids_;
    /// Data of sampled vectors for analysis.
    Vector<float> sample_datas_;
    /// Number of vectors sampled for analysis.
    uint32_t sample_size_;
    /// Statistics for each sub-index in the Pyramid.
    Vector<SubIndexStats> subindex_stats_;

    /// Ground truth distances for sampled vectors.
    UnorderedMap<InnerIdType, DistHeapPtr> ground_truth_;
    /// Search results for sampled vectors.
    UnorderedMap<InnerIdType, Vector<LabelType>> search_result_;

    /// Number of top results to consider.
    uint32_t topk_{100};
    /// Search parameters string for analysis queries.
    std::string search_params_;
    /// Search time cost in milliseconds.
    float search_time_ms_{0.0F};
    /// Information about nodes with low recall scores.
    Vector<LowRecallNodeInfo> low_recall_nodes_;
};

}  // namespace vsag
