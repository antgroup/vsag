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

class PyramidAnalyzer : public AnalyzerBase {
public:
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

    JsonType
    GetStats() override;

    JsonType
    AnalyzeIndexBySearch(const SearchRequest& request) override;

    float
    GetDuplicateRatio();

private:
    struct SubIndexStats {
        SubIndexStats(Allocator* allocator) : ids(allocator) {
        }
        std::string path;
        uint32_t size{0};
        float recall{0.0F};
        bool is_problematic{false};
        IndexNode::Status status{IndexNode::Status::FLAT};
        Vector<InnerIdType> ids;
    };

    struct LowRecallNodeInfo {
        std::string path;
        uint32_t size;
        float recall;
        float duplicate_ratio{0.0F};
        bool entry_point_duplicate{false};
        uint32_t entry_point_group_size{0};
    };

    struct GraphQualityAnalysis {
        float avg_degree{0.0F};
        uint32_t zero_out_degree_count{0};
        uint32_t zero_in_degree_count{0};
        uint32_t max_out_degree{0};
        uint32_t max_in_degree{0};
        float neighbor_recall{0.0F};
        uint32_t component_count{0};
        uint32_t max_component_size{0};
        uint32_t singleton_count{0};
        float connectivity_ratio{0.0F};
        InnerIdType entry_point{0};
        uint32_t max_distance_from_entry{0};
        uint32_t unreachable_count{0};
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

    Pyramid* pyramid_;

    Vector<InnerIdType> sample_ids_;
    Vector<float> sample_datas_;
    uint32_t sample_size_;
    Vector<SubIndexStats> subindex_stats_;

    UnorderedMap<InnerIdType, DistHeapPtr> ground_truth_;
    UnorderedMap<InnerIdType, Vector<LabelType>> search_result_;

    uint32_t topk_{100};
    std::string search_params_;
    float search_time_ms_{0.0F};
    Vector<LowRecallNodeInfo> low_recall_nodes_;
};

}  // namespace vsag
