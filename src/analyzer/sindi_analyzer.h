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

#include <vector>

#include "algorithm/sindi/sindi.h"
#include "analyzer.h"

namespace vsag {

class SINDIAnalyzer : public AnalyzerBase {
public:
    SINDIAnalyzer(SINDI* sindi, const AnalyzerParam& param)
        : AnalyzerBase(sindi->allocator_, static_cast<uint32_t>(sindi->GetNumElements())),
          sindi_(sindi),
          topk_(param.topk),
          base_sample_size_(param.base_sample_size),
          search_params_(param.search_params) {
    }

    JsonType
    GetStats() override;

    JsonType
    AnalyzeIndexBySearch(const SearchRequest& request) override;

private:
    void
    get_pruned_sparse_vector_by_inner_id(InnerIdType inner_id,
                                         SparseVector* data,
                                         Allocator* specified_allocator) const;

    std::vector<std::pair<float, int64_t>>
    collect_coarse_candidates(const SparseVector& query,
                              const SINDISearchParameter& search_param,
                              int64_t candidate_count) const;

    std::vector<std::pair<float, int64_t>>
    collect_doc_prune_candidates(const SparseVector& query,
                                 const SINDISearchParameter& search_param,
                                 int64_t candidate_count) const;

    std::vector<std::pair<float, int64_t>>
    rerank_candidates(const SparseVector& query,
                      const std::vector<std::pair<float, int64_t>>& coarse_candidates,
                      int64_t topk) const;

    float
    calculate_exact_distance_by_label(const SparseVector& query,
                                      int64_t label,
                                      const DatasetPtr& base_dataset = nullptr) const;

    bool
    get_original_sparse_vector_by_inner_id(InnerIdType inner_id,
                                           SparseVector* data,
                                           Allocator* specified_allocator,
                                           const DatasetPtr& base_dataset = nullptr) const;

    float
    calculate_pruned_distance_by_label(const SparseVector& query, int64_t label) const;

    DatasetPtr
    calculate_ground_truth(const DatasetPtr& query_dataset,
                           int64_t topk,
                           const DatasetPtr& base_dataset = nullptr) const;

    JsonType
    get_active_term_count_stats() const;

    JsonType
    get_posting_length_distribution_stats() const;

    JsonType
    get_quantization_range_stats() const;

    JsonType
    get_mean_doc_retained(const DatasetPtr& base_dataset = nullptr) const;

    JsonType
    calculate_distance_quality_stats(const DatasetPtr& query_dataset,
                                     const SINDISearchParameter& search_param,
                                     const DatasetPtr& base_dataset,
                                     int64_t topk) const;

    JsonType
    calculate_recall_stats(const DatasetPtr& query_dataset,
                           const DatasetPtr& ground_truth,
                           const SINDISearchParameter& search_param,
                           int64_t topk) const;

    JsonType
    calculate_postings_scanned_stats(const DatasetPtr& query_dataset,
                                     const SINDISearchParameter& search_param) const;

    JsonType
    get_base_search_stats(const std::string& search_param,
                          const DatasetPtr& base_dataset = nullptr) const;

private:
    SINDI* sindi_{nullptr};
    int64_t topk_{10};
    uint64_t base_sample_size_{10};
    std::string search_params_;
};

}  // namespace vsag
