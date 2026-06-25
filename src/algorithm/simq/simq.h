
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

#include <shared_mutex>

#include "algorithm/hgraph/hgraph.h"
#include "algorithm/hgraph/hgraph_parameter.h"
#include "algorithm/inner_index_interface.h"
#include "datacell/flatten_interface.h"
#include "index_common_param.h"
#include "simq_parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

class SIMQ : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit SIMQ(const SIMQParameterPtr& param, const IndexCommonParam& common_param);

    explicit SIMQ(const ParamPtr& param, const IndexCommonParam& common_param)
        : SIMQ(std::dynamic_pointer_cast<SIMQParameter>(param), common_param){};

    ~SIMQ() override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    std::vector<int64_t>
    Add(const DatasetPtr& data, AddMode mode = AddMode::DEFAULT) override;

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    IndexType
    GetIndexType() const override {
        return IndexType::SIMQ;
    }

    std::string
    GetName() const override {
        return INDEX_SIMQ;
    }

    int64_t
    GetNumElements() const override {
        return static_cast<int64_t>(total_count_);
    }

    void
    InitFeatures() override;

private:
    void
    run_clustering(const float* flat_vecs,
                   const Vector<InnerIdType>& vec_to_doc,
                   int64_t num_vecs,
                   int64_t dim);

    void
    build_rep_hgraph(const float* flat_vecs, int64_t dim);

    std::vector<std::pair<InnerIdType, float>>
    coarse_search(const float* query_tokens, uint32_t query_token_count, int64_t coarse_k) const;

    void
    serialize_rep_hgraph(StreamWriter& writer) const;

    void
    deserialize_rep_hgraph(StreamReader& reader);

    void
    split_cluster_incremental(InnerIdType cluster_idx);

private:
    IndexCommonParam common_param_;
    int64_t num_clusters_{0};

    // Per-cluster doc-ID lists; mutable for incremental Add.
    Vector<Vector<InnerIdType>> cluster_lists_;

    std::shared_ptr<HGraph> rep_hgraph_{nullptr};

    FlattenInterfacePtr mv_codes_{nullptr};

    Vector<InnerIdType> vec_to_cluster_;

    // Token-level metadata for precise split: maps global token_id → doc inner_id / offset / dist
    Vector<InnerIdType> token_to_doc_;
    Vector<uint32_t> token_to_offset_;
    Vector<float> token_to_dist_;

    // Per-cluster token count for O(1) split threshold check during Add
    Vector<uint64_t> cluster_token_counts_;

    float init_cluster_ratio_{0.2f};
    int64_t max_cluster_size_{64};
    int64_t split_start_idx_{32};
    int64_t random_seed_{42};
    int64_t default_coarse_k_{8};
    int64_t default_rerank_k_{100};

    uint64_t resize_increase_count_bit_{10};

    mutable std::shared_mutex global_mutex_;
};

}  // namespace vsag
