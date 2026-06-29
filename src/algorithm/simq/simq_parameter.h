
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

#include "algorithm/index_search_parameter.h"
#include "algorithm/inner_index_parameter.h"
#include "datacell/flatten_interface_parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

class SIMQParameter : public InnerIndexParameter {
public:
    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    // Clustering construction parameters
    float init_cluster_ratio{0.2f};  // fraction of vectors used as initial seeds
    int64_t max_cluster_size{64};    // split threshold
    int64_t split_start_idx{32};     // index within sorted cluster where new cluster begins
    int64_t random_seed{42};

    // Search parameters (index-level defaults, overridable per-query)
    int64_t coarse_k{8};    // clusters searched per query token
    int64_t rerank_k{100};  // docs reranked with exact MaxSim

    // Storage for doc token vectors (used during MaxSim rerank)
    FlattenInterfaceParamPtr base_codes_param{nullptr};
};

DEFINE_POINTER(SIMQParameter);

class SIMQSearchParameters : public IndexSearchParameter {
public:
    int64_t coarse_k{-1};  // -1 means use index default
    int64_t rerank_k{-1};  // -1 means use index default

    static SIMQSearchParameters
    FromJson(const std::string& json_string);
};

}  // namespace vsag
