
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

#include "simq_parameter.h"

#include <fmt/format.h>

#include "inner_string_params.h"
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"

namespace vsag {

static constexpr const char* SIMQ_INIT_CLUSTER_RATIO = "init_cluster_ratio";
static constexpr const char* SIMQ_MAX_CLUSTER_SIZE = "max_cluster_size";
static constexpr const char* SIMQ_SPLIT_START_IDX = "split_start_idx";
static constexpr const char* SIMQ_RANDOM_SEED = "random_seed";
static constexpr const char* SIMQ_COARSE_K = "coarse_k";
static constexpr const char* SIMQ_RERANK_K = "rerank_k";

void
SIMQParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);

    if (json.Contains(SIMQ_INIT_CLUSTER_RATIO))
        init_cluster_ratio = json[SIMQ_INIT_CLUSTER_RATIO].GetFloat();
    if (json.Contains(SIMQ_MAX_CLUSTER_SIZE))
        max_cluster_size = json[SIMQ_MAX_CLUSTER_SIZE].GetInt();
    if (json.Contains(SIMQ_SPLIT_START_IDX))
        split_start_idx = json[SIMQ_SPLIT_START_IDX].GetInt();
    if (json.Contains(SIMQ_RANDOM_SEED))
        random_seed = json[SIMQ_RANDOM_SEED].GetInt();
    if (json.Contains(SIMQ_COARSE_K))
        coarse_k = json[SIMQ_COARSE_K].GetInt();
    if (json.Contains(SIMQ_RERANK_K))
        rerank_k = json[SIMQ_RERANK_K].GetInt();

    CHECK_ARGUMENT(init_cluster_ratio > 0.0f && init_cluster_ratio <= 1.0f,
                   "simq: init_cluster_ratio must be in (0, 1]");
    CHECK_ARGUMENT(max_cluster_size > 1, "simq: max_cluster_size must be > 1");
    CHECK_ARGUMENT(split_start_idx > 1 && split_start_idx < max_cluster_size,
                   "simq: split_start_idx must be in (1, max_cluster_size)");
    CHECK_ARGUMENT(coarse_k > 0, "simq: coarse_k must be > 0");
    CHECK_ARGUMENT(rerank_k > 0, "simq: rerank_k must be > 0");

    CHECK_ARGUMENT(json.Contains(BASE_CODES_KEY),
                   fmt::format("simq parameters must contain {}", BASE_CODES_KEY));
    base_codes_param = CreateFlattenParam(json[BASE_CODES_KEY]);
}

JsonType
SIMQParameter::ToJson() const {
    CHECK_ARGUMENT(base_codes_param != nullptr, "simq: base_codes_param is not initialized");
    JsonType json = InnerIndexParameter::ToJson();
    json[TYPE_KEY].SetString(INDEX_SIMQ);
    json[SIMQ_INIT_CLUSTER_RATIO].SetFloat(init_cluster_ratio);
    json[SIMQ_MAX_CLUSTER_SIZE].SetInt(max_cluster_size);
    json[SIMQ_SPLIT_START_IDX].SetInt(split_start_idx);
    json[SIMQ_RANDOM_SEED].SetInt(random_seed);
    json[SIMQ_COARSE_K].SetInt(coarse_k);
    json[SIMQ_RERANK_K].SetInt(rerank_k);
    json[BASE_CODES_KEY].SetJson(base_codes_param->ToJson());
    return json;
}

bool
SIMQParameter::CheckCompatibility(const vsag::ParamPtr& other) const {
    if (not InnerIndexParameter::CheckCompatibility(other)) {
        return false;
    }
    PARAM_CAST_OR_RETURN(SIMQParameter, p, other);
    CHECK_SUB_PARAM(*this, *p, base_codes_param);
    return true;
}

SIMQSearchParameters
SIMQSearchParameters::FromJson(const std::string& json_string) {
    SIMQSearchParameters obj;
    if (json_string.empty()) {
        return obj;
    }
    auto params = JsonType::Parse(json_string);
    if (!params.Contains(INDEX_SIMQ)) {
        return obj;
    }
    const auto& simq_params = params[INDEX_SIMQ];
    obj.IndexSearchParameter::FromJson(simq_params);
    if (simq_params.Contains(SIMQ_COARSE_K))
        obj.coarse_k = simq_params[SIMQ_COARSE_K].GetInt();
    if (simq_params.Contains(SIMQ_RERANK_K))
        obj.rerank_k = simq_params[SIMQ_RERANK_K].GetInt();
    return obj;
}

}  // namespace vsag
