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

#include "mci_parameter.h"

#include <fmt/format.h>

#include "common.h"
#include "datacell/flatten_datacell_parameter.h"
#include "impl/logger/logger.h"
#include "inner_string_params.h"
#include "vsag/constants.h"

namespace vsag {

void
MCIParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);
    CHECK_ARGUMENT(json.Contains(BASE_CODES_KEY),
                   fmt::format("mci parameters must contains {}", BASE_CODES_KEY));
    this->base_codes_param = CreateFlattenParam(json[BASE_CODES_KEY]);

    if (json.Contains(MCI_PARAMETER_MAX_DEGREE)) {
        this->max_degree = static_cast<uint64_t>(json[MCI_PARAMETER_MAX_DEGREE].GetInt());
    }
    if (json.Contains(MCI_PARAMETER_MCS)) {
        this->mcs = static_cast<uint64_t>(json[MCI_PARAMETER_MCS].GetInt());
    }
    if (json.Contains(MCI_PARAMETER_CLIQUE_MAX)) {
        this->clique_max = static_cast<uint64_t>(json[MCI_PARAMETER_CLIQUE_MAX].GetInt());
    }
    if (json.Contains(MCI_PARAMETER_ALPHA)) {
        this->alpha = json[MCI_PARAMETER_ALPHA].GetFloat();
    }
    if (json.Contains(MCI_PARAMETER_KNNG_PATH)) {
        this->knng_path = json[MCI_PARAMETER_KNNG_PATH].GetString();
    }
    if (json.Contains(MCI_PARAMETER_CLIQUE_PATH)) {
        this->clique_path = json[MCI_PARAMETER_CLIQUE_PATH].GetString();
    }
    if (json.Contains(MCI_PARAMETER_USE_HGRAPH_HYBRID)) {
        this->use_hgraph_hybrid = json[MCI_PARAMETER_USE_HGRAPH_HYBRID].GetBool();
    }
    if (json.Contains(MCI_PARAMETER_HGRAPH_VALID_RATIO_THRESHOLD)) {
        this->hgraph_valid_ratio_threshold =
            json[MCI_PARAMETER_HGRAPH_VALID_RATIO_THRESHOLD].GetFloat();
    }
    if (json.Contains(MCI_PARAMETER_HGRAPH_INDEX_PARAM)) {
        this->hgraph_param = std::make_shared<HGraphParameter>();
        this->hgraph_param->FromJson(json[MCI_PARAMETER_HGRAPH_INDEX_PARAM]);
    }
    if (json.Contains(MCI_PARAMETER_HGRAPH_INDEX_PATH)) {
        this->hgraph_index_path = json[MCI_PARAMETER_HGRAPH_INDEX_PATH].GetString();
    }
    if (json.Contains(MCI_PARAMETER_HGRAPH_EF_SEARCH)) {
        this->hgraph_ef_search = json[MCI_PARAMETER_HGRAPH_EF_SEARCH].GetInt();
    }

    CHECK_ARGUMENT(this->max_degree > 0, "mci max_degree must be positive");
    CHECK_ARGUMENT(this->mcs > 0, "mci mcs must be positive");
    CHECK_ARGUMENT(this->clique_max > 0, "mci clique_max must be positive");
    CHECK_ARGUMENT(this->alpha >= 1.0F, "mci alpha must be greater than or equal to 1.0");
    CHECK_ARGUMENT(
        this->hgraph_valid_ratio_threshold >= 0.0F and this->hgraph_valid_ratio_threshold <= 1.0F,
        "mci hgraph_valid_ratio_threshold must be in range [0, 1]");
    CHECK_ARGUMENT(this->hgraph_ef_search > 0, "mci hgraph_ef_search must be positive");
    CHECK_ARGUMENT(not this->use_hgraph_hybrid or this->hgraph_param != nullptr,
                   "mci use_hgraph_hybrid requires hgraph_index_param");
}

JsonType
MCIParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[TYPE_KEY].SetString(INDEX_TYPE_MCI);
    json[BASE_CODES_KEY].SetJson(this->base_codes_param->ToJson());
    json[MCI_PARAMETER_MAX_DEGREE].SetInt(static_cast<int64_t>(this->max_degree));
    json[MCI_PARAMETER_MCS].SetInt(static_cast<int64_t>(this->mcs));
    json[MCI_PARAMETER_CLIQUE_MAX].SetInt(static_cast<int64_t>(this->clique_max));
    json[MCI_PARAMETER_ALPHA].SetFloat(this->alpha);
    json[MCI_PARAMETER_KNNG_PATH].SetString(this->knng_path);
    json[MCI_PARAMETER_CLIQUE_PATH].SetString(this->clique_path);
    json[MCI_PARAMETER_USE_HGRAPH_HYBRID].SetBool(this->use_hgraph_hybrid);
    json[MCI_PARAMETER_HGRAPH_VALID_RATIO_THRESHOLD].SetFloat(this->hgraph_valid_ratio_threshold);
    if (this->hgraph_param != nullptr) {
        json[MCI_PARAMETER_HGRAPH_INDEX_PARAM].SetJson(this->hgraph_param->ToJson());
    }
    json[MCI_PARAMETER_HGRAPH_EF_SEARCH].SetInt(this->hgraph_ef_search);
    return json;
}

bool
MCIParameter::CheckCompatibility(const ParamPtr& other) const {
    if (not InnerIndexParameter::CheckCompatibility(other)) {
        return false;
    }
    auto mci_param = std::dynamic_pointer_cast<MCIParameter>(other);
    if (not mci_param) {
        logger::error("MCIParameter::CheckCompatibility: other parameter is not MCIParameter");
        return false;
    }
    if (not this->base_codes_param->CheckCompatibility(mci_param->base_codes_param)) {
        logger::error("MCIParameter::CheckCompatibility: base_codes_param mismatch");
        return false;
    }
    if (this->max_degree != mci_param->max_degree) {
        logger::error("MCIParameter::CheckCompatibility: max_degree mismatch");
        return false;
    }
    if (this->mcs != mci_param->mcs) {
        logger::error("MCIParameter::CheckCompatibility: mcs mismatch");
        return false;
    }
    if (this->clique_max != mci_param->clique_max) {
        logger::error("MCIParameter::CheckCompatibility: clique_max mismatch");
        return false;
    }
    return true;
}

MCISearchParameters
MCISearchParameters::FromJson(const std::string& json_string) {
    MCISearchParameters obj;
    if (json_string.empty()) {
        return obj;
    }

    auto params = JsonType::Parse(json_string);
    JsonType mci_params;
    if (params.Contains(INDEX_MCI)) {
        mci_params = params[INDEX_MCI];
    } else {
        mci_params = params;
    }

    obj.IndexSearchParameter::FromJson(mci_params);
    if (mci_params.Contains(MCI_SEARCH_PARAMETER_EF_SEARCH)) {
        obj.ef_search = mci_params[MCI_SEARCH_PARAMETER_EF_SEARCH].GetInt();
    }
    if (mci_params.Contains(MCI_SEARCH_PARAMETER_SEED_COUNT)) {
        obj.seed_count =
            static_cast<uint64_t>(mci_params[MCI_SEARCH_PARAMETER_SEED_COUNT].GetInt());
    }
    if (mci_params.Contains(MCI_SEARCH_PARAMETER_HOPS_LIMIT)) {
        obj.hops_limit =
            static_cast<uint32_t>(mci_params[MCI_SEARCH_PARAMETER_HOPS_LIMIT].GetInt());
    }
    if (mci_params.Contains(MCI_SEARCH_PARAMETER_RABITQ_ONE_BIT_SEARCH)) {
        obj.rabitq_one_bit_search =
            mci_params[MCI_SEARCH_PARAMETER_RABITQ_ONE_BIT_SEARCH].GetBool();
    }

    CHECK_ARGUMENT(obj.ef_search > 0, "mci ef_search must be positive");
    CHECK_ARGUMENT(obj.seed_count > 0, "mci seed_count must be positive");
    return obj;
}

}  // namespace vsag