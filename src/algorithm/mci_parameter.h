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

#include "hgraph_parameter.h"
#include "index_search_parameter.h"
#include "inner_index_parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

static constexpr const char* MCI_PARAMETER_MAX_DEGREE = "max_degree";
static constexpr const char* MCI_PARAMETER_MCS = "mcs";
static constexpr const char* MCI_PARAMETER_CLIQUE_MAX = "clique_max";
static constexpr const char* MCI_PARAMETER_ALPHA = "alpha";
static constexpr const char* MCI_PARAMETER_KNNG_PATH = "knng_path";
static constexpr const char* MCI_PARAMETER_CLIQUE_PATH = "clique_path";
static constexpr const char* MCI_PARAMETER_USE_HGRAPH_HYBRID = "use_hgraph_hybrid";
static constexpr const char* MCI_PARAMETER_HGRAPH_VALID_RATIO_THRESHOLD =
    "hgraph_valid_ratio_threshold";
static constexpr const char* MCI_PARAMETER_HGRAPH_INDEX_PARAM = "hgraph_index_param";
static constexpr const char* MCI_PARAMETER_HGRAPH_INDEX_PATH = "hgraph_index_path";
static constexpr const char* MCI_PARAMETER_HGRAPH_EF_SEARCH = "hgraph_ef_search";
static constexpr const char* MCI_SEARCH_PARAMETER_EF_SEARCH = "ef_search";
static constexpr const char* MCI_SEARCH_PARAMETER_SEED_COUNT = "seed_count";
static constexpr const char* MCI_SEARCH_PARAMETER_HOPS_LIMIT = "hops_limit";
static constexpr const char* MCI_SEARCH_PARAMETER_RABITQ_ONE_BIT_SEARCH = "rabitq_one_bit_search";

DEFINE_POINTER(MCIParameter);
class MCIParameter : public InnerIndexParameter {
public:
    explicit MCIParameter() = default;

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const ParamPtr& other) const override;

public:
    FlattenInterfaceParamPtr base_codes_param{nullptr};
    uint64_t max_degree{32};
    uint64_t mcs{200};
    uint64_t clique_max{50};
    float alpha{1.2F};
    std::string knng_path{};
    std::string clique_path{};
    bool use_hgraph_hybrid{false};
    float hgraph_valid_ratio_threshold{1.0F};
    HGraphParameterPtr hgraph_param{nullptr};
    std::string hgraph_index_path{};
    int64_t hgraph_ef_search{100};
};

class MCISearchParameters : public IndexSearchParameter {
public:
    static MCISearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t ef_search{30};
    uint64_t seed_count{32};
    uint32_t hops_limit{std::numeric_limits<uint32_t>::max()};
    bool rabitq_one_bit_search{false};
};

}  // namespace vsag