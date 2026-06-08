
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

#include "pyramid_zparameters.h"

#include "common.h"
#include "impl/logger/logger.h"
#include "index/diskann_zparameters.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "utils/param_compat_macros.h"

// NOLINTBEGIN(readability-simplify-boolean-expr)

namespace vsag {

void
PyramidParameters::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);
    // init graph param
    const auto& graph_json = json[GRAPH_KEY];

    graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE, graph_json);
    this->alpha = graph_json[ALPHA_KEY].GetFloat();
    this->max_degree = graph_json[GRAPH_PARAM_MAX_DEGREE_KEY].GetInt();

    this->graph_type = graph_json[GRAPH_TYPE_KEY].GetString();
    if (this->graph_type == GRAPH_TYPE_ODESCENT) {
        this->odescent_param = std::make_shared<ODescentParameter>();
        this->odescent_param->FromJson(graph_json);
    } else {
        if (json.Contains(EF_CONSTRUCTION_KEY)) {
            this->ef_construction = json[EF_CONSTRUCTION_KEY].GetInt();
        }
    }

    this->base_codes_param = CreateFlattenParam(json[BASE_CODES_KEY]);

    if (json.Contains(NO_BUILD_LEVELS)) {
        const auto& no_build_levels_json = json[NO_BUILD_LEVELS];
        CHECK_ARGUMENT(no_build_levels_json.IsArray(),
                       fmt::format("build_without_levels must be a list of integers"));
        this->no_build_levels = no_build_levels_json.GetVector();
        std::sort(this->no_build_levels.begin(), this->no_build_levels.end());
    }

    this->use_reorder = json[USE_REORDER_KEY].GetBool();
    if (this->use_reorder) {
        this->precise_codes_param = CreateFlattenParam(json[PRECISE_CODES_KEY]);
    }

    if (json.Contains(INDEX_MIN_SIZE)) {
        this->index_min_size = json[INDEX_MIN_SIZE].GetInt();
    }

    if (json.Contains(SUPPORT_DUPLICATE)) {
        this->support_duplicate = json[SUPPORT_DUPLICATE].GetBool();
    }
}
JsonType
PyramidParameters::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[NO_BUILD_LEVELS].SetVector(no_build_levels);
    json[BASE_CODES_KEY].SetJson(base_codes_param->ToJson());

    auto graph_json = graph_param->ToJson();
    graph_json[ALPHA_KEY].SetFloat(this->alpha);
    graph_json[GRAPH_TYPE_KEY].SetString(this->graph_type);
    if (this->graph_type == GRAPH_TYPE_ODESCENT) {
        graph_json.UpdateJson(odescent_param->ToJson());
    } else {
        json[EF_CONSTRUCTION_KEY].SetInt(this->ef_construction);
    }
    json[GRAPH_KEY].SetJson(graph_json);
    json[USE_REORDER_KEY].SetBool(this->use_reorder);
    json[INDEX_MIN_SIZE].SetInt(index_min_size);
    json[SUPPORT_DUPLICATE].SetBool(support_duplicate);
    if (this->use_reorder) {
        json[PRECISE_CODES_KEY].SetJson(precise_codes_param->ToJson());
    }
    return json;
}

bool
PyramidParameters::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(PyramidParameters, p, other);
    CHECK_SUB_PARAM(*this, *p, graph_param);
    CHECK_SUB_PARAM(*this, *p, base_codes_param);
    if (no_build_levels.size() != p->no_build_levels.size() ||
        not std::is_permutation(
            no_build_levels.begin(), no_build_levels.end(), p->no_build_levels.begin())) {
        logger::error("PyramidParameters::CheckCompatibility: no_build_levels are not compatible");
        return false;
    }
    CHECK_FIELD_EQ(*this, *p, use_reorder);
    if (this->use_reorder) {
        CHECK_SUB_PARAM(*this, *p, precise_codes_param);
    }
    CHECK_FIELD_EQ(*this, *p, index_min_size);
    CHECK_FIELD_EQ(*this, *p, support_duplicate);
    return true;
}

PyramidSearchParameters
PyramidSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::Parse(json_string);

    PyramidSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.Contains(INDEX_PYRAMID),
                   fmt::format("parameters must contains {}", INDEX_PYRAMID));
    obj.IndexSearchParameter::FromJson(params[INDEX_PYRAMID]);

    CHECK_ARGUMENT(
        params[INDEX_PYRAMID].Contains(PYRAMID_PARAMETER_EF_SEARCH),
        fmt::format("parameters[{}] must contains {}", INDEX_PYRAMID, PYRAMID_PARAMETER_EF_SEARCH));
    obj.ef_search = params[INDEX_PYRAMID][PYRAMID_PARAMETER_EF_SEARCH].GetInt();
    if (params[INDEX_PYRAMID].Contains(PYRAMID_PARAMETER_SUBINDEX_EF_SEARCH)) {
        obj.subindex_ef_search =
            params[INDEX_PYRAMID][PYRAMID_PARAMETER_SUBINDEX_EF_SEARCH].GetInt();
    }
    return obj;
}
}  // namespace vsag

// NOLINTEND(readability-simplify-boolean-expr)
