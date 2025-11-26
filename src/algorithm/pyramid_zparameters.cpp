
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

#include "algorithm/pyramid_zparameters.h"

#include "common.h"
#include "impl/logger/logger.h"
#include "index/diskann_zparameters.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"

// NOLINTBEGIN(readability-simplify-boolean-expr)

namespace vsag {

void
PyramidParameters::FromJson(const JsonType& json) {
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
    return json;
}

bool
PyramidParameters::CheckCompatibility(const ParamPtr& other) const {
    auto pyramid_param = std::dynamic_pointer_cast<PyramidParameters>(other);
    if (not pyramid_param) {
        logger::error(
            "PyramidParameters::CheckCompatibility: other parameter is not PyramidParameters");
        return false;
    }
    if (not graph_param->CheckCompatibility(pyramid_param->graph_param)) {
        logger::error("PyramidParameters::CheckCompatibility: graph parameters are not compatible");
        return false;
    }

    if (not base_codes_param->CheckCompatibility(pyramid_param->base_codes_param)) {
        logger::error(
            "PyramidParameters::CheckCompatibility: flatten data cell parameters are not "
            "compatible");
        return false;
    }
    if (no_build_levels.size() != pyramid_param->no_build_levels.size() ||
        not std::is_permutation(no_build_levels.begin(),
                                no_build_levels.end(),
                                pyramid_param->no_build_levels.begin())) {
        logger::error("PyramidParameters::CheckCompatibility: no_build_levels are not compatible");
        return false;
    }

    if (pyramid_param->use_reorder != this->use_reorder) {
        logger::error(
            "PyramidParameters::CheckCompatibility: use_reorder settings are not compatible");
        return false;
    }

    if (this->use_reorder &&
        not precise_codes_param->CheckCompatibility(pyramid_param->precise_codes_param)) {
        logger::error(
            "PyramidParameters::CheckCompatibility: precise_codes_param are not compatible");
        return false;
    }

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
        params[INDEX_PYRAMID].Contains(HNSW_PARAMETER_EF_RUNTIME),
        fmt::format("parameters[{}] must contains {}", INDEX_PYRAMID, HNSW_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[INDEX_PYRAMID][HNSW_PARAMETER_EF_RUNTIME].GetInt();
    CHECK_ARGUMENT((1 <= obj.ef_search) and (obj.ef_search <= 1000),
                   fmt::format("ef_search({}) must in range[1, 1000]", obj.ef_search));
    return obj;
}
}  // namespace vsag

// NOLINTEND(readability-simplify-boolean-expr)
