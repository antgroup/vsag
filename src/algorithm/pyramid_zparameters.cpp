
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
#include "index/diskann_zparameters.h"
#include "io/memory_io_parameter.h"
#include "logger.h"
#include "quantization/fp32_quantizer_parameter.h"

// NOLINTBEGIN(readability-simplify-boolean-expr)

namespace vsag {

void
PyramidParameters::FromJson(const JsonType& json) {
    // init graph param
    CHECK_ARGUMENT(json.contains(GRAPH_TYPE_ODESCENT),
                   fmt::format("pyramid parameters must contains {}", GRAPH_TYPE_ODESCENT));
    const auto& graph_json = json[GRAPH_TYPE_ODESCENT];
    graph_param = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE, graph_json);
    odescent_param = std::make_shared<ODescentParameter>();
    odescent_param->FromJson(graph_json);
    this->flatten_data_cell_param = std::make_shared<FlattenDataCellParameter>();
    if (json.contains(PYRAMID_PARAMETER_BASE_CODES)) {
        this->flatten_data_cell_param->FromJson(json[PYRAMID_PARAMETER_BASE_CODES]);
    } else {
        this->flatten_data_cell_param->io_parameter = std::make_shared<MemoryIOParameter>();
        this->flatten_data_cell_param->quantizer_parameter =
            std::make_shared<FP32QuantizerParameter>();
    }

    if (json.contains(BUILD_EF_CONSTRUCTION)) {
        this->ef_construction = json[BUILD_EF_CONSTRUCTION];
    }

    if (json.contains(NO_BUILD_LEVELS)) {
        const auto& no_build_levels_json = json[NO_BUILD_LEVELS];
        CHECK_ARGUMENT(no_build_levels_json.is_array(),
                       fmt::format("build_without_levels must be a list of integers"));
        for (const auto& item : no_build_levels_json) {
            CHECK_ARGUMENT(item.is_number_integer(),
                           "build_without_levels must be a list of integers");
        }
        this->no_build_levels = no_build_levels_json.get<std::vector<int32_t>>();
    }
}
JsonType
PyramidParameters::ToJson() const {
    JsonType json;
    json[GRAPH_TYPE_ODESCENT] = graph_param->ToJson();
    json[GRAPH_TYPE_ODESCENT].update(odescent_param->ToJson());
    json[PYRAMID_PARAMETER_BASE_CODES] = flatten_data_cell_param->ToJson();
    json[NO_BUILD_LEVELS] = no_build_levels;
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

    if (not flatten_data_cell_param->CheckCompatibility(pyramid_param->flatten_data_cell_param)) {
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

    return true;
}

PyramidSearchParameters
PyramidSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    PyramidSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.contains(INDEX_PYRAMID),
                   fmt::format("parameters must contains {}", INDEX_PYRAMID));

    CHECK_ARGUMENT(
        params[INDEX_PYRAMID].contains(HNSW_PARAMETER_EF_RUNTIME),
        fmt::format("parameters[{}] must contains {}", INDEX_PYRAMID, HNSW_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[INDEX_PYRAMID][HNSW_PARAMETER_EF_RUNTIME];
    CHECK_ARGUMENT((1 <= obj.ef_search) and (obj.ef_search <= 1000),
                   fmt::format("ef_search({}) must in range[1, 1000]", obj.ef_search));
    return obj;
}
}  // namespace vsag

// NOLINTEND(readability-simplify-boolean-expr)
