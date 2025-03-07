
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
#include "diskann_zparameters.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"

// NOLINTBEGIN(readability-simplify-boolean-expr)

namespace vsag {

void
PyramidParameters::FromJson(const JsonType& json) {
    // init graph param
    CHECK_ARGUMENT(json.contains(GRAPH_TYPE_ODESCENT),
                   fmt::format("pyramid parameters must contains {}", GRAPH_TYPE_ODESCENT));
    const auto& graph_json = json[GRAPH_TYPE_ODESCENT];
    graph_param = GraphInterfaceParameter::GetGraphParameterByJson(graph_json);
    // FIXME(inabao): This issue, where the edge length in the sparse graph defined in HGraph is half of the intended length, has been addressed here and will be revised in a subsequent PR.
    std::dynamic_pointer_cast<GraphDataCellParameter>(graph_param)->max_degree_ *= 2;
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
}
JsonType
PyramidParameters::ToJson() {
    JsonType json;
    json[GRAPH_TYPE_ODESCENT] = graph_param->ToJson();
    json[GRAPH_TYPE_ODESCENT][GRAPH_PARAM_MAX_DEGREE] =
        std::dynamic_pointer_cast<GraphDataCellParameter>(graph_param)->max_degree_ / 2;
    json[GRAPH_TYPE_ODESCENT].update(odescent_param->ToJson());
    json[PYRAMID_PARAMETER_BASE_CODES] = flatten_data_cell_param->ToJson();
    return json;
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
