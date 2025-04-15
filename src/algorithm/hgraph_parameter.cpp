
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

#include "hgraph_parameter.h"

#include <fmt/format-inl.h>

#include "data_cell/graph_interface_parameter.h"
#include "inner_string_params.h"
#include "vsag/constants.h"

namespace vsag {

HGraphParameter::HGraphParameter(const JsonType& json) : HGraphParameter() {
    this->FromJson(json);
}

HGraphParameter::HGraphParameter() : name(INDEX_TYPE_HGRAPH) {
}

void
HGraphParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.contains(HGRAPH_USE_REORDER_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_USE_REORDER_KEY));
    this->use_reorder = json[HGRAPH_USE_REORDER_KEY];

    if (json.contains(HGRAPH_IGNORE_REORDER_KEY)) {
        this->ignore_reorder = json[HGRAPH_IGNORE_REORDER_KEY];
    }

    CHECK_ARGUMENT(json.contains(HGRAPH_BASE_CODES_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_BASE_CODES_KEY));
    const auto& base_codes_json = json[HGRAPH_BASE_CODES_KEY];
    this->base_codes_param = std::make_shared<FlattenDataCellParameter>();
    this->base_codes_param->FromJson(base_codes_json);

    if (use_reorder) {
        CHECK_ARGUMENT(json.contains(HGRAPH_PRECISE_CODES_KEY),
                       fmt::format("hgraph parameters must contains {}", HGRAPH_PRECISE_CODES_KEY));
        const auto& precise_codes_json = json[HGRAPH_PRECISE_CODES_KEY];
        this->precise_codes_param = std::make_shared<FlattenDataCellParameter>();
        this->precise_codes_param->FromJson(precise_codes_json);
    }

    CHECK_ARGUMENT(json.contains(HGRAPH_GRAPH_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_GRAPH_KEY));
    const auto& graph_json = json[HGRAPH_GRAPH_KEY];
    this->bottom_graph_param = GraphInterfaceParameter::GetGraphParameterByJson(graph_json);

    if (json.contains(BUILD_PARAMS_KEY)) {
        const auto& build_params = json[BUILD_PARAMS_KEY];
        if (build_params.contains(BUILD_EF_CONSTRUCTION)) {
            this->ef_construction = build_params[BUILD_EF_CONSTRUCTION];
        }
        if (build_params.contains(BUILD_THREAD_COUNT)) {
            this->build_thread_count = build_params[BUILD_THREAD_COUNT];
        }
    }

    CHECK_ARGUMENT(json.contains(HGRAPH_EXTRA_INFO_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_EXTRA_INFO_KEY));
    const auto& extra_info_json = json[HGRAPH_EXTRA_INFO_KEY];
    this->extra_info_param = std::make_shared<ExtraInfoDataCellParameter>();
    this->extra_info_param->FromJson(extra_info_json);
}

JsonType
HGraphParameter::ToJson() {
    JsonType json;
    json["type"] = INDEX_TYPE_HGRAPH;

    json[HGRAPH_USE_REORDER_KEY] = this->use_reorder;
    json[HGRAPH_BASE_CODES_KEY] = this->base_codes_param->ToJson();
    if (use_reorder) {
        json[HGRAPH_PRECISE_CODES_KEY] = this->precise_codes_param->ToJson();
    }
    json[HGRAPH_GRAPH_KEY] = this->bottom_graph_param->ToJson();

    json[BUILD_PARAMS_KEY][BUILD_EF_CONSTRUCTION] = this->ef_construction;
    json[BUILD_PARAMS_KEY][BUILD_THREAD_COUNT] = this->build_thread_count;
    json[HGRAPH_EXTRA_INFO_KEY] = this->extra_info_param->ToJson();
    return json;
}

// NOLINTBEGIN(readability-simplify-boolean-expr)

HGraphSearchParameters
HGraphSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    HGraphSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.contains(INDEX_TYPE_HGRAPH),
                   fmt::format("parameters must contains {}", INDEX_TYPE_HGRAPH));

    CHECK_ARGUMENT(
        params[INDEX_TYPE_HGRAPH].contains(HGRAPH_PARAMETER_EF_RUNTIME),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_TYPE_HGRAPH, HGRAPH_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_EF_RUNTIME];
    if (params[INDEX_TYPE_HGRAPH].contains(HGRAPH_USE_EXTRA_INFO_FILTER)) {
        obj.use_extra_info_filter = params[INDEX_TYPE_HGRAPH][HGRAPH_USE_EXTRA_INFO_FILTER];
    }
    CHECK_ARGUMENT((1 <= obj.ef_search) and (obj.ef_search <= 1000),
                   fmt::format("ef_search({}) must in range[1, 1000]", obj.ef_search));

    return obj;
}
}  // namespace vsag

// NOLINTEND(readability-simplify-boolean-expr)
