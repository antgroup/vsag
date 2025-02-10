
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

namespace vsag {

HGraphParameter::HGraphParameter(const JsonType& json) : HGraphParameter() {
    this->FromJson(json);
}

HGraphParameter::HGraphParameter() : name_(INDEX_TYPE_HGRAPH) {
}

void
HGraphParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.contains(HGRAPH_USE_REORDER_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_USE_REORDER_KEY));
    this->use_reorder_ = json[HGRAPH_USE_REORDER_KEY];

    CHECK_ARGUMENT(json.contains(HGRAPH_BASE_CODES_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_BASE_CODES_KEY));
    const auto& base_codes_json = json[HGRAPH_BASE_CODES_KEY];
    this->base_codes_param_ = std::make_shared<FlattenDataCellParameter>();
    this->base_codes_param_->FromJson(base_codes_json);

    if (use_reorder_) {
        CHECK_ARGUMENT(json.contains(HGRAPH_PRECISE_CODES_KEY),
                       fmt::format("hgraph parameters must contains {}", HGRAPH_PRECISE_CODES_KEY));
        const auto& precise_codes_json = json[HGRAPH_PRECISE_CODES_KEY];
        this->precise_codes_param_ = std::make_shared<FlattenDataCellParameter>();
        this->precise_codes_param_->FromJson(precise_codes_json);
    }

    CHECK_ARGUMENT(json.contains(HGRAPH_GRAPH_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_GRAPH_KEY));
    const auto& graph_json = json[HGRAPH_GRAPH_KEY];
    this->bottom_graph_param_ = GraphInterfaceParameter::GetGraphParameterByJson(graph_json);

    if (json.contains(BUILD_PARAMS_KEY)) {
        const auto& build_params = json[BUILD_PARAMS_KEY];
        if (build_params.contains(BUILD_EF_CONSTRUCTION)) {
            this->ef_construction_ = build_params[BUILD_EF_CONSTRUCTION];
        }
        if (build_params.contains(BUILD_THREAD_COUNT)) {
            this->build_thread_count_ = build_params[BUILD_THREAD_COUNT];
        }
    }
}

JsonType
HGraphParameter::ToJson() {
    JsonType json;
    json["type"] = INDEX_TYPE_HGRAPH;

    json[HGRAPH_USE_REORDER_KEY] = this->use_reorder_;
    json[HGRAPH_BASE_CODES_KEY] = this->base_codes_param_->ToJson();
    if (use_reorder_) {
        json[HGRAPH_PRECISE_CODES_KEY] = this->precise_codes_param_->ToJson();
    }
    json[HGRAPH_GRAPH_KEY] = this->bottom_graph_param_->ToJson();

    json[BUILD_PARAMS_KEY][BUILD_EF_CONSTRUCTION] = this->ef_construction_;
    json[BUILD_PARAMS_KEY][BUILD_THREAD_COUNT] = this->build_thread_count_;
    return json;
}

}  // namespace vsag
