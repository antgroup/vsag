
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

#include "graph_interface_parameter.h"
#include "inner_string_params.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(CompressedGraphDatacellParam, CompressedGraphDatacellParameter);
class CompressedGraphDatacellParameter : public GraphInterfaceParameter {
public:
    CompressedGraphDatacellParameter()
        : GraphInterfaceParameter(GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED) {
    }

    void
    FromJson(const JsonType& json) override {
        if (json.Contains(GRAPH_PARAM_MAX_DEGREE_KEY)) {
            this->max_degree_ = json[GRAPH_PARAM_MAX_DEGREE_KEY].GetInt();
        }
    }

    JsonType
    ToJson() const override {
        JsonType json;
        json[GRAPH_PARAM_MAX_DEGREE_KEY].SetInt(this->max_degree_);
        json[GRAPH_STORAGE_TYPE_KEY].SetString(GRAPH_STORAGE_TYPE_VALUE_COMPRESSED);
        return json;
    }
};
}  // namespace vsag
