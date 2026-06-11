
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
#include "utils/param_compat_macros.h"
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
        if (json.Contains(SUPPORT_DUPLICATE)) {
            this->support_duplicate_ = json[SUPPORT_DUPLICATE].GetBool();
        }
        if (json.Contains(HGRAPH_USE_REVERSE_EDGES_KEY)) {
            this->use_reverse_edges_ = json[HGRAPH_USE_REVERSE_EDGES_KEY].GetBool();
        }
    }

    JsonType
    ToJson() const override {
        JsonType json;
        json[GRAPH_PARAM_MAX_DEGREE_KEY].SetInt(this->max_degree_);
        json[GRAPH_STORAGE_TYPE_KEY].SetString(GRAPH_STORAGE_TYPE_VALUE_COMPRESSED);
        json[SUPPORT_DUPLICATE].SetBool(this->support_duplicate_);
        json[HGRAPH_USE_REVERSE_EDGES_KEY].SetBool(this->use_reverse_edges_);
        return json;
    }

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override {
        PARAM_CAST_OR_RETURN(CompressedGraphDatacellParameter, p, other);
        CHECK_FIELD_EQ(*this, *p, max_degree_);
        CHECK_FIELD_EQ(*this, *p, support_duplicate_);
        CHECK_FIELD_EQ(*this, *p, use_reverse_edges_);
        return true;
    }
};
}  // namespace vsag
