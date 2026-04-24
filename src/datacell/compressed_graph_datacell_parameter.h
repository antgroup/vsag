
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
#include "impl/logger/logger.h"
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
        auto graph_param = std::dynamic_pointer_cast<CompressedGraphDatacellParameter>(other);
        if (not graph_param) {
            logger::error(
                "CompressedGraphDatacellParameter::CheckCompatibility: other parameter "
                "is not a CompressedGraphDatacellParameter");
            return false;
        }
        if (max_degree_ != graph_param->max_degree_) {
            logger::error(
                "CompressedGraphDatacellParameter::CheckCompatibility: max_degree_ "
                "mismatch: {} vs {}",
                max_degree_,
                graph_param->max_degree_);
            return false;
        }
        if (support_duplicate_ != graph_param->support_duplicate_) {
            logger::error(
                "CompressedGraphDatacellParameter::CheckCompatibility: "
                "support_duplicate_ mismatch: {} vs {}",
                support_duplicate_,
                graph_param->support_duplicate_);
            return false;
        }
        if (use_reverse_edges_ != graph_param->use_reverse_edges_) {
            logger::error(
                "CompressedGraphDatacellParameter::CheckCompatibility: "
                "use_reverse_edges_ mismatch: {} vs {}",
                use_reverse_edges_,
                graph_param->use_reverse_edges_);
            return false;
        }
        return true;
    }
};
}  // namespace vsag
