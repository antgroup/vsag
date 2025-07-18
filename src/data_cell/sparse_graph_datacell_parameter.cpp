
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

#include "sparse_graph_datacell_parameter.h"

#include "logger.h"

namespace vsag {
SparseGraphDatacellParameter::SparseGraphDatacellParameter()
    : GraphInterfaceParameter(GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE) {
}

void
SparseGraphDatacellParameter::FromJson(const JsonType& json) {
    if (json.contains(GRAPH_PARAM_MAX_DEGREE)) {
        this->max_degree_ = json[GRAPH_PARAM_MAX_DEGREE];
    }
    if (json.contains(GRAPH_SUPPORT_REMOVE)) {
        this->support_delete_ = json[GRAPH_SUPPORT_REMOVE];
    }
    if (json.contains(REMOVE_FLAG_BIT)) {
        this->remove_flag_bit_ = json[REMOVE_FLAG_BIT];
    }
}

JsonType
SparseGraphDatacellParameter::ToJson() const {
    JsonType json;
    json[GRAPH_PARAM_MAX_DEGREE] = this->max_degree_;
    json[GRAPH_SUPPORT_REMOVE] = support_delete_;
    json[REMOVE_FLAG_BIT] = remove_flag_bit_;
    return json;
}

bool
SparseGraphDatacellParameter::CheckCompatibility(const ParamPtr& other) const {
    auto graph_param = std::dynamic_pointer_cast<SparseGraphDatacellParameter>(other);
    if (not graph_param) {
        logger::error(
            "SparseGraphDatacellParameter::CheckCompatibility: other parameter is not a "
            "SparseGraphDatacellParameter");
        return false;
    }
    if (max_degree_ != graph_param->max_degree_) {
        logger::error(
            "SparseGraphDatacellParameter::CheckCompatibility: max_degree_ mismatch: {} vs {}",
            max_degree_,
            graph_param->max_degree_);
        return false;
    }
    if (support_delete_ != graph_param->support_delete_) {
        logger::error(
            "SparseGraphDatacellParameter::CheckCompatibility: support_delete_ mismatch: {} vs {}",
            support_delete_,
            graph_param->support_delete_);
        return false;
    }
    if (remove_flag_bit_ != graph_param->remove_flag_bit_) {
        logger::error(
            "SparseGraphDatacellParameter::CheckCompatibility: remove_flag_bit_ mismatch: {} vs {}",
            remove_flag_bit_,
            graph_param->remove_flag_bit_);
        return false;
    }
    return true;
}
}  // namespace vsag