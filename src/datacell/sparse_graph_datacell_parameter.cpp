
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

#include "utils/param_compat_macros.h"

namespace vsag {
SparseGraphDatacellParameter::SparseGraphDatacellParameter()
    : GraphInterfaceParameter(GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE) {
}

void
SparseGraphDatacellParameter::FromJson(const JsonType& json) {
    if (json.Contains(GRAPH_PARAM_MAX_DEGREE_KEY)) {
        this->max_degree_ = json[GRAPH_PARAM_MAX_DEGREE_KEY].GetInt();
    }
    if (json.Contains(GRAPH_SUPPORT_REMOVE)) {
        this->support_delete_ = json[GRAPH_SUPPORT_REMOVE].GetBool();
    }
    if (json.Contains(REMOVE_FLAG_BIT)) {
        this->remove_flag_bit_ = static_cast<uint32_t>(json[REMOVE_FLAG_BIT].GetInt());
    }
    if (json.Contains(SUPPORT_DUPLICATE)) {
        this->support_duplicate_ = json[SUPPORT_DUPLICATE].GetBool();
    }
    if (json.Contains(HGRAPH_USE_REVERSE_EDGES_KEY)) {
        this->use_reverse_edges_ = json[HGRAPH_USE_REVERSE_EDGES_KEY].GetBool();
    }
}

JsonType
SparseGraphDatacellParameter::ToJson() const {
    JsonType json;
    json[GRAPH_PARAM_MAX_DEGREE_KEY].SetInt(this->max_degree_);
    json[GRAPH_SUPPORT_REMOVE].SetBool(this->support_delete_);
    json[REMOVE_FLAG_BIT].SetInt(this->remove_flag_bit_);
    json[SUPPORT_DUPLICATE].SetBool(this->support_duplicate_);
    json[HGRAPH_USE_REVERSE_EDGES_KEY].SetBool(this->use_reverse_edges_);
    return json;
}

bool
SparseGraphDatacellParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(SparseGraphDatacellParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, max_degree_);
    CHECK_FIELD_EQ(*this, *p, support_delete_);
    CHECK_FIELD_EQ(*this, *p, remove_flag_bit_);
    CHECK_FIELD_EQ(*this, *p, support_duplicate_);
    CHECK_FIELD_EQ(*this, *p, use_reverse_edges_);
    return true;
}
}  // namespace vsag
