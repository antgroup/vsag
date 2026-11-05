
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

#include "graph_datacell_parameter.h"

#include <fmt/format.h>

#include "inner_string_params.h"
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"
namespace vsag {

void
GraphDataCellParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.Contains(IO_PARAMS_KEY),
                   fmt::format("graph interface parameters must contains {}", IO_PARAMS_KEY));
    this->io_parameter_ = IOParameter::GetIOParameterByJson(json[IO_PARAMS_KEY]);
    if (json.Contains(GRAPH_PARAM_MAX_DEGREE_KEY)) {
        this->max_degree_ = json[GRAPH_PARAM_MAX_DEGREE_KEY].GetUint64();
    }
    if (json.Contains(GRAPH_PARAM_INIT_MAX_CAPACITY_KEY)) {
        this->init_max_capacity_ = json[GRAPH_PARAM_INIT_MAX_CAPACITY_KEY].GetUint64();
    }
    if (json.Contains(GRAPH_SUPPORT_REMOVE)) {
        this->support_remove_ = json[GRAPH_SUPPORT_REMOVE].GetBool();
    }
    if (json.Contains(REMOVE_FLAG_BIT)) {
        this->remove_flag_bit_ = json[REMOVE_FLAG_BIT].GetInt();
    }
    if (json.Contains(HGRAPH_USE_REVERSE_EDGES_KEY)) {
        this->use_reverse_edges_ = json[HGRAPH_USE_REVERSE_EDGES_KEY].GetBool();
    }
    if (json.Contains(SUPPORT_DUPLICATE)) {
        this->support_duplicate_ = json[SUPPORT_DUPLICATE].GetBool();
    }
}

JsonType
GraphDataCellParameter::ToJson() const {
    JsonType json;
    json[IO_PARAMS_KEY].SetJson(this->io_parameter_->ToJson());
    json[GRAPH_PARAM_MAX_DEGREE_KEY].SetUint64(this->max_degree_);
    json[GRAPH_PARAM_INIT_MAX_CAPACITY_KEY].SetUint64(this->init_max_capacity_);
    json[GRAPH_SUPPORT_REMOVE].SetBool(this->support_remove_);
    json[REMOVE_FLAG_BIT].SetInt(this->remove_flag_bit_);
    json[HGRAPH_USE_REVERSE_EDGES_KEY].SetBool(this->use_reverse_edges_);
    json[SUPPORT_DUPLICATE].SetBool(this->support_duplicate_);
    return json;
}
bool
GraphDataCellParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(GraphDataCellParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, max_degree_);
    CHECK_FIELD_EQ(*this, *p, support_remove_);
    CHECK_FIELD_EQ(*this, *p, remove_flag_bit_);
    CHECK_FIELD_EQ(*this, *p, use_reverse_edges_);
    CHECK_FIELD_EQ(*this, *p, support_duplicate_);
    return true;
}

}  // namespace vsag
