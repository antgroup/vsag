
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
#include "logger.h"
#include "vsag/constants.h"
namespace vsag {

void
GraphDataCellParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.contains(IO_PARAMS_KEY),
                   fmt::format("graph interface parameters must contains {}", IO_PARAMS_KEY));
    this->io_parameter_ = IOParameter::GetIOParameterByJson(json[IO_PARAMS_KEY]);
    if (json.contains(GRAPH_PARAM_MAX_DEGREE)) {
        this->max_degree_ = json[GRAPH_PARAM_MAX_DEGREE];
    }
    if (json.contains(GRAPH_PARAM_INIT_MAX_CAPACITY)) {
        this->init_max_capacity_ = json[GRAPH_PARAM_INIT_MAX_CAPACITY];
    }
    if (json.contains(GRAPH_SUPPORT_REMOVE)) {
        this->support_remove_ = json[GRAPH_SUPPORT_REMOVE];
    }
    if (json.contains(REMOVE_FLAG_BIT)) {
        this->remove_flag_bit_ = json[REMOVE_FLAG_BIT];
    }
}

JsonType
GraphDataCellParameter::ToJson() const {
    JsonType json;
    json[IO_PARAMS_KEY] = this->io_parameter_->ToJson();
    json[GRAPH_PARAM_MAX_DEGREE] = this->max_degree_;
    json[GRAPH_PARAM_INIT_MAX_CAPACITY] = this->init_max_capacity_;
    json[GRAPH_SUPPORT_REMOVE] = this->support_remove_;
    json[REMOVE_FLAG_BIT] = this->remove_flag_bit_;
    return json;
}
bool
GraphDataCellParameter::CheckCompatibility(const ParamPtr& other) const {
    auto graph_param = std::dynamic_pointer_cast<GraphDataCellParameter>(other);
    if (not graph_param) {
        logger::error(
            "GraphDataCellParameter::CheckCompatibility: other parameter is not a "
            "GraphDataCellParameter");
        return false;
    }
    if (max_degree_ != graph_param->max_degree_) {
        logger::error("GraphDataCellParameter::CheckCompatibility: max_degree_ mismatch: {} vs {}",
                      max_degree_,
                      graph_param->max_degree_);
        return false;
    }
    if (support_remove_ != graph_param->support_remove_) {
        logger::error(
            "GraphDataCellParameter::CheckCompatibility: support_remove_ mismatch: {} vs {}",
            support_remove_,
            graph_param->support_remove_);
        return false;
    }
    if (remove_flag_bit_ != graph_param->remove_flag_bit_) {
        logger::error(
            "GraphDataCellParameter::CheckCompatibility: remove_flag_bit_ mismatch: {} vs {}",
            remove_flag_bit_,
            graph_param->remove_flag_bit_);
        return false;
    }
    return true;
}

}  // namespace vsag
