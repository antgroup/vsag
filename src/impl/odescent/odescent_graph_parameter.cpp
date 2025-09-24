
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

#include "odescent_graph_parameter.h"

#include <fmt/format.h>

#include "vsag/constants.h"

namespace vsag {

void
ODescentParameter::FromJson(const vsag::JsonType& json) {
    CHECK_ARGUMENT(json.Contains(HGRAPH_GRAPH_MAX_DEGREE),
                   fmt::format("odescent parameters must contains {}", HGRAPH_GRAPH_MAX_DEGREE));
    max_degree = json[HGRAPH_GRAPH_MAX_DEGREE].GetInt();
    if (json.Contains(ODESCENT_PARAMETER_ALPHA)) {
        alpha = json[ODESCENT_PARAMETER_ALPHA].GetFloat();
    }
    if (json.Contains(ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE)) {
        sample_rate = json[ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE].GetFloat();
    }
    if (json.Contains(ODESCENT_PARAMETER_GRAPH_ITER_TURN)) {
        turn = json[ODESCENT_PARAMETER_GRAPH_ITER_TURN].GetInt();
    }
    if (json.Contains(ODESCENT_PARAMETER_MIN_IN_DEGREE)) {
        min_in_degree = json[ODESCENT_PARAMETER_MIN_IN_DEGREE].GetInt();
    }
    if (json.Contains(ODESCENT_PARAMETER_BUILD_BLOCK_SIZE)) {
        block_size = json[ODESCENT_PARAMETER_BUILD_BLOCK_SIZE].GetInt();
    }
}

JsonType
ODescentParameter::ToJson() const {
    JsonType json;
    json[ODESCENT_PARAMETER_ALPHA].SetFloat(alpha);
    json[ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE].SetFloat(sample_rate);
    json[ODESCENT_PARAMETER_GRAPH_ITER_TURN].SetInt(turn);
    json[HGRAPH_GRAPH_MAX_DEGREE].SetInt(max_degree);
    json[ODESCENT_PARAMETER_MIN_IN_DEGREE].SetInt(min_in_degree);
    json[ODESCENT_PARAMETER_BUILD_BLOCK_SIZE].SetInt(block_size);
    return json;
}

}  // namespace vsag
