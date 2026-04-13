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

#include "parameter.h"
#include "utils/pointer_define.h"

namespace vsag {

/**
 * @file odescent_graph_parameter.h
 * @brief Parameters for ODescent graph construction algorithm.
 */

DEFINE_POINTER(ODescentParameter);

/**
 * @brief Configuration parameters for the ODescent graph builder.
 *
 * ODescentParameter contains all configuration options for the ODescent
 * algorithm, which builds a proximity graph by iteratively optimizing
 * neighbor connections based on distance relationships.
 */
class ODescentParameter : public Parameter {
public:
    ODescentParameter() = default;

    /**
     * @brief Loads parameters from JSON configuration.
     *
     * @param json JSON object containing parameter values.
     */
    void
    FromJson(const JsonType& json) override;

    /**
     * @brief Exports parameters to JSON format.
     *
     * @return JsonType JSON object with current parameter values.
     */
    JsonType
    ToJson() const override;

public:
    /// Number of optimization iterations.
    int64_t turn{30};

    /// Alpha parameter controlling edge selection.
    float alpha{1};

    /// Sampling rate for candidate neighbor selection.
    float sample_rate{0.3};

    /// Minimum in-degree for each node.
    int64_t min_in_degree{1};

    /// Maximum degree for each node.
    int64_t max_degree{32};

    /// Block size for parallel processing.
    int64_t block_size{10000};
};

}  // namespace vsag