/**
 * @file graph_datacell_parameter.h
 * @brief Parameter class for graph data cell configuration.
 */

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
#include "io/io_parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(GraphDataCellParam, GraphDataCellParameter);

/**
 * @brief Parameter class for graph data cell configuration.
 *
 * This class extends GraphInterfaceParameter and provides configuration
 * for graph-based data cell storage with IO parameters and removal support.
 */
class GraphDataCellParameter : public GraphInterfaceParameter {
public:
    /**
     * @brief Constructs a GraphDataCellParameter with flat storage type.
     */
    GraphDataCellParameter()
        : GraphInterfaceParameter(GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
    }

    /**
     * @brief Loads parameters from JSON configuration.
     *
     * @param json JSON configuration object.
     */
    void
    FromJson(const JsonType& json) override;

    /**
     * @brief Exports parameters to JSON format.
     *
     * @return JSON object containing the parameter values.
     */
    JsonType
    ToJson() const override;

    /**
     * @brief Checks compatibility with another parameter.
     *
     * @param other Another parameter to compare with.
     * @return True if parameters are compatible, false otherwise.
     */
    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    IOParamPtr io_parameter_{nullptr};  ///< IO configuration parameter

    uint64_t init_max_capacity_{100};  ///< Initial maximum capacity

    bool support_remove_{false};   ///< Whether to support node removal
    uint32_t remove_flag_bit_{8};  ///< Number of bits for remove flag
};
}  // namespace vsag