/**
 * @file graph_interface_parameter.h
 * @brief Parameter class for graph interface configuration.
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

#include "parameter.h"
#include "utils/pointer_define.h"
namespace vsag {
DEFINE_POINTER2(GraphInterfaceParam, GraphInterfaceParameter);

/**
 * @brief Enumeration of graph storage types.
 */
enum class GraphStorageTypes {
    GRAPH_STORAGE_TYPE_VALUE_FLAT = 0,        ///< Flat graph storage format
    GRAPH_STORAGE_TYPE_VALUE_COMPRESSED = 1,  ///< Compressed graph storage format
    GRAPH_STORAGE_TYPE_SPARSE = 2             ///< Sparse graph storage format
};

/**
 * @brief Parameter class for graph interface configuration.
 *
 * This class provides configuration parameters for graph-based index structures,
 * including storage type, maximum degree, and duplicate handling options.
 */
class GraphInterfaceParameter : public Parameter {
public:
    /**
     * @brief Creates a graph parameter instance based on JSON configuration.
     *
     * @param graph_type The type of graph storage to use.
     * @param json JSON configuration object.
     * @return Shared pointer to the created GraphInterfaceParameter.
     */
    static GraphInterfaceParamPtr
    GetGraphParameterByJson(GraphStorageTypes graph_type, const JsonType& json);

public:
    GraphStorageTypes graph_storage_type_{GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT};

    uint64_t max_degree_{64};  ///< Maximum degree of graph nodes

    bool use_reverse_edges_{false};  ///< Whether to store reverse edges
    bool support_duplicate_{false};  ///< Whether to support duplicate IDs

protected:
    /**
     * @brief Constructs a GraphInterfaceParameter with the specified graph type.
     *
     * @param graph_type The graph storage type for this parameter.
     */
    explicit GraphInterfaceParameter(GraphStorageTypes graph_type)
        : graph_storage_type_(graph_type){};
};

}  // namespace vsag