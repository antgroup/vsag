
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

#include <memory>
#include <string>

#include "algorithm/hnswlib/hnswlib.h"
#include "data_type.h"
#include "index_common_param.h"

namespace vsag {

/**
 * @brief Configuration parameters for HNSW index construction.
 *
 * This structure contains all parameters needed to configure an HNSW index,
 * including graph structure settings, distance metric, and optimization options.
 */
struct HnswParameters {
public:
    /**
     * @brief Creates HnswParameters from JSON configuration.
     *
     * @param hnsw_param_obj JSON object containing HNSW parameters.
     * @param index_common_param Common index parameters.
     * @return HnswParameters Parsed configuration structure.
     */
    static HnswParameters
    FromJson(const JsonType& hnsw_param_obj, const IndexCommonParam& index_common_param);

public:
    std::shared_ptr<hnswlib::SpaceInterface> space;  ///< Distance metric space
    int64_t max_degree;                              ///< Maximum degree of each node in the graph
    int64_t ef_construction;                         ///< Beam width during construction
    bool use_conjugate_graph{false};                 ///< Enable conjugate graph optimization
    bool normalize{false};                           ///< Normalize vectors before indexing
    bool use_reversed_edges{false};                  ///< Enable reversed edges tracking
    DataTypes type{DataTypes::DATA_TYPE_FLOAT};      ///< Data type of vectors

protected:
    HnswParameters() = default;
};

/**
 * @brief Configuration parameters for fresh HNSW index construction.
 *
 * Specialized parameters for creating a new HNSW index from scratch.
 */
struct FreshHnswParameters : public HnswParameters {
public:
    /**
     * @brief Creates FreshHnswParameters from JSON configuration.
     *
     * @param hnsw_param_obj JSON object containing HNSW parameters.
     * @param index_common_param Common index parameters.
     * @return HnswParameters Parsed configuration structure.
     */
    static HnswParameters
    FromJson(const JsonType& hnsw_param_obj, const IndexCommonParam& index_common_param);

private:
    FreshHnswParameters() = default;
};

/**
 * @brief Configuration parameters for HNSW search operations.
 *
 * This structure contains parameters that control search behavior,
 * including beam width and optimization settings.
 */
struct HnswSearchParameters {
public:
    /**
     * @brief Creates HnswSearchParameters from JSON string.
     *
     * @param json_string JSON string containing search parameters.
     * @return HnswSearchParameters Parsed search configuration.
     */
    static HnswSearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t ef_search;                ///< Beam width during search
    float skip_ratio{0.9};            ///< Skip ratio for early termination
    bool use_conjugate_graph_search;  ///< Enable conjugate graph during search

private:
    HnswSearchParameters() = default;
};

}  // namespace vsag
