
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

#include <distance.h>

#include <string>

#include "index_common_param.h"

namespace vsag {

/**
 * @brief Configuration parameters for DiskANN index construction.
 *
 * This structure contains all parameters needed to configure a DiskANN index,
 * including graph structure settings, PQ compression options, and I/O behavior.
 */
struct DiskannParameters {
public:
    /**
     * @brief Creates DiskannParameters from JSON configuration.
     *
     * @param diskann_param_obj JSON object containing DiskANN parameters.
     * @param index_common_param Common index parameters.
     * @return DiskannParameters Parsed configuration structure.
     */
    static DiskannParameters
    FromJson(const JsonType& diskann_param_obj, const IndexCommonParam& index_common_param);

public:
    int64_t dim{-1};                              ///< Vector dimension
    diskann::Metric metric{diskann::Metric::L2};  ///< Distance metric type
    int64_t max_degree{-1};                       ///< Maximum degree of each node
    int64_t ef_construction{-1};                  ///< Beam width during construction
    int64_t pq_dims{-1};                          ///< PQ code dimension
    float pq_sample_rate{.0f};                    ///< Sampling rate for PQ training

    bool use_preload = false;   ///< Preload index into memory
    bool use_reference = true;  ///< Use reference vectors
    bool use_opq = false;       ///< Use Optimized PQ
    bool use_bsa = false;       ///< Use BSA optimization
    bool use_async_io = false;  ///< Enable asynchronous I/O

    std::string graph_type = "vamana";  ///< Graph construction algorithm type
    float alpha = 1.2;                  ///< Alpha parameter for graph pruning
    int64_t turn = 40;                  ///< Number of rounds for graph construction
    float sample_rate = 0.3;            ///< Sampling rate for graph construction

    bool support_calc_distance_by_id = false;  ///< Support distance calculation by ID

private:
    DiskannParameters() = default;
};

/**
 * @brief Configuration parameters for DiskANN search operations.
 *
 * This structure contains parameters that control search behavior,
 * including beam width, I/O limits, and optimization settings.
 */
struct DiskannSearchParameters {
public:
    /**
     * @brief Creates DiskannSearchParameters from JSON string.
     *
     * @param json_string JSON string containing search parameters.
     * @return DiskannSearchParameters Parsed search configuration.
     */
    static DiskannSearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t ef_search{400};   ///< Beam width during search
    uint64_t beam_search{4};  ///< Number of parallel search paths
    int64_t io_limit{200};    ///< Maximum number of I/O operations

    bool use_reorder = false;   ///< Enable reordering of results
    bool use_async_io = false;  ///< Enable asynchronous I/O during search

private:
    DiskannSearchParameters() = default;
};

}  // namespace vsag
