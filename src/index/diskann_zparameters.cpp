
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

#include "diskann_zparameters.h"

#include <fmt/format.h>

#include "common.h"
#include "index_common_param.h"
#include "vsag/constants.h"

// NOLINTBEGIN(readability-simplify-boolean-expr)

namespace vsag {

DiskannParameters
DiskannParameters::FromJson(
    JsonType& diskann_param_obj,  // NOLINT(readability-function-cognitive-complexity)
    const IndexCommonParam& index_common_param) {
    DiskannParameters obj;

    CHECK_ARGUMENT(
        index_common_param.data_type_ == DataTypes::DATA_TYPE_FLOAT,
        fmt::format("parameters[{}] supports {} only now", PARAMETER_DTYPE, DATATYPE_FLOAT32));

    // set obj.dim
    obj.dim = index_common_param.dim_;

    // set obj.metric
    if (index_common_param.metric_ == MetricType::METRIC_TYPE_L2SQR) {
        obj.metric = diskann::Metric::L2;
    } else if (index_common_param.metric_ == MetricType::METRIC_TYPE_IP) {
        obj.metric = diskann::Metric::INNER_PRODUCT;
    } else if (index_common_param.metric_ == MetricType::METRIC_TYPE_COSINE) {
        obj.metric = diskann::Metric::COSINE;
    } else {
        throw std::invalid_argument(fmt::format("parameters[{}] must in [{}, {}, {}], now is {}",
                                                PARAMETER_METRIC_TYPE,
                                                METRIC_L2,
                                                METRIC_IP,
                                                METRIC_COSINE,
                                                (int)obj.metric));
    }

    // set obj.max_degree
    CHECK_ARGUMENT(
        diskann_param_obj.contains(DISKANN_PARAMETER_R),
        fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_R));
    obj.max_degree = diskann_param_obj[DISKANN_PARAMETER_R];
    CHECK_ARGUMENT((5 <= obj.max_degree) and (obj.max_degree <= 128),
                   fmt::format("max_degree({}) must in range[5, 128]", obj.max_degree));

    // set obj.pq_dims
    CHECK_ARGUMENT(
        diskann_param_obj.contains(DISKANN_PARAMETER_DISK_PQ_DIMS),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_DISK_PQ_DIMS));
    obj.pq_dims = diskann_param_obj[DISKANN_PARAMETER_DISK_PQ_DIMS];

    // set obj.pq_sample_rate
    CHECK_ARGUMENT(
        diskann_param_obj.contains(DISKANN_PARAMETER_P_VAL),
        fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_P_VAL));
    obj.pq_sample_rate = diskann_param_obj[DISKANN_PARAMETER_P_VAL];

    // optional
    // set obj.use_preload
    if (diskann_param_obj.contains(DISKANN_PARAMETER_PRELOAD)) {
        obj.use_preload = diskann_param_obj[DISKANN_PARAMETER_PRELOAD];
    }
    // set obj.use_reference
    if (diskann_param_obj.contains(DISKANN_PARAMETER_USE_REFERENCE)) {
        obj.use_reference = diskann_param_obj[DISKANN_PARAMETER_USE_REFERENCE];
    }
    // set obj.use_opq
    if (diskann_param_obj.contains(DISKANN_PARAMETER_USE_OPQ)) {
        obj.use_opq = diskann_param_obj[DISKANN_PARAMETER_USE_OPQ];
    }

    // set obj.use_bsa
    if (diskann_param_obj.contains(DISKANN_PARAMETER_USE_BSA)) {
        obj.use_bsa = diskann_param_obj[DISKANN_PARAMETER_USE_BSA];
    }

    // set obj.graph_type
    if (diskann_param_obj.contains(DISKANN_PARAMETER_GRAPH_TYPE)) {
        obj.graph_type = diskann_param_obj[DISKANN_PARAMETER_GRAPH_TYPE];
    }

    if (obj.graph_type == DISKANN_GRAPH_TYPE_VAMANA) {
        // set obj.ef_construction
        CHECK_ARGUMENT(
            diskann_param_obj.contains(DISKANN_PARAMETER_L),
            fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_L));
        obj.ef_construction = diskann_param_obj[DISKANN_PARAMETER_L];
        CHECK_ARGUMENT((obj.max_degree <= obj.ef_construction) and (obj.ef_construction <= 1000),
                       fmt::format("ef_construction({}) must in range[$max_degree({}), 64]",
                                   obj.ef_construction,
                                   obj.max_degree));
    } else if (obj.graph_type == GRAPH_TYPE_ODESCENT) {
        // set obj.alpha
        if (diskann_param_obj.contains(ODESCENT_PARAMETER_ALPHA)) {
            obj.alpha = diskann_param_obj[ODESCENT_PARAMETER_ALPHA];
            CHECK_ARGUMENT(
                (obj.alpha >= 1.0 && obj.alpha <= 2.0),
                fmt::format(
                    "{} must in range[1.0, 2.0], now is {}", ODESCENT_PARAMETER_ALPHA, obj.alpha));
        }
        // set obj.turn
        if (diskann_param_obj.contains(ODESCENT_PARAMETER_GRAPH_ITER_TURN)) {
            obj.turn = diskann_param_obj[ODESCENT_PARAMETER_GRAPH_ITER_TURN];
            CHECK_ARGUMENT((obj.turn > 0),
                           fmt::format("{} must be greater than 0, now is {}",
                                       ODESCENT_PARAMETER_GRAPH_ITER_TURN,
                                       obj.turn));
        }
        // set obj.sample_rate
        if (diskann_param_obj.contains(ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE)) {
            obj.sample_rate = diskann_param_obj[ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE];
            CHECK_ARGUMENT((obj.sample_rate > 0.05 && obj.sample_rate < 0.5),
                           fmt::format("{} must in range[0.05, 0.5], now is {}",
                                       ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
                                       obj.sample_rate));
        }
    } else {
        throw std::invalid_argument(fmt::format("parameters[{}] must in [{}, {}], now is {}",
                                                DISKANN_PARAMETER_GRAPH_TYPE,
                                                DISKANN_GRAPH_TYPE_VAMANA,
                                                GRAPH_TYPE_ODESCENT,
                                                obj.graph_type));
    }
    return obj;
}

DiskannSearchParameters
DiskannSearchParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);

    DiskannSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.contains(INDEX_DISKANN),
                   fmt::format("parameters must contains {}", INDEX_DISKANN));
    CHECK_ARGUMENT(
        params[INDEX_DISKANN].contains(DISKANN_PARAMETER_EF_SEARCH),
        fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_EF_SEARCH));
    obj.ef_search = params[INDEX_DISKANN][DISKANN_PARAMETER_EF_SEARCH];
    CHECK_ARGUMENT((1 <= obj.ef_search) and (obj.ef_search <= 1000),
                   fmt::format("ef_search({}) must in range[1, 1000]", obj.ef_search));

    // set obj.beam_search
    CHECK_ARGUMENT(
        params[INDEX_DISKANN].contains(DISKANN_PARAMETER_BEAM_SEARCH),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_BEAM_SEARCH));
    obj.beam_search = params[INDEX_DISKANN][DISKANN_PARAMETER_BEAM_SEARCH];
    CHECK_ARGUMENT((1 <= obj.beam_search) and (obj.beam_search <= 64),
                   fmt::format("beam_search({}) must in range[1, 64]", obj.beam_search));

    // set obj.io_limit
    CHECK_ARGUMENT(
        params[INDEX_DISKANN].contains(DISKANN_PARAMETER_IO_LIMIT),
        fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_IO_LIMIT));
    obj.io_limit = params[INDEX_DISKANN][DISKANN_PARAMETER_IO_LIMIT];
    CHECK_ARGUMENT((1 <= obj.io_limit) and (obj.io_limit <= 512),
                   fmt::format("io_limit({}) must in range[1, 512]", obj.io_limit));

    // optional
    // set obj.use_reorder
    if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_REORDER)) {
        obj.use_reorder = params[INDEX_DISKANN][DISKANN_PARAMETER_REORDER];
    }

    // set obj.use_async_io
    if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_USE_ASYNC_IO)) {
        obj.use_async_io = params[INDEX_DISKANN][DISKANN_PARAMETER_USE_ASYNC_IO];
    }

    return obj;
}

}  // namespace vsag

// NOLINTEND(readability-simplify-boolean-expr)
