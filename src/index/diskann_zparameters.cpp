
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

#include "index_common_param.h"

namespace vsag {

CreateDiskannParameters
CreateDiskannParameters::FromJson(const std::string& json_string) {
    JsonType params = JsonType::parse(json_string);
    CreateDiskannParameters obj;

    auto index_common_param = IndexCommonParam::CheckAndCreate(json_string);

    CHECK_ARGUMENT(
        index_common_param.data_type_ == DataTypes::DATA_TYPE_FLOAT,
        fmt::format("parameters[{}] supports {} only now", PARAMETER_DTYPE, DATATYPE_FLOAT32));

    // set obj.dim
    obj.dim = index_common_param.dim_;

    // set obj.dtype
    obj.dtype = params[PARAMETER_DTYPE];

    // set obj.metric
    if (index_common_param.metric_ == MetricType::METRIC_TYPE_L2SQR) {
        obj.metric = diskann::Metric::L2;
    } else if (index_common_param.metric_ == MetricType::METRIC_TYPE_IP) {
        obj.metric = diskann::Metric::INNER_PRODUCT;
    } else if (params[PARAMETER_METRIC_TYPE] == METRIC_COSINE) {
        obj.metric = diskann::Metric::COSINE;
    } else {
        std::string metric = params[PARAMETER_METRIC_TYPE];
        throw std::invalid_argument(fmt::format("parameters[{}] must in [{}, {}, {}], now is {}",
                                                PARAMETER_METRIC_TYPE,
                                                METRIC_L2,
                                                METRIC_IP,
                                                METRIC_COSINE,
                                                metric));
    }

    CHECK_ARGUMENT(params.contains(INDEX_DISKANN),
                   fmt::format("parameters must contains {}", INDEX_DISKANN));

    // set obj.max_degree
    CHECK_ARGUMENT(
        params[INDEX_DISKANN].contains(DISKANN_PARAMETER_R),
        fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_R));
    obj.max_degree = params[INDEX_DISKANN][DISKANN_PARAMETER_R];
    CHECK_ARGUMENT((5 <= obj.max_degree) and (obj.max_degree <= 128),
                   fmt::format("max_degree({}) must in range[5, 128]", obj.max_degree));

    // set obj.ef_construction
    CHECK_ARGUMENT(
        params[INDEX_DISKANN].contains(DISKANN_PARAMETER_L),
        fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_L));
    obj.ef_construction = params[INDEX_DISKANN][DISKANN_PARAMETER_L];
    CHECK_ARGUMENT((obj.max_degree <= obj.ef_construction) and (obj.ef_construction <= 1000),
                   fmt::format("ef_construction({}) must in range[$max_degree({}), 64]",
                               obj.ef_construction,
                               obj.max_degree));

    // set obj.pq_dims
    CHECK_ARGUMENT(
        params[INDEX_DISKANN].contains(DISKANN_PARAMETER_DISK_PQ_DIMS),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_DISK_PQ_DIMS));
    obj.pq_dims = params[INDEX_DISKANN][DISKANN_PARAMETER_DISK_PQ_DIMS];

    // set obj.pq_sample_rate
    CHECK_ARGUMENT(
        params[INDEX_DISKANN].contains(DISKANN_PARAMETER_P_VAL),
        fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_P_VAL));
    obj.pq_sample_rate = params[INDEX_DISKANN][DISKANN_PARAMETER_P_VAL];

    // optional
    // set obj.use_preload
    if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_PRELOAD)) {
        obj.use_preload = params[INDEX_DISKANN][DISKANN_PARAMETER_PRELOAD];
    }
    // set obj.use_reference
    if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_USE_REFERENCE)) {
        obj.use_reference = params[INDEX_DISKANN][DISKANN_PARAMETER_USE_REFERENCE];
    }
    // set obj.use_opq
    if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_USE_OPQ)) {
        obj.use_opq = params[INDEX_DISKANN][DISKANN_PARAMETER_USE_OPQ];
    }

    // set obj.use_bsa
    if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_USE_BSA)) {
        obj.use_bsa = params[INDEX_DISKANN][DISKANN_PARAMETER_USE_BSA];
    }

    // set obj.use_async_io
    if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_USE_ASYNC_IO)) {
        obj.use_async_io = params[INDEX_DISKANN][DISKANN_PARAMETER_USE_ASYNC_IO];
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
    CHECK_ARGUMENT((1 <= obj.beam_search) and (obj.beam_search <= 30),
                   fmt::format("beam_search({}) must in range[1, 30]", obj.beam_search));

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

    return obj;
}

}  // namespace vsag
