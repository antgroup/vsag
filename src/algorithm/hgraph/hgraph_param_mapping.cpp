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

#include <fmt/format.h>

#include "common.h"
#include "hgraph.h"  // IWYU pragma: keep
#include "hgraph_parameter.h"
#include "quantization/rabitq_quantization/rabitq_quantizer_parameter.h"

namespace vsag {

namespace {

void
map_rabitq_split_param(const JsonType& external_json, JsonType& inner_json) {
    if (not external_json.Contains(RABITQ_BITS_PER_DIM_PRECISE)) {
        return;
    }

    CHECK_ARGUMENT(
        external_json.Contains(RABITQ_BITS_PER_DIM_BASE),
        fmt::format("{} requires {}", RABITQ_BITS_PER_DIM_PRECISE, RABITQ_BITS_PER_DIM_BASE));
    CHECK_ARGUMENT(
        external_json.Contains(HGRAPH_BASE_QUANTIZATION_TYPE),
        fmt::format("{} requires {}", RABITQ_BITS_PER_DIM_PRECISE, HGRAPH_BASE_QUANTIZATION_TYPE));
    CHECK_ARGUMENT(
        external_json.Contains(HGRAPH_PRECISE_QUANTIZATION_TYPE),
        fmt::format(
            "{} requires {}", RABITQ_BITS_PER_DIM_PRECISE, HGRAPH_PRECISE_QUANTIZATION_TYPE));

    const auto base_quantization_type = external_json[HGRAPH_BASE_QUANTIZATION_TYPE].GetString();
    const auto precise_quantization_type =
        external_json[HGRAPH_PRECISE_QUANTIZATION_TYPE].GetString();
    CHECK_ARGUMENT(base_quantization_type == QUANTIZATION_TYPE_VALUE_RABITQ,
                   fmt::format("{} requires {}={}",
                               RABITQ_BITS_PER_DIM_PRECISE,
                               HGRAPH_BASE_QUANTIZATION_TYPE,
                               QUANTIZATION_TYPE_VALUE_RABITQ));
    CHECK_ARGUMENT(precise_quantization_type == QUANTIZATION_TYPE_VALUE_RABITQ,
                   fmt::format("{} requires {}={}",
                               RABITQ_BITS_PER_DIM_PRECISE,
                               HGRAPH_PRECISE_QUANTIZATION_TYPE,
                               QUANTIZATION_TYPE_VALUE_RABITQ));

    const int64_t filter_bits = external_json[RABITQ_BITS_PER_DIM_BASE].GetInt();
    const int64_t supplement_bits = external_json[RABITQ_BITS_PER_DIM_PRECISE].GetInt();
    CHECK_ARGUMENT(
        filter_bits >= 1,
        fmt::format("{} must be in [1, 8], got {}", RABITQ_BITS_PER_DIM_BASE, filter_bits));
    CHECK_ARGUMENT(
        filter_bits <= 8,
        fmt::format("{} must be in [1, 8], got {}", RABITQ_BITS_PER_DIM_BASE, filter_bits));
    CHECK_ARGUMENT(
        supplement_bits >= 1,
        fmt::format("{} must be in [1, 8], got {}", RABITQ_BITS_PER_DIM_PRECISE, supplement_bits));
    CHECK_ARGUMENT(
        supplement_bits <= 8,
        fmt::format("{} must be in [1, 8], got {}", RABITQ_BITS_PER_DIM_PRECISE, supplement_bits));
    const int64_t total_bits = filter_bits + supplement_bits;
    CHECK_ARGUMENT(total_bits <= 8,
                   fmt::format("{} + {} must be no greater than 8, got {}",
                               RABITQ_BITS_PER_DIM_BASE,
                               RABITQ_BITS_PER_DIM_PRECISE,
                               total_bits));

    inner_json[REORDER_SOURCE_KEY].SetString(HGRAPH_REORDER_SOURCE_BASE);
    inner_json[BASE_CODES_KEY][CODES_TYPE_KEY].SetString(RABITQ_SPLIT_CODES);
    inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][RABITQ_QUANTIZATION_VERSION_KEY].SetString(
        RaBitQuantizerParameter::RABITQ_VERSION_SPLIT);
    inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY]
        .SetInt(32);
    inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][RABITQ_QUANTIZATION_BITS_PER_DIM_FILTER_KEY]
        .SetInt(filter_bits);
    inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY]
        .SetInt(total_bits);
}

}  // namespace

JsonType
HGraph::map_hgraph_param(const JsonType& hgraph_json) {
    static const ConstParamMap external_mapping = {
        {
            HGRAPH_USE_REORDER,
            {
                USE_REORDER_KEY,
            },
        },
        {
            HGRAPH_REORDER_SOURCE,
            {
                REORDER_SOURCE_KEY,
            },
        },
        {
            HGRAPH_USE_ELP_OPTIMIZER,
            {
                HGRAPH_USE_ELP_OPTIMIZER_KEY,
            },
        },
        {
            HGRAPH_USE_REVERSE_EDGES,
            {
                GRAPH_KEY,
                HGRAPH_USE_REVERSE_EDGES_KEY,
            },
        },
        {
            HGRAPH_IGNORE_REORDER,
            {
                HGRAPH_IGNORE_REORDER_KEY,
            },
        },
        {
            HGRAPH_BUILD_BY_BASE_QUANTIZATION,
            {
                HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY,
            },
        },
        {
            USE_ATTRIBUTE_FILTER,
            {
                USE_ATTRIBUTE_FILTER_KEY,
            },
        },
        {
            HGRAPH_BASE_QUANTIZATION_TYPE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                HOLD_MOLDS,
            },
        },
        {
            HGRAPH_BASE_IO_TYPE,
            {
                BASE_CODES_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_BASE_SUPPLEMENT_IO_TYPE,
            {
                BASE_CODES_KEY,
                SUPPLEMENT_IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_BASE_SUPPLEMENT_FILE_PATH,
            {
                BASE_CODES_KEY,
                SUPPLEMENT_IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_PRECISE_IO_TYPE,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_BASE_FILE_PATH,
            {
                BASE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_BASE_DIRECT_READ,
            {
                BASE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_DIRECT_READ_KEY,
            },
        },
        {
            HGRAPH_PRECISE_FILE_PATH,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_PRECISE_DIRECT_READ,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_DIRECT_READ_KEY,
            },
        },
        {
            HGRAPH_PRECISE_QUANTIZATION_TYPE,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_IO_TYPE,
            {
                GRAPH_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_FILE_PATH,
            {
                GRAPH_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                HOLD_MOLDS,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                STORE_RAW_VECTOR_KEY,
            },
        },
        {
            RAW_VECTOR_IO_TYPE,
            {
                RAW_VECTOR_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            RAW_VECTOR_FILE_PATH,
            {
                RAW_VECTOR_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_GRAPH_MAX_DEGREE,
            {
                GRAPH_KEY,
                GRAPH_PARAM_MAX_DEGREE_KEY,
            },
        },
        {
            HGRAPH_BUILD_EF_CONSTRUCTION,
            {
                EF_CONSTRUCTION_KEY,
            },
        },
        {
            HGRAPH_BUILD_ALPHA,
            {
                ALPHA_KEY,
            },
        },
        {
            HGRAPH_INIT_CAPACITY,
            {
                GRAPH_KEY,
                GRAPH_PARAM_INIT_MAX_CAPACITY_KEY,
            },
        },
        {
            HGRAPH_GRAPH_TYPE,
            {
                GRAPH_KEY,
                GRAPH_TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_STORAGE_TYPE,
            {
                GRAPH_KEY,
                GRAPH_STORAGE_TYPE_KEY,
            },
        },
        {
            ODESCENT_PARAMETER_ALPHA,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_ALPHA,
            },
        },
        {
            ODESCENT_PARAMETER_GRAPH_ITER_TURN,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_GRAPH_ITER_TURN,
            },
        },
        {
            ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
            },
        },
        {
            ODESCENT_PARAMETER_MIN_IN_DEGREE,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_MIN_IN_DEGREE,
            },
        },
        {
            ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
            },
        },
        {
            HGRAPH_BUILD_THREAD_COUNT,
            {
                BUILD_THREAD_COUNT_KEY,
            },
        },
        {
            SQ4_UNIFORM_TRUNC_RATE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY,
            },
        },
        {
            RABITQ_PCA_DIM,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                PCA_DIM_KEY,
            },
        },
        {
            INDEX_TQ_CHAIN,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TQ_CHAIN_KEY,
            },
        },
        {
            INDEX_MRLE_DIM,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                MRLE_DIM_KEY,
            },
        },
        {
            RABITQ_BITS_PER_DIM_QUERY,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY,
            },
        },
        {
            RABITQ_BITS_PER_DIM_BASE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY,
            },
        },
        {
            RABITQ_BITS_PER_DIM_PRECISE,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY,
            },
        },
        {
            RABITQ_ERROR_RATE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_ERROR_RATE_KEY,
            },
        },
        {
            HGRAPH_BASE_PQ_DIM,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                PRODUCT_QUANTIZATION_DIM_KEY,
            },
        },
        {
            RABITQ_USE_FHT,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                USE_FHT_KEY,
            },
        },
        {
            FAST_ENCODE_RABITQ,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                FAST_ENCODE_RABITQ_KEY,
            },
        },
        {
            FAST_ENCODE_RABITQ,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                FAST_ENCODE_RABITQ_KEY,
            },
        },
        {
            FAST_ENCODE_RABITQ_ROUNDS,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                FAST_ENCODE_RABITQ_ROUNDS_KEY,
            },
        },
        {
            FAST_ENCODE_RABITQ_ROUNDS,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                FAST_ENCODE_RABITQ_ROUNDS_KEY,
            },
        },
        {
            HGRAPH_SUPPORT_REMOVE,
            {GRAPH_KEY, GRAPH_SUPPORT_REMOVE},
        },
        {
            HGRAPH_SUPPORT_FORCE_REMOVE,
            {
                SUPPORT_FORCE_REMOVE,
            },
        },
        {
            HGRAPH_REMOVE_FLAG_BIT,
            {GRAPH_KEY, REMOVE_FLAG_BIT},
        },
        {
            HGRAPH_SUPPORT_DUPLICATE,
            {
                SUPPORT_DUPLICATE,
            },
        },
        {
            HGRAPH_DUPLICATE_DISTANCE_THRESHOLD,
            {
                DUPLICATE_DISTANCE_THRESHOLD,
            },
        },
        {
            HGRAPH_SUPPORT_DUPLICATE,
            {
                GRAPH_KEY,
                SUPPORT_DUPLICATE,
            },
        },
        {
            HGRAPH_PERSIST_SOURCE_ID,
            {
                HGRAPH_PERSIST_SOURCE_ID_KEY,
            },
        },
        {
            HGRAPH_LABEL_REMAP_TYPE,
            {
                LABEL_REMAP_TYPE_KEY,
            },
        }};
    const std::string hgraph_params_template =
        R"(
    {
        "{TYPE_KEY}": "{INDEX_TYPE_HGRAPH}",
        "{USE_REORDER_KEY}": false,
        "{HGRAPH_USE_ENV_OPTIMIZER}": false,
        "{HGRAPH_IGNORE_REORDER_KEY}": false,
        "{HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY}": false,
        "{HGRAPH_USE_ATTRIBUTE_FILTER_KEY}": false,
        "{GRAPH_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{HGRAPH_USE_REVERSE_EDGES_KEY}": false,
            "{GRAPH_TYPE_KEY}": "{GRAPH_TYPE_VALUE_NSW}",
            "{GRAPH_STORAGE_TYPE_KEY}": "{GRAPH_STORAGE_TYPE_VALUE_FLAT}",
            "{ODESCENT_PARAMETER_BUILD_BLOCK_SIZE}": 10000,
            "{ODESCENT_PARAMETER_MIN_IN_DEGREE}": 1,
            "{ODESCENT_PARAMETER_ALPHA}": 1.2,
            "{ODESCENT_PARAMETER_GRAPH_ITER_TURN}": 30,
            "{ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE}": 0.2,
            "{GRAPH_PARAM_MAX_DEGREE_KEY}": 64,
            "{GRAPH_PARAM_INIT_MAX_CAPACITY_KEY}": 100,
            "{GRAPH_SUPPORT_REMOVE}": false,
            "{REMOVE_FLAG_BIT}": 8,
            "{SUPPORT_DUPLICATE}": false
        },
        "{BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{RABITQ_QUANTIZATION_VERSION_KEY}": "standard",
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}": 32,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY}": 1,
                "{RABITQ_QUANTIZATION_ERROR_RATE_KEY}": 1.9,
                "{FAST_ENCODE_RABITQ_KEY}": true,
                "{FAST_ENCODE_RABITQ_ROUNDS_KEY}": 6,
                "{TQ_CHAIN_KEY}": "",
                "mrle_dim": 0,
                "nbits": 8,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{FAST_ENCODE_RABITQ_KEY}": true,
                "{FAST_ENCODE_RABITQ_ROUNDS_KEY}": 6,
                "{PCA_DIM_KEY}": 0,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{STORE_RAW_VECTOR_KEY}": false,
        "{LABEL_REMAP_TYPE_KEY}": "{LABEL_REMAP_TYPE_VALUE_PG}",
        "{RAW_VECTOR_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{HOLD_MOLDS}": true
            }
        },
        "{BUILD_THREAD_COUNT_KEY}": 100,
        "{EXTRA_INFO_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            }
        },
        "{ATTR_PARAMS_KEY}": {
            "{ATTR_HAS_BUCKETS_KEY}": false
        },
        "{HGRAPH_SUPPORT_DUPLICATE}": false,
        "{SUPPORT_FORCE_REMOVE}": false,
        "{HGRAPH_PERSIST_SOURCE_ID_KEY}": false,
        "{EF_CONSTRUCTION_KEY}": 400
    })";

    std::string str = format_map(hgraph_params_template, DEFAULT_MAP);
    auto inner_json = JsonType::Parse(str);
    mapping_external_param_to_inner(hgraph_json, external_mapping, inner_json);
    map_rabitq_split_param(hgraph_json, inner_json);

    return inner_json;
}

ParamPtr
HGraph::CheckAndMappingExternalParam(const JsonType& external_param,
                                     const IndexCommonParam& common_param) {
    auto inner_json = map_hgraph_param(external_param);
    if (common_param.data_type_ == DataTypes::DATA_TYPE_SPARSE) {
        inner_json[BASE_CODES_KEY][CODES_TYPE_KEY].SetString(SPARSE_CODES);
        inner_json[PRECISE_CODES_KEY][CODES_TYPE_KEY].SetString(SPARSE_CODES);
        inner_json[RAW_VECTOR_KEY][CODES_TYPE_KEY].SetString(SPARSE_CODES);
    }

    if (external_param.Contains(INDEX_MRLE_DIM)) {
        CHECK_ARGUMENT(external_param[INDEX_MRLE_DIM].IsNumberInteger(),
                       fmt::format("mrle_dim must be an integer, got {}",
                                   external_param[INDEX_MRLE_DIM].Dump()));
        int64_t mrle_dim = external_param[INDEX_MRLE_DIM].GetInt();
        bool valid_mrle_dim = mrle_dim >= 0 and mrle_dim <= static_cast<int64_t>(common_param.dim_);
        CHECK_ARGUMENT(
            valid_mrle_dim,
            fmt::format("mrle_dim({}) must be in range [0, {}]", mrle_dim, common_param.dim_));
    }

    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->data_type = common_param.data_type_;
    hgraph_parameter->FromJson(inner_json);
    uint64_t max_degree = hgraph_parameter->bottom_graph_param->max_degree_;

    auto max_degree_threshold = std::max<int64_t>(common_param.dim_, 128);
    CHECK_ARGUMENT(  // NOLINT
        (4 <= max_degree) and (max_degree <= max_degree_threshold),
        fmt::format("max_degree({}) must in range[4, {}]", max_degree, max_degree_threshold));

    auto construction_threshold = std::max<uint64_t>(1000UL, AMPLIFICATION_FACTOR * max_degree);
    CHECK_ARGUMENT((max_degree <= hgraph_parameter->ef_construction) and  // NOLINT
                       (hgraph_parameter->ef_construction <= construction_threshold),
                   fmt::format("ef_construction({}) must in range[$max_degree({}), {}]",
                               hgraph_parameter->ef_construction,
                               max_degree,
                               construction_threshold));

    return hgraph_parameter;
}

}  // namespace vsag
