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

#include <nlohmann/json.hpp>

#include "common.h"
#include "hgraph.h"  // IWYU pragma: keep
#include "hgraph_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "quantization/rabitq_quantization/rabitq_quantizer_parameter.h"
#include "quantization/transform_quantization/transform_quantizer_parameter.h"

namespace vsag {

namespace {

JsonType
build_default_io_param(const std::string& io_type = IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
    return IOParameter::CreateDefault(io_type)->ToJson();
}

JsonType
build_default_quantization_param(const std::string& quantization_type,
                                 const std::string& tq_chain,
                                 bool hold_molds) {
    if (quantization_type == QUANTIZATION_TYPE_VALUE_TQ) {
        return TransformQuantizerParameter::CreateDefault(tq_chain)->ToJson();
    }
    auto parameter = QuantizerParameter::CreateDefault(quantization_type);
    if (auto fp32 = std::dynamic_pointer_cast<FP32QuantizerParameter>(parameter); fp32 != nullptr) {
        fp32->hold_molds = hold_molds;
    }
    return parameter->ToJson();
}

JsonType
build_default_flatten_param(const std::string& quantization_type,
                            const std::string& io_type,
                            const std::string& tq_chain = "",
                            bool hold_molds = false) {
    auto parameter = FlattenDataCellParameter::CreateDefault(
        quantization_type == QUANTIZATION_TYPE_VALUE_TQ ? QUANTIZATION_TYPE_VALUE_FP32
                                                        : quantization_type,
        io_type,
        hold_molds);
    if (quantization_type == QUANTIZATION_TYPE_VALUE_TQ) {
        parameter->quantizer_parameter = TransformQuantizerParameter::CreateDefault(tq_chain);
    }
    return parameter->ToJson();
}

JsonType
build_default_hgraph_inner_param(const JsonType& external_json) {
    const auto base_quantization_type =
        external_json.Contains(HGRAPH_BASE_QUANTIZATION_TYPE)
            ? external_json[HGRAPH_BASE_QUANTIZATION_TYPE].GetString()
            : std::string(QUANTIZATION_TYPE_VALUE_FP32);
    const auto precise_quantization_type =
        external_json.Contains(HGRAPH_PRECISE_QUANTIZATION_TYPE)
            ? external_json[HGRAPH_PRECISE_QUANTIZATION_TYPE].GetString()
            : std::string(QUANTIZATION_TYPE_VALUE_FP32);
    const auto base_io_type = external_json.Contains(HGRAPH_BASE_IO_TYPE)
                                  ? external_json[HGRAPH_BASE_IO_TYPE].GetString()
                                  : std::string(IO_TYPE_VALUE_BLOCK_MEMORY_IO);
    const auto precise_io_type = external_json.Contains(HGRAPH_PRECISE_IO_TYPE)
                                     ? external_json[HGRAPH_PRECISE_IO_TYPE].GetString()
                                     : std::string(IO_TYPE_VALUE_BLOCK_MEMORY_IO);
    const auto graph_io_type = external_json.Contains(HGRAPH_GRAPH_IO_TYPE)
                                   ? external_json[HGRAPH_GRAPH_IO_TYPE].GetString()
                                   : std::string(IO_TYPE_VALUE_BLOCK_MEMORY_IO);
    const auto tq_chain = external_json.Contains(INDEX_TQ_CHAIN)
                              ? external_json[INDEX_TQ_CHAIN].GetString()
                              : std::string();
    JsonType json;
    json[TYPE_KEY].SetString(INDEX_TYPE_HGRAPH);
    json[USE_REORDER_KEY].SetBool(false);
    json[HGRAPH_USE_ELP_OPTIMIZER_KEY].SetBool(false);
    json[HGRAPH_IGNORE_REORDER_KEY].SetBool(false);
    json[HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY].SetBool(false);
    json[RESIZE_INCREASE_COUNT_BIT].SetUint64(DEFAULT_RESIZE_INCREASE_COUNT_BIT);
    json[USE_ATTRIBUTE_FILTER_KEY].SetBool(false);

    JsonType graph;
    graph[IO_PARAMS_KEY].SetJson(build_default_io_param(graph_io_type));
    graph[HGRAPH_USE_REVERSE_EDGES_KEY].SetBool(false);
    graph[GRAPH_TYPE_KEY].SetString(GRAPH_TYPE_VALUE_NSW);
    graph[GRAPH_STORAGE_TYPE_KEY].SetString(GRAPH_STORAGE_TYPE_VALUE_FLAT);
    graph[ODESCENT_PARAMETER_BUILD_BLOCK_SIZE].SetInt(10000);
    graph[ODESCENT_PARAMETER_MIN_IN_DEGREE].SetInt(1);
    graph[ODESCENT_PARAMETER_ALPHA].SetFloat(1.2F);
    graph[ODESCENT_PARAMETER_GRAPH_ITER_TURN].SetInt(30);
    graph[ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE].SetFloat(0.2F);
    graph[GRAPH_PARAM_MAX_DEGREE_KEY].SetInt(64);
    graph[GRAPH_PARAM_INIT_MAX_CAPACITY_KEY].SetInt(100);
    graph[GRAPH_SUPPORT_REMOVE].SetBool(false);
    graph[REMOVE_FLAG_BIT].SetInt(8);
    graph[SUPPORT_DUPLICATE].SetBool(false);
    json[GRAPH_KEY].SetJson(graph);

    json[BASE_CODES_KEY].SetJson(
        build_default_flatten_param(base_quantization_type, base_io_type, tq_chain));
    json[PRECISE_CODES_KEY].SetJson(
        build_default_flatten_param(precise_quantization_type, precise_io_type));
    json[RAW_VECTOR_KEY].SetJson(build_default_flatten_param(
        QUANTIZATION_TYPE_VALUE_FP32, IO_TYPE_VALUE_BLOCK_MEMORY_IO, "", true));
    json[STORE_RAW_VECTOR_KEY].SetBool(false);
    json[LABEL_REMAP_TYPE_KEY].SetString(LABEL_REMAP_TYPE_VALUE_PG);
    json[BUILD_THREAD_COUNT_KEY].SetInt(100);

    JsonType extra_info;
    extra_info[IO_PARAMS_KEY].SetJson(build_default_io_param());
    json[EXTRA_INFO_KEY].SetJson(extra_info);
    JsonType attr;
    attr[ATTR_HAS_BUCKETS_KEY].SetBool(false);
    json[ATTR_PARAMS_KEY].SetJson(attr);

    json[SUPPORT_DUPLICATE].SetBool(false);
    json[DEDUPLICATE_STORAGE].SetBool(false);
    json[SUPPORT_FORCE_REMOVE].SetBool(false);
    json[HGRAPH_PERSIST_SOURCE_ID_KEY].SetBool(false);
    json[EF_CONSTRUCTION_KEY].SetInt(400);
    return json;
}

}  // namespace

JsonType
HGraph::map_hgraph_param(const JsonType& hgraph_json) {
    CHECK_ARGUMENT(not hgraph_json.Contains(HGRAPH_MCI_KEY),
                   "nested hgraph mci parameters are not supported; use flat mci_* parameters");
    auto inner_json = build_default_hgraph_inner_param(hgraph_json);
    for (const auto& [key, ignored] : hgraph_json.GetInnerJson()->items()) {
        (void)ignored;
        auto value = hgraph_json[key];
        if (key == HGRAPH_USE_REORDER) {
            inner_json[USE_REORDER_KEY].SetJson(value);
        } else if (key == HGRAPH_REORDER_SOURCE) {
            inner_json[REORDER_SOURCE_KEY].SetJson(value);
        } else if (key == HGRAPH_USE_ELP_OPTIMIZER) {
            inner_json[HGRAPH_USE_ELP_OPTIMIZER_KEY].SetJson(value);
        } else if (key == HGRAPH_USE_REVERSE_EDGES) {
            inner_json[GRAPH_KEY][HGRAPH_USE_REVERSE_EDGES_KEY].SetJson(value);
        } else if (key == HGRAPH_IGNORE_REORDER) {
            inner_json[HGRAPH_IGNORE_REORDER_KEY].SetJson(value);
        } else if (key == HGRAPH_BUILD_BY_BASE_QUANTIZATION) {
            inner_json[HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY].SetJson(value);
        } else if (key == USE_ATTRIBUTE_FILTER) {
            inner_json[USE_ATTRIBUTE_FILTER_KEY].SetJson(value);
        } else if (key == HGRAPH_BASE_QUANTIZATION_TYPE) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == STORE_RAW_VECTOR) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][HOLD_MOLDS].SetJson(value);
            inner_json[PRECISE_CODES_KEY][QUANTIZATION_PARAMS_KEY][HOLD_MOLDS].SetJson(value);
            inner_json[STORE_RAW_VECTOR_KEY].SetJson(value);
        } else if (key == HGRAPH_BASE_IO_TYPE) {
            inner_json[BASE_CODES_KEY][IO_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == HGRAPH_BASE_SUPPLEMENT_IO_TYPE) {
            inner_json[BASE_CODES_KEY][SUPPLEMENT_IO_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == HGRAPH_BASE_SUPPLEMENT_FILE_PATH) {
            inner_json[BASE_CODES_KEY][SUPPLEMENT_IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetJson(value);
        } else if (key == HGRAPH_PRECISE_IO_TYPE) {
            inner_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == HGRAPH_BASE_FILE_PATH) {
            inner_json[BASE_CODES_KEY][IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetJson(value);
        } else if (key == HGRAPH_BASE_DIRECT_READ) {
            inner_json[BASE_CODES_KEY][IO_PARAMS_KEY][IO_DIRECT_READ_KEY].SetJson(value);
        } else if (key == HGRAPH_BASE_CACHE_TOTAL_SIZE) {
            inner_json[BASE_CODES_KEY][IO_PARAMS_KEY][READ_CACHE_TOTAL_CACHE_SIZE_KEY].SetJson(
                value);
        } else if (key == HGRAPH_PRECISE_FILE_PATH) {
            inner_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetJson(value);
        } else if (key == HGRAPH_PRECISE_DIRECT_READ) {
            inner_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][IO_DIRECT_READ_KEY].SetJson(value);
        } else if (key == HGRAPH_PRECISE_CACHE_TOTAL_SIZE) {
            inner_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][READ_CACHE_TOTAL_CACHE_SIZE_KEY].SetJson(
                value);
        } else if (key == HGRAPH_PRECISE_QUANTIZATION_TYPE) {
            inner_json[PRECISE_CODES_KEY][QUANTIZATION_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == HGRAPH_GRAPH_IO_TYPE) {
            inner_json[GRAPH_KEY][IO_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == HGRAPH_GRAPH_FILE_PATH) {
            inner_json[GRAPH_KEY][IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetJson(value);
        } else if (key == HGRAPH_GRAPH_CACHE_TOTAL_SIZE) {
            inner_json[GRAPH_KEY][IO_PARAMS_KEY][READ_CACHE_TOTAL_CACHE_SIZE_KEY].SetJson(value);
        } else if (key == RAW_VECTOR_IO_TYPE) {
            inner_json[RAW_VECTOR_KEY][IO_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == RAW_VECTOR_FILE_PATH) {
            inner_json[RAW_VECTOR_KEY][IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetJson(value);
        } else if (key == HGRAPH_GRAPH_MAX_DEGREE) {
            inner_json[GRAPH_KEY][GRAPH_PARAM_MAX_DEGREE_KEY].SetJson(value);
        } else if (key == HGRAPH_BUILD_EF_CONSTRUCTION) {
            inner_json[EF_CONSTRUCTION_KEY].SetJson(value);
        } else if (key == HGRAPH_BUILD_ALPHA) {
            inner_json[ALPHA_KEY].SetJson(value);
        } else if (key == HGRAPH_INIT_CAPACITY) {
            inner_json[GRAPH_KEY][GRAPH_PARAM_INIT_MAX_CAPACITY_KEY].SetJson(value);
        } else if (key == RESIZE_INCREASE_COUNT_BIT) {
            inner_json[RESIZE_INCREASE_COUNT_BIT].SetJson(value);
        } else if (key == HGRAPH_GRAPH_TYPE) {
            inner_json[GRAPH_KEY][GRAPH_TYPE_KEY].SetJson(value);
        } else if (key == HGRAPH_GRAPH_STORAGE_TYPE) {
            inner_json[GRAPH_KEY][GRAPH_STORAGE_TYPE_KEY].SetJson(value);
        } else if (key == ODESCENT_PARAMETER_ALPHA) {
            inner_json[GRAPH_KEY][ODESCENT_PARAMETER_ALPHA].SetJson(value);
        } else if (key == ODESCENT_PARAMETER_GRAPH_ITER_TURN) {
            inner_json[GRAPH_KEY][ODESCENT_PARAMETER_GRAPH_ITER_TURN].SetJson(value);
        } else if (key == ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE) {
            inner_json[GRAPH_KEY][ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE].SetJson(value);
        } else if (key == ODESCENT_PARAMETER_MIN_IN_DEGREE) {
            inner_json[GRAPH_KEY][ODESCENT_PARAMETER_MIN_IN_DEGREE].SetJson(value);
        } else if (key == ODESCENT_PARAMETER_BUILD_BLOCK_SIZE) {
            inner_json[GRAPH_KEY][ODESCENT_PARAMETER_BUILD_BLOCK_SIZE].SetJson(value);
        } else if (key == HGRAPH_BUILD_THREAD_COUNT) {
            inner_json[BUILD_THREAD_COUNT_KEY].SetJson(value);
        } else if (key == SQ4_UNIFORM_TRUNC_RATE) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY]
                      [SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY]
                          .SetJson(value);
        } else if (key == RABITQ_PCA_DIM) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][PCA_DIM_KEY].SetJson(value);
        } else if (key == INDEX_TQ_CHAIN) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][TQ_CHAIN_KEY].SetJson(value);
        } else if (key == INDEX_MRLE_DIM) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][MRLE_DIM_KEY].SetJson(value);
        } else if (key == RABITQ_BITS_PER_DIM_QUERY) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY]
                      [RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY]
                          .SetJson(value);
        } else if (key == RABITQ_BITS_PER_DIM_BASE) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY]
                      [RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY]
                          .SetJson(value);
        } else if (key == RABITQ_BITS_PER_DIM_PRECISE) {
            inner_json[PRECISE_CODES_KEY][QUANTIZATION_PARAMS_KEY]
                      [RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY]
                          .SetJson(value);
        } else if (key == RABITQ_ERROR_RATE) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][RABITQ_QUANTIZATION_ERROR_RATE_KEY]
                .SetJson(value);
        } else if (key == HGRAPH_BASE_PQ_DIM) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][PRODUCT_QUANTIZATION_DIM_KEY]
                .SetJson(value);
        } else if (key == RABITQ_USE_FHT) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][USE_FHT_KEY].SetJson(value);
        } else if (key == FAST_ENCODE_RABITQ) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][FAST_ENCODE_RABITQ_KEY].SetJson(
                value);
            inner_json[PRECISE_CODES_KEY][QUANTIZATION_PARAMS_KEY][FAST_ENCODE_RABITQ_KEY].SetJson(
                value);
        } else if (key == FAST_ENCODE_RABITQ_ROUNDS) {
            inner_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][FAST_ENCODE_RABITQ_ROUNDS_KEY]
                .SetJson(value);
            inner_json[PRECISE_CODES_KEY][QUANTIZATION_PARAMS_KEY][FAST_ENCODE_RABITQ_ROUNDS_KEY]
                .SetJson(value);
        } else if (key == HGRAPH_SUPPORT_REMOVE) {
            inner_json[GRAPH_KEY][GRAPH_SUPPORT_REMOVE].SetJson(value);
        } else if (key == HGRAPH_SUPPORT_FORCE_REMOVE) {
            inner_json[SUPPORT_FORCE_REMOVE].SetJson(value);
        } else if (key == HGRAPH_REMOVE_FLAG_BIT) {
            inner_json[GRAPH_KEY][REMOVE_FLAG_BIT].SetJson(value);
        } else if (key == HGRAPH_SUPPORT_DUPLICATE) {
            inner_json[SUPPORT_DUPLICATE].SetJson(value);
            inner_json[GRAPH_KEY][SUPPORT_DUPLICATE].SetJson(value);
        } else if (key == HGRAPH_DEDUPLICATE_STORAGE) {
            inner_json[DEDUPLICATE_STORAGE].SetJson(value);
        } else if (key == HGRAPH_DUPLICATE_DISTANCE_THRESHOLD) {
            inner_json[DUPLICATE_DISTANCE_THRESHOLD].SetJson(value);
        } else if (key == HGRAPH_PERSIST_SOURCE_ID) {
            inner_json[HGRAPH_PERSIST_SOURCE_ID_KEY].SetJson(value);
        } else if (key == HGRAPH_LABEL_REMAP_TYPE) {
            inner_json[LABEL_REMAP_TYPE_KEY].SetJson(value);
        } else if (key == HGRAPH_USE_MCI || key == HGRAPH_MCI_MCS || key == HGRAPH_MCI_CLIQUE_MAX ||
                   key == HGRAPH_MCI_ALPHA || key == HGRAPH_MCI_KNNG_SOURCE ||
                   key == HGRAPH_MCI_INCREMENTAL_JOIN_RATIO_THRESHOLD_KEY ||
                   key == HGRAPH_MCI_INCREMENTAL_ADDED_MCT_KEY ||
                   key == HGRAPH_MCI_INCREMENTAL_CLIQUE_MAX_KEY) {
            inner_json[key].SetJson(value);
        } else if (key == HGRAPH_BASE_ENABLE_READ_CACHE) {
            inner_json[BASE_CODES_KEY][IO_PARAMS_KEY][READ_CACHE_ENABLED_KEY].SetJson(value);
        } else if (key == HGRAPH_PRECISE_ENABLE_READ_CACHE) {
            inner_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][READ_CACHE_ENABLED_KEY].SetJson(value);
        } else if (key == HGRAPH_GRAPH_ENABLE_READ_CACHE) {
            inner_json[GRAPH_KEY][IO_PARAMS_KEY][READ_CACHE_ENABLED_KEY].SetJson(value);
        } else if (key == HGRAPH_RAW_VECTOR_ENABLE_READ_CACHE) {
            inner_json[RAW_VECTOR_KEY][IO_PARAMS_KEY][READ_CACHE_ENABLED_KEY].SetJson(value);
        } else if (key == HGRAPH_RAW_VECTOR_CACHE_TOTAL_SIZE) {
            inner_json[RAW_VECTOR_KEY][IO_PARAMS_KEY][READ_CACHE_TOTAL_CACHE_SIZE_KEY].SetJson(
                value);
        } else {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("invalid config param: {}", key));
        }
    }
    ApplyRaBitQSplitConfig(ParseRaBitQSplitConfig(hgraph_json), inner_json);
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

    ValidateMRLEDim(external_param, common_param.dim_);
    if (RequiresRawVectorForMRLERaBitQSplit(inner_json)) {
        inner_json[STORE_RAW_VECTOR_KEY].SetBool(true);
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
