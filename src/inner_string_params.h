
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

#include <string>
#include <unordered_map>

#include "vsag/constants.h"

namespace vsag {
// Index Type
const char* const INDEX_TYPE_HGRAPH = "hgraph";
const char* const INDEX_TYPE_IVF = "ivf";
const char* const INDEX_TYPE_BRUTE_FORCE = "brute_force";
const char* const INDEX_TYPE_GNO_IMI = "gno_imi";
const char* const INDEX_TYPE_PYRAMID = "pyramid";

const char* const TYPE_KEY = "type";
const char* const USE_REORDER_KEY = "use_reorder";
const char* const EXTRA_INFO_KEY = "extra_info";
const char* const USE_ATTRIBUTE_FILTER_KEY = "use_attribute_filter";
const char* const BUILD_THREAD_COUNT_KEY = "build_thread_count";
const char* const BASE_CODES_KEY = "base_codes";
const char* const PRECISE_CODES_KEY = "precise_codes";
const char* const STORE_RAW_VECTOR_KEY = "store_raw_vector";
const char* const RAW_VECTOR_KEY = "raw_vector";
const char* const ATTR_HAS_BUCKETS_KEY = "has_buckets";
const char* const ATTR_PARAMS_KEY = "attr_params";

// Parameter key for hgraph
const char* const HGRAPH_USE_ELP_OPTIMIZER_KEY = "use_elp_optimizer";
const char* const HGRAPH_IGNORE_REORDER_KEY = "ignore_reorder";
const char* const HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY = "build_by_base";
const char* const GRAPH_KEY = "graph";
const char* const ALPHA_KEY = "alpha";

// IO param key
const char* const IO_PARAMS_KEY = "io_params";
const char* const IO_TYPE_VALUE_MEMORY_IO = "memory_io";
const char* const IO_TYPE_VALUE_BUFFER_IO = "buffer_io";
const char* const IO_TYPE_VALUE_MMAP_IO = "mmap_io";
const char* const IO_TYPE_VALUE_READER_IO = "reader_io";
const char* const IO_TYPE_VALUE_ASYNC_IO = "async_io";
const char* const IO_TYPE_VALUE_BLOCK_MEMORY_IO = "block_memory_io";
const char* const BLOCK_IO_BLOCK_SIZE_KEY = "block_size";

// IO param for file
const char* const IO_FILE_PATH_KEY = "file_path";
const char* const DEFAULT_FILE_PATH_VALUE = "./default_file_path";

// quantization params key
const char* const QUANTIZATION_PARAMS_KEY = "quantization_params";
// quantization type
const char* const QUANTIZATION_TYPE_VALUE_SQ8 = "sq8";
const char* const QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM = "sq8_uniform";
const char* const QUANTIZATION_TYPE_VALUE_SQ4 = "sq4";
const char* const QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM = "sq4_uniform";
const char* const QUANTIZATION_TYPE_VALUE_FP32 = "fp32";
const char* const QUANTIZATION_TYPE_VALUE_FP16 = "fp16";
const char* const QUANTIZATION_TYPE_VALUE_BF16 = "bf16";
const char* const QUANTIZATION_TYPE_VALUE_INT8 = "int8";
const char* const QUANTIZATION_TYPE_VALUE_PQ = "pq";
const char* const QUANTIZATION_TYPE_VALUE_PQFS = "pqfs";
const char* const QUANTIZATION_TYPE_VALUE_RABITQ = "rabitq";
const char* const QUANTIZATION_TYPE_VALUE_SPARSE = "sparse";
const char* const QUANTIZATION_TYPE_VALUE_TQ = "tq";

// vector transformer type
const char* const TRANSFORMER_TYPE_VALUE_PCA = "pca";
const char* const TRANSFORMER_TYPE_VALUE_ROM = "rom";
const char* const TRANSFORMER_TYPE_VALUE_FHT = "fht";
const char* const TRANSFORMER_TYPE_VALUE_RESIDUAL = "residual";
const char* const TRANSFORMER_TYPE_VALUE_NORMALIZE = "normalize";

// vector transformer param
const char* const INPUT_DIM_KEY = "input_dim";
const char* const PCA_DIM_KEY = "pca_dim";
const char* const USE_FHT_KEY = "use_fht";

// quantization param
const char* const TQ_CHAIN_KEY = "tq_chain";
const char* const RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY = "rabitq_bits_per_dim_query";
const char* const SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY = "sq4_uniform_trunc_rate";
const char* const PRODUCT_QUANTIZATION_DIM_KEY = "pq_dim";
const char* const PRODUCT_QUANTIZATION_BITS_KEY = "pq_bits";

// sparse index param
const char* const SPARSE_NEED_SORT = "need_sort";
const char* const SPARSE_QUERY_PRUNE_RATIO = "query_prune_ratio";
const char* const SPARSE_DOC_PRUNE_RATIO = "doc_prune_ratio";
const char* const SPARSE_TERM_PRUNE_RATIO = "term_prune_ratio";
const char* const SPARSE_TERM_ID_LIMIT = "term_id_limit";
const char* const SPARSE_WINDOW_SIZE = "window_size";
const char* const SPARSE_DESERIALIZE_WITHOUT_FOOTER = "deserialize_without_footer";

// graph param value
const char* const GRAPH_PARAM_MAX_DEGREE_KEY = "max_degree";
const char* const GRAPH_PARAM_INIT_MAX_CAPACITY_KEY = "init_capacity";
const char* const EF_CONSTRUCTION_KEY = "ef_construction";

const char* const GRAPH_TYPE_KEY = "graph_type";
const char* const GRAPH_TYPE_VALUE_ODESCENT = "odescent";
const char* const GRAPH_TYPE_VALUE_NSW = "nsw";

const char* const GRAPH_STORAGE_TYPE_KEY = "graph_storage_type";
const char* const GRAPH_STORAGE_TYPE_VALUE_COMPRESSED = "compressed";
const char* const GRAPH_STORAGE_TYPE_VALUE_FLAT = "flat";

// bucket params for IVF index
const char* const BUCKET_PARAMS_KEY = "buckets_params";
const char* const BUCKET_PER_DATA_KEY = "buckets_per_data";
const char* const BUCKETS_COUNT_KEY = "buckets_count";
const char* const BUCKET_USE_RESIDUAL_KEY = "use_residual";

const char* const IVF_TRAIN_TYPE_KEY = "ivf_train_type";
const char* const IVF_TRAIN_TYPE_RANDOM = "random";
const char* const IVF_TRAIN_TYPE_KMEANS = "kmeans";

const char* const IVF_TRAIN_SAMPLE_COUNT_KEY = "ivf_train_sample_count";
const char* const IVF_PARTITION_STRATEGY_PARAMS_KEY = "partition_strategy";
const char* const IVF_PARTITION_STRATEGY_TYPE_KEY = "partition_strategy_type";
const char* const IVF_PARTITION_STRATEGY_TYPE_NEAREST = "ivf";
const char* const IVF_PARTITION_STRATEGY_TYPE_GNO_IMI = "gno_imi";

const char* const GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY = "first_order_buckets_count";
const char* const GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY = "second_order_buckets_count";

const char* const GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO = "first_order_scan_ratio";
const char* const FLATTEN_DATA_CELL = "flatten_data_cell";
const char* const SPARSE_VECTOR_DATA_CELL = "sparse_vector_data_cell";

// for pyramid index
const char* const NO_BUILD_LEVELS = "no_build_levels";

const char* const GRAPH_SUPPORT_REMOVE = "support_remove";
const char* const REMOVE_FLAG_BIT = "remove_flag_bit";
const char* const HOLD_MOLDS = "hold_molds";
const char* const SUPPORT_DUPLICATE = "support_duplicate";
const char* const SUPPORT_TOMBSTONE = "support_tombstone";

const char* const DATACELL_OFFSETS = "datacell_offsets";
const char* const DATACELL_SIZES = "datacell_sizes";
const char* const BASIC_INFO = "basic_info";

const char* const CODES_TYPE_KEY = "codes_type";
const char* const FLATTEN_CODES = "flatten";
const char* const SPARSE_CODES = "sparse";

const char* const IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT = "scan_buckets_count";
const char* const SEARCH_PARAM_FACTOR = "factor";
const char* const SEARCH_PARALLELISM = "parallelism";
const char* const SEARCH_MAX_TIME_COST_MS = "timeout_ms";
const char* const SPARSE_N_CANDIDATE = "n_candidate";

const std::unordered_map<std::string, std::string> DEFAULT_MAP = {
    {"INDEX_TYPE_HGRAPH", INDEX_TYPE_HGRAPH},
    {"INDEX_TYPE_IVF", INDEX_TYPE_IVF},
    {"INDEX_TYPE_GNO_IMI", INDEX_TYPE_GNO_IMI},
    {"INDEX_TYPE_PYRAMID", INDEX_TYPE_PYRAMID},
    {"TYPE_KEY", TYPE_KEY},
    {"HGRAPH_USE_ELP_OPTIMIZER_KEY", HGRAPH_USE_ELP_OPTIMIZER_KEY},
    {"HGRAPH_IGNORE_REORDER_KEY", HGRAPH_IGNORE_REORDER_KEY},
    {"HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY", HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY},
    {"GRAPH_KEY", GRAPH_KEY},
    {"BASE_CODES_KEY", BASE_CODES_KEY},
    {"PRECISE_CODES_KEY", PRECISE_CODES_KEY},
    {"HGRAPH_SUPPORT_DUPLICATE", HGRAPH_SUPPORT_DUPLICATE},
    {"IO_TYPE_VALUE_MEMORY_IO", IO_TYPE_VALUE_MEMORY_IO},
    {"IO_TYPE_VALUE_BLOCK_MEMORY_IO", IO_TYPE_VALUE_BLOCK_MEMORY_IO},
    {"IO_TYPE_VALUE_BUFFER_IO", IO_TYPE_VALUE_BUFFER_IO},
    {"IO_PARAMS_KEY", IO_PARAMS_KEY},
    {"BLOCK_IO_BLOCK_SIZE_KEY", BLOCK_IO_BLOCK_SIZE_KEY},
    {"QUANTIZATION_TYPE_VALUE_SQ8", QUANTIZATION_TYPE_VALUE_SQ8},
    {"QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM", QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM},
    {"QUANTIZATION_TYPE_VALUE_SQ4", QUANTIZATION_TYPE_VALUE_SQ4},
    {"QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM", QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM},
    {"QUANTIZATION_TYPE_VALUE_FP32", QUANTIZATION_TYPE_VALUE_FP32},
    {"QUANTIZATION_TYPE_VALUE_PQ", QUANTIZATION_TYPE_VALUE_PQ},
    {"QUANTIZATION_TYPE_VALUE_PQFS", QUANTIZATION_TYPE_VALUE_PQFS},
    {"QUANTIZATION_TYPE_VALUE_FP16", QUANTIZATION_TYPE_VALUE_FP16},
    {"QUANTIZATION_TYPE_VALUE_BF16", QUANTIZATION_TYPE_VALUE_BF16},
    {"QUANTIZATION_TYPE_VALUE_RABITQ", QUANTIZATION_TYPE_VALUE_RABITQ},
    {"PRODUCT_QUANTIZATION_DIM_KEY", PRODUCT_QUANTIZATION_DIM_KEY},
    {"PRODUCT_QUANTIZATION_BITS_KEY", PRODUCT_QUANTIZATION_BITS_KEY},
    {"GRAPH_TYPE_VALUE_NSW", GRAPH_TYPE_VALUE_NSW},
    {"GRAPH_TYPE_VALUE_ODESCENT", GRAPH_TYPE_VALUE_ODESCENT},
    {"GRAPH_STORAGE_TYPE_KEY", GRAPH_STORAGE_TYPE_KEY},
    {"GRAPH_STORAGE_TYPE_VALUE_FLAT", GRAPH_STORAGE_TYPE_VALUE_FLAT},
    {"GRAPH_STORAGE_TYPE_VALUE_COMPRESSED", GRAPH_STORAGE_TYPE_VALUE_COMPRESSED},
    {"QUANTIZATION_PARAMS_KEY", QUANTIZATION_PARAMS_KEY},
    {"GRAPH_PARAM_MAX_DEGREE_KEY", GRAPH_PARAM_MAX_DEGREE_KEY},
    {"GRAPH_PARAM_INIT_MAX_CAPACITY_KEY", GRAPH_PARAM_INIT_MAX_CAPACITY_KEY},
    {"EF_CONSTRUCTION_KEY", EF_CONSTRUCTION_KEY},
    {"ALPHA_KEY", ALPHA_KEY},
    {"BUCKETS_COUNT_KEY", BUCKETS_COUNT_KEY},
    {"BUCKET_PARAMS_KEY", BUCKET_PARAMS_KEY},
    {"IO_FILE_PATH_KEY", IO_FILE_PATH_KEY},
    {"DEFAULT_FILE_PATH_VALUE", DEFAULT_FILE_PATH_VALUE},
    {"PRECISE_CODES_KEY", PRECISE_CODES_KEY},
    {"USE_REORDER_KEY", USE_REORDER_KEY},
    {"USE_ATTRIBUTE_FILTER_KEY", USE_ATTRIBUTE_FILTER_KEY},
    {"SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY", SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY},
    {"PCA_DIM_KEY", PCA_DIM_KEY},
    {"IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT", IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT},
    {"GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY", GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY},
    {"GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY", GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY},
    {"BUCKETS_COUNT_KEY", BUCKETS_COUNT_KEY},
    {"IVF_TRAIN_TYPE_KEY", IVF_TRAIN_TYPE_KEY},
    {"ODESCENT_PARAMETER_BUILD_BLOCK_SIZE", ODESCENT_PARAMETER_BUILD_BLOCK_SIZE},
    {"ODESCENT_PARAMETER_ALPHA", ODESCENT_PARAMETER_ALPHA},
    {"ODESCENT_PARAMETER_GRAPH_ITER_TURN", ODESCENT_PARAMETER_GRAPH_ITER_TURN},
    {"ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE", ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE},
    {"ODESCENT_PARAMETER_MIN_IN_DEGREE", ODESCENT_PARAMETER_MIN_IN_DEGREE},
    {"EXTRA_INFO_KEY", EXTRA_INFO_KEY},
    {"CODES_TYPE_KEY", CODES_TYPE_KEY},
    {"SEARCH_PARAM_FACTOR", SEARCH_PARAM_FACTOR},
    {"BUCKET_PER_DATA_KEY", BUCKET_PER_DATA_KEY},
    {"IVF_PARTITION_STRATEGY_PARAMS_KEY", IVF_PARTITION_STRATEGY_PARAMS_KEY},
    {"IVF_PARTITION_STRATEGY_TYPE_KEY", IVF_PARTITION_STRATEGY_TYPE_KEY},
    {"IVF_PARTITION_STRATEGY_TYPE_NEAREST", IVF_PARTITION_STRATEGY_TYPE_NEAREST},
    {"IVF_TRAIN_TYPE_KMEANS", IVF_TRAIN_TYPE_KMEANS},
    {"BUILD_THREAD_COUNT_KEY", BUILD_THREAD_COUNT_KEY},
    {"SEARCH_PARALLELISM", SEARCH_PARALLELISM},
    {"GRAPH_SUPPORT_REMOVE", GRAPH_SUPPORT_REMOVE},
    {"REMOVE_FLAG_BIT", REMOVE_FLAG_BIT},
    {"HOLD_MOLDS", HOLD_MOLDS},
    {"IVF_PARTITION_STRATEGY_TYPE_GNO_IMI", IVF_PARTITION_STRATEGY_TYPE_GNO_IMI},
    {"STORE_RAW_VECTOR_KEY", STORE_RAW_VECTOR_KEY},
    {"RAW_VECTOR_KEY", RAW_VECTOR_KEY},
    {"ATTR_HAS_BUCKETS_KEY", ATTR_HAS_BUCKETS_KEY},
    {"ATTR_PARAMS_KEY", ATTR_PARAMS_KEY},
    {"RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY", RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY},
    {"TQ_CHAIN_KEY", TQ_CHAIN_KEY},
    {"NO_BUILD_LEVELS", NO_BUILD_LEVELS},
    {"GRAPH_TYPE_KEY", GRAPH_TYPE_KEY}};

}  // namespace vsag
