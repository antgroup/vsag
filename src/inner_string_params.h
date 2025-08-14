
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
const char* const INDEX_TYPE_GNO_IMI = "gno_imi";

// Parameter key for hgraph
const char* const HGRAPH_USE_REORDER_KEY = "use_reorder";
const char* const HGRAPH_USE_ELP_OPTIMIZER_KEY = "use_elp_optimizer";
const char* const HGRAPH_IGNORE_REORDER_KEY = "ignore_reorder";
const char* const HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY = "build_by_base";
const char* const HGRAPH_GRAPH_KEY = "graph";
const char* const HGRAPH_BASE_CODES_KEY = "base_codes";
const char* const HGRAPH_PRECISE_CODES_KEY = "precise_codes";
const char* const HGRAPH_EXTRA_INFO_KEY = "extra_info";

// IO param key
const char* const IO_PARAMS_KEY = "io_params";
// IO type
const char* const IO_TYPE_KEY = "type";
const char* const IO_TYPE_VALUE_MEMORY_IO = "memory_io";
const char* const IO_TYPE_VALUE_BUFFER_IO = "buffer_io";
const char* const IO_TYPE_VALUE_MMAP_IO = "mmap_io";
const char* const IO_TYPE_VALUE_READER_IO = "reader_io";
const char* const IO_TYPE_VALUE_ASYNC_IO = "async_io";
const char* const IO_TYPE_VALUE_BLOCK_MEMORY_IO = "block_memory_io";
const char* const BLOCK_IO_BLOCK_SIZE_KEY = "block_size";
const char* const IO_FILE_PATH = "file_path";
const char* const DEFAULT_FILE_PATH_VALUE = "./default_file_path";

// quantization params key
const char* const QUANTIZATION_PARAMS_KEY = "quantization_params";
// quantization type
const char* const QUANTIZATION_TYPE_KEY = "type";
const char* const QUANTIZATION_TYPE_VALUE_SQ8 = "sq8";
const char* const QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM = "sq8_uniform";
const char* const QUANTIZATION_TYPE_VALUE_SQ4 = "sq4";
const char* const QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM = "sq4_uniform";
const char* const QUANTIZATION_TYPE_VALUE_FP32 = "fp32";
const char* const QUANTIZATION_TYPE_VALUE_FP16 = "fp16";
const char* const QUANTIZATION_TYPE_VALUE_BF16 = "bf16";
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
const char* const INPUT_DIM = "input_dim";
const char* const PCA_DIM = "pca_dim";
const char* const USE_FHT = "use_fht";

// quantization param
const char* const TQ_CHAIN = "tq_chain";
const char* const RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY = "rabitq_bits_per_dim_query";
const char* const SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE = "sq4_uniform_trunc_rate";
const char* const PRODUCT_QUANTIZATION_DIM = "pq_dim";
const char* const PRODUCT_QUANTIZATION_BITS = "pq_bits";

// sparse index param
const char* const SPARSE_NEED_SORT = "need_sort";
const char* const SPARSE_QUERY_PRUNE_RATIO = "query_prune_ratio";
const char* const SPARSE_DOC_PRUNE_RATIO = "doc_prune_ratio";
const char* const SPARSE_TERM_PRUNE_RATIO = "term_prune_ratio";
const char* const SPARSE_WINDOW_SIZE = "window_size";
const char* const SPARSE_USE_REORDER = "use_reorder";
const char* const SPARSE_N_CANDIDATE = "n_candidate";

// graph param value
const char* const GRAPH_PARAM_MAX_DEGREE = "max_degree";
const char* const GRAPH_PARAM_INIT_MAX_CAPACITY = "init_capacity";

const char* const BUILD_PARAMS_KEY = "build_params";
const char* const BUILD_THREAD_COUNT = "build_thread_count";
const char* const BUILD_EF_CONSTRUCTION = "ef_construction";

const char* const GRAPH_TYPE_KEY = "graph_type";

const char* const GRAPH_STORAGE_TYPE_KEY = "graph_storage_type";
const char* const GRAPH_STORAGE_TYPE_COMPRESSED = "compressed";
const char* const GRAPH_STORAGE_TYPE_FLAT = "flat";

const char* const BUCKET_PARAMS_KEY = "buckets_params";
const char* const BUCKET_PER_DATA_KEY = "buckets_per_data";
const char* const NO_BUILD_LEVELS = "no_build_levels";

const char* const BUCKETS_COUNT_KEY = "buckets_count";
const char* const BUCKET_USE_RESIDUAL = "use_residual";
const char* const IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT = "scan_buckets_count";
const char* const IVF_SEARCH_PARAM_FACTOR = "factor";
const char* const IVF_SEARCH_PARALLELISM = "parallelism";
const char* const SEARCH_MAX_TIME_COST_MS = "timeout_ms";

const char* const IVF_USE_REORDER_KEY = "use_reorder";
const char* const IVF_PRECISE_CODES_KEY = "precise_codes";
const char* const USE_ATTRIBUTE_FILTER_KEY = "use_attribute_filter";
const char* const IVF_THREAD_COUNT_KEY = "thread_count";
const char* const IVF_TRAIN_TYPE_KEY = "ivf_train_type";
const char* const IVF_TRAIN_TYPE_RANDOM = "random";
const char* const IVF_TRAIN_TYPE_KMEANS = "kmeans";

const char* const IVF_PARTITION_STRATEGY_PARAMS_KEY = "partition_strategy";
const char* const IVF_PARTITION_STRATEGY_TYPE_KEY = "partition_strategy_type";
const char* const IVF_PARTITION_STRATEGY_TYPE_NEAREST = "ivf";
const char* const IVF_PARTITION_STRATEGY_TYPE_GNO_IMI = "gno_imi";

const char* const GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY = "first_order_buckets_count";
const char* const GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY = "second_order_buckets_count";

const char* const GNO_IMI_SEARCH_PARAM_FIRST_ORDER_SCAN_RATIO = "first_order_scan_ratio";
const char* const FLATTEN_DATA_CELL = "flatten_data_cell";
const char* const SPARSE_VECTOR_DATA_CELL = "sparse_vector_data_cell";

const char* const GRAPH_SUPPORT_REMOVE = "support_remove";
const char* const REMOVE_FLAG_BIT = "remove_flag_bit";
const char* const HOLD_MOLDS = "hold_molds";
const char* const SUPPORT_DUPLICATE = "support_duplicate";

const char* const DATACELL_OFFSETS = "datacell_offsets";
const char* const DATACELL_SIZES = "datacell_sizes";
const char* const BASIC_INFO = "basic_info";
const char* const INDEX_TYPE = "type";

const std::unordered_map<std::string, std::string> DEFAULT_MAP = {
    {"INDEX_TYPE_HGRAPH", INDEX_TYPE_HGRAPH},
    {"INDEX_TYPE_IVF", INDEX_TYPE_IVF},
    {"INDEX_TYPE_GNO_IMI", INDEX_TYPE_GNO_IMI},
    {"HGRAPH_USE_REORDER_KEY", HGRAPH_USE_REORDER_KEY},
    {"HGRAPH_USE_ELP_OPTIMIZER_KEY", HGRAPH_USE_ELP_OPTIMIZER_KEY},
    {"HGRAPH_IGNORE_REORDER_KEY", HGRAPH_IGNORE_REORDER_KEY},
    {"HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY", HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY},
    {"HGRAPH_GRAPH_KEY", HGRAPH_GRAPH_KEY},
    {"HGRAPH_BASE_CODES_KEY", HGRAPH_BASE_CODES_KEY},
    {"HGRAPH_PRECISE_CODES_KEY", HGRAPH_PRECISE_CODES_KEY},
    {"HGRAPH_SUPPORT_DUPLICATE", HGRAPH_SUPPORT_DUPLICATE},
    {"IO_TYPE_KEY", IO_TYPE_KEY},
    {"IO_TYPE_VALUE_MEMORY_IO", IO_TYPE_VALUE_MEMORY_IO},
    {"IO_TYPE_VALUE_BLOCK_MEMORY_IO", IO_TYPE_VALUE_BLOCK_MEMORY_IO},
    {"IO_TYPE_VALUE_BUFFER_IO", IO_TYPE_VALUE_BUFFER_IO},
    {"IO_PARAMS_KEY", IO_PARAMS_KEY},
    {"BLOCK_IO_BLOCK_SIZE_KEY", BLOCK_IO_BLOCK_SIZE_KEY},
    {"QUANTIZATION_TYPE_KEY", QUANTIZATION_TYPE_KEY},
    {"QUANTIZATION_TYPE_VALUE_SQ8", QUANTIZATION_TYPE_VALUE_SQ8},
    {"QUANTIZATION_TYPE_VALUE_FP32", QUANTIZATION_TYPE_VALUE_FP32},
    {"QUANTIZATION_TYPE_VALUE_PQ", QUANTIZATION_TYPE_VALUE_PQ},
    {"QUANTIZATION_TYPE_VALUE_PQFS", QUANTIZATION_TYPE_VALUE_PQFS},
    {"QUANTIZATION_TYPE_VALUE_FP16", QUANTIZATION_TYPE_VALUE_FP16},
    {"QUANTIZATION_TYPE_VALUE_BF16", QUANTIZATION_TYPE_VALUE_BF16},
    {"QUANTIZATION_TYPE_VALUE_RABITQ", QUANTIZATION_TYPE_VALUE_RABITQ},
    {"PRODUCT_QUANTIZATION_DIM", PRODUCT_QUANTIZATION_DIM},
    {"PRODUCT_QUANTIZATION_BITS", PRODUCT_QUANTIZATION_BITS},
    {"GRAPH_TYPE_NSW", GRAPH_TYPE_NSW},
    {"GRAPH_STORAGE_TYPE_KEY", GRAPH_STORAGE_TYPE_KEY},
    {"GRAPH_STORAGE_TYPE_FLAT", GRAPH_STORAGE_TYPE_FLAT},
    {"GRAPH_STORAGE_TYPE_COMPRESSED", GRAPH_STORAGE_TYPE_COMPRESSED},
    {"QUANTIZATION_PARAMS_KEY", QUANTIZATION_PARAMS_KEY},
    {"GRAPH_PARAM_MAX_DEGREE", GRAPH_PARAM_MAX_DEGREE},
    {"GRAPH_PARAM_INIT_MAX_CAPACITY", GRAPH_PARAM_INIT_MAX_CAPACITY},
    {"BUILD_PARAMS_KEY", BUILD_PARAMS_KEY},
    {"BUILD_THREAD_COUNT", BUILD_THREAD_COUNT},
    {"BUILD_EF_CONSTRUCTION", BUILD_EF_CONSTRUCTION},
    {"BUCKETS_COUNT_KEY", BUCKETS_COUNT_KEY},
    {"BUCKET_PARAMS_KEY", BUCKET_PARAMS_KEY},
    {"IO_FILE_PATH", IO_FILE_PATH},
    {"DEFAULT_FILE_PATH_VALUE", DEFAULT_FILE_PATH_VALUE},
    {"IVF_PRECISE_CODES_KEY", IVF_PRECISE_CODES_KEY},
    {"IVF_USE_REORDER_KEY", IVF_USE_REORDER_KEY},
    {"USE_ATTRIBUTE_FILTER_KEY", USE_ATTRIBUTE_FILTER_KEY},
    {"SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE", SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE},
    {"PCA_DIM", PCA_DIM},
    {"IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT", IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT},
    {"GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY", GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY},
    {"GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY", GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY},
    {"BUCKETS_COUNT_KEY", BUCKETS_COUNT_KEY},
    {"IVF_TRAIN_TYPE_KEY", IVF_TRAIN_TYPE_KEY},
    {"ODESCENT_PARAMETER_BUILD_BLOCK_SIZE", ODESCENT_PARAMETER_BUILD_BLOCK_SIZE},
    {"ODESCENT_PARAMETER_ALPHA", ODESCENT_PARAMETER_ALPHA},
    {"ODESCENT_PARAMETER_GRAPH_ITER_TURN", ODESCENT_PARAMETER_GRAPH_ITER_TURN},
    {"ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE", ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE},
    {"HGRAPH_EXTRA_INFO_KEY", HGRAPH_EXTRA_INFO_KEY},
    {"IVF_SEARCH_PARAM_FACTOR", IVF_SEARCH_PARAM_FACTOR},
    {"BUCKET_PER_DATA_KEY", BUCKET_PER_DATA_KEY},
    {"IVF_PARTITION_STRATEGY_PARAMS_KEY", IVF_PARTITION_STRATEGY_PARAMS_KEY},
    {"IVF_PARTITION_STRATEGY_TYPE_KEY", IVF_PARTITION_STRATEGY_TYPE_KEY},
    {"IVF_PARTITION_STRATEGY_TYPE_NEAREST", IVF_PARTITION_STRATEGY_TYPE_NEAREST},
    {"IVF_TRAIN_TYPE_KMEANS", IVF_TRAIN_TYPE_KMEANS},
    {"IVF_THREAD_COUNT_KEY", IVF_THREAD_COUNT_KEY},
    {"IVF_SEARCH_PARALLELISM", IVF_SEARCH_PARALLELISM},
    {"HGRAPH_STORE_RAW_VECTOR", HGRAPH_STORE_RAW_VECTOR},
    {"GRAPH_SUPPORT_REMOVE", GRAPH_SUPPORT_REMOVE},
    {"REMOVE_FLAG_BIT", REMOVE_FLAG_BIT},
    {"HOLD_MOLDS", HOLD_MOLDS},
    {"IVF_PARTITION_STRATEGY_TYPE_GNO_IMI", IVF_PARTITION_STRATEGY_TYPE_GNO_IMI}};

}  // namespace vsag
