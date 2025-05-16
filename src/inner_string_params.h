
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

// Parameter key for hgraph
const char* const HGRAPH_USE_REORDER_KEY = "use_reorder";
const char* const HGRAPH_USE_ELP_OPTIMIZER_KEY = "use_elp_optimizer";
const char* const HGRAPH_IGNORE_REORDER_KEY = "ignore_reorder";
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

// quantization param
const char* const PCA_DIM = "pca_dim";
const char* const RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY = "rabitq_bits_per_dim_query";
const char* const SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE = "sq4_uniform_trunc_rate";
const char* const PRODUCT_QUANTIZATION_DIM = "pq_dim";
const char* const PRODUCT_QUANTIZATION_BITS = "pq_bits";

// graph param value
const char* const GRAPH_PARAM_MAX_DEGREE = "max_degree";
const char* const GRAPH_PARAM_INIT_MAX_CAPACITY = "init_capacity";

const char* const BUILD_PARAMS_KEY = "build_params";
const char* const BUILD_THREAD_COUNT = "build_thread_count";
const char* const BUILD_EF_CONSTRUCTION = "ef_construction";

const char* const SPARSE_NEED_SORT = "need_sort";
const char* const GRAPH_TYPE_KEY = "graph_type";

const char* const BUCKET_PARAMS_KEY = "buckets_params";
const char* const NO_BUILD_LEVELS = "no_build_levels";

const char* const BUCKETS_COUNT_KEY = "buckets_count";
const char* const IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT = "scan_buckets_count";
const char* const IVF_SEARCH_PARAM_FACTOR = "factor";
const char* const IVF_USE_REORDER_KEY = "use_reorder";
const char* const IVF_PRECISE_CODES_KEY = "precise_codes";
const char* const IVF_TRAIN_TYPE_KEY = "ivf_train_type";
const char* const IVF_TRAIN_TYPE_RANDOM = "random";
const char* const IVF_TRAIN_TYPE_KMEANS = "kmeans";

const char* const FLATTEN_DATA_CELL = "flatten_data_cell";
const char* const SPARSE_VECTOR_DATA_CELL = "sparse_vector_data_cell";

const char* const GRAPH_SUPPORT_REMOVE = "support_remove";
const char* const REMOVE_FLAG_BIT = "remove_flag_bit";

const std::unordered_map<std::string, std::string> DEFAULT_MAP = {
    {"INDEX_TYPE_HGRAPH", INDEX_TYPE_HGRAPH},
    {"INDEX_TYPE_IVF", INDEX_TYPE_IVF},
    {"HGRAPH_USE_REORDER_KEY", HGRAPH_USE_REORDER_KEY},
    {"HGRAPH_USE_ELP_OPTIMIZER_KEY", HGRAPH_USE_ELP_OPTIMIZER_KEY},
    {"HGRAPH_IGNORE_REORDER_KEY", HGRAPH_IGNORE_REORDER_KEY},
    {"HGRAPH_GRAPH_KEY", HGRAPH_GRAPH_KEY},
    {"HGRAPH_BASE_CODES_KEY", HGRAPH_BASE_CODES_KEY},
    {"HGRAPH_PRECISE_CODES_KEY", HGRAPH_PRECISE_CODES_KEY},
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
    {"SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE", SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE},
    {"PCA_DIM", PCA_DIM},
    {"IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT", IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT},
    {"IVF_TRAIN_TYPE_KEY", IVF_TRAIN_TYPE_KEY},
    {"HGRAPH_EXTRA_INFO_KEY", HGRAPH_EXTRA_INFO_KEY},
    {"IVF_SEARCH_PARAM_FACTOR", IVF_SEARCH_PARAM_FACTOR},
};

}  // namespace vsag
