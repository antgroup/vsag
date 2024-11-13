
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

namespace vsag {
// Index Type
const char* const INDEX_TYPE_HGRAPH = "hgraph";

// Parameter key for hgraph
const char* const HGRAPH_USE_REORDER_KEY = "use_reorder";
const char* const HGRAPH_GRAPH_KEY = "graph";
const char* const HGRAPH_BASE_CODES_KEY = "base_codes";
const char* const HGRAPH_PRECISE_CODES_KEY = "precise_codes";

// IO type
const char* const IO_TYPE_KEY = "io_type";
const char* const IO_TYPE_VALUE_MEMORY_IO = "memory";
const char* const IO_TYPE_VALUE_BLOCK_MEMORY_IO = "block_memory";

// IO param key
const char* const IO_PARAMS_KEY = "io_params";
const char* const BLOCK_IO_BLOCK_SIZE_KEY = "block_size";

// quantization type
const char* const QUANTIZATION_TYPE_KEY = "quantization_type";
const char* const QUANTIZATION_TYPE_VALUE_SQ8 = "sq8";
const char* const QUANTIZATION_TYPE_VALUE_SQ4 = "sq4";
const char* const QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM = "sq4_uniform";
const char* const QUANTIZATION_TYPE_VALUE_FP32 = "fp32";
const char* const QUANTIZATION_TYPE_VALUE_PQ = "pq";

// quantization params key
const char* const QUANTIZATION_PARAMS_KEY = "quantization_params";

// graph param key
const char* const GRAPH_PARAMS_KEY = "graph_params";

// graph param value
const char* const GRAPH_PARAM_MAX_DEGREE = "max_degree";
const char* const GRAPH_PARAM_INIT_MAX_CAPACITY = "init_capacity";

const char* const BUILD_THREAD_COUNT = "build_thread_count";

const std::unordered_map<std::string, std::string> DEFAULT_MAP = {
    {"HGRAPH_USE_REORDER_KEY", HGRAPH_USE_REORDER_KEY},
    {"HGRAPH_GRAPH_KEY", HGRAPH_GRAPH_KEY},
    {"HGRAPH_BASE_CODES_KEY", HGRAPH_BASE_CODES_KEY},
    {"HGRAPH_PRECISE_CODES_KEY", HGRAPH_PRECISE_CODES_KEY},
    {"IO_TYPE_KEY", IO_TYPE_KEY},
    {"IO_TYPE_VALUE_MEMORY_IO", IO_TYPE_VALUE_MEMORY_IO},
    {"IO_TYPE_VALUE_BLOCK_MEMORY_IO", IO_TYPE_VALUE_BLOCK_MEMORY_IO},
    {"IO_PARAMS_KEY", IO_PARAMS_KEY},
    {"BLOCK_IO_BLOCK_SIZE_KEY", BLOCK_IO_BLOCK_SIZE_KEY},
    {"QUANTIZATION_TYPE_KEY", QUANTIZATION_TYPE_KEY},
    {"QUANTIZATION_TYPE_VALUE_SQ8", QUANTIZATION_TYPE_VALUE_SQ8},
    {"QUANTIZATION_TYPE_VALUE_FP32", QUANTIZATION_TYPE_VALUE_FP32},
    {"QUANTIZATION_TYPE_VALUE_PQ", QUANTIZATION_TYPE_VALUE_PQ},
    {"QUANTIZATION_PARAMS_KEY", QUANTIZATION_PARAMS_KEY},
    {"GRAPH_PARAMS_KEY", GRAPH_PARAMS_KEY},
    {"GRAPH_PARAM_MAX_DEGREE", GRAPH_PARAM_MAX_DEGREE},
    {"GRAPH_PARAM_INIT_MAX_CAPACITY", GRAPH_PARAM_INIT_MAX_CAPACITY},
    {"BUILD_THREAD_COUNT", BUILD_THREAD_COUNT},
};

}  // namespace vsag
