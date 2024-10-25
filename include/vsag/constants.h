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

namespace vsag {

extern const char* const INDEX_HGRAPH;
extern const char* const INDEX_DISKANN;
extern const char* const INDEX_HNSW;
extern const char* const INDEX_FRESH_HNSW;
extern const char* const MULTI_INDEX;
extern const char* const DIM;
extern const char* const NUM_ELEMENTS;
extern const char* const IDS;
extern const char* const DISTS;
extern const char* const FLOAT32_VECTORS;
extern const char* const INT8_VECTORS;
extern const char* const DATAKEY_TAGS;
extern const char* const HNSW_DATA;
extern const char* const CONJUGATE_GRAPH_DATA;
extern const char* const DISKANN_PQ;
extern const char* const DISKANN_COMPRESSED_VECTOR;
extern const char* const DISKANN_LAYOUT_FILE;
extern const char* const DISKANN_TAG_FILE;
extern const char* const DISKANN_GRAPH;
extern const char* const SIMPLEFLAT_VECTORS;
extern const char* const SIMPLEFLAT_IDS;
extern const char* const METRIC_L2;
extern const char* const METRIC_COSINE;
extern const char* const METRIC_IP;
extern const char* const DATATYPE_FLOAT32;
extern const char* const DATATYPE_INT8;
extern const char* const BLANK_INDEX;

// parameters
extern const char* const PARAMETER_DTYPE;
extern const char* const PARAMETER_DIM;
extern const char* const PARAMETER_METRIC_TYPE;
extern const char* const PARAMETER_INDEX_TYPE;
extern const char* const PARAMETER_USE_CONJUGATE_GRAPH;
extern const char* const PARAMETER_USE_CONJUGATE_GRAPH_SEARCH;

extern const char* const DISKANN_PARAMETER_L;
extern const char* const DISKANN_PARAMETER_R;
extern const char* const DISKANN_PARAMETER_P_VAL;
extern const char* const DISKANN_PARAMETER_DISK_PQ_DIMS;
extern const char* const DISKANN_PARAMETER_PRELOAD;
extern const char* const DISKANN_PARAMETER_USE_REFERENCE;
extern const char* const DISKANN_PARAMETER_USE_OPQ;
extern const char* const DISKANN_PARAMETER_USE_ASYNC_IO;
extern const char* const DISKANN_PARAMETER_USE_BSA;

extern const char* const DISKANN_PARAMETER_BEAM_SEARCH;
extern const char* const DISKANN_PARAMETER_IO_LIMIT;
extern const char* const DISKANN_PARAMETER_EF_SEARCH;
extern const char* const DISKANN_PARAMETER_REORDER;

extern const char* const HNSW_PARAMETER_EF_RUNTIME;
extern const char* const HNSW_PARAMETER_M;
extern const char* const HNSW_PARAMETER_CONSTRUCTION;
extern const char* const HNSW_PARAMETER_USE_STATIC;
extern const char* const HNSW_PARAMETER_REVERSED_EDGES;

// statstic key
extern const char* const STATSTIC_MEMORY;
extern const char* const STATSTIC_INDEX_NAME;
extern const char* const STATSTIC_DATA_NUM;

extern const char* const STATSTIC_KNN_TIME;
extern const char* const STATSTIC_KNN_IO;
extern const char* const STATSTIC_KNN_HOP;
extern const char* const STATSTIC_KNN_IO_TIME;
extern const char* const STATSTIC_KNN_CACHE_HIT;
extern const char* const STATSTIC_RANGE_TIME;
extern const char* const STATSTIC_RANGE_IO;
extern const char* const STATSTIC_RANGE_HOP;
extern const char* const STATSTIC_RANGE_CACHE_HIT;
extern const char* const STATSTIC_RANGE_IO_TIME;

//Error message
extern const char* const MESSAGE_PARAMETER;

// Serialize key
extern const char* const SERIALIZE_MAGIC_NUM;
extern const char* const SERIALIZE_VERSION;

// Index Type
extern const char* const INDEX_TYPE_HGRAPH;

// Parameter key for hgraph
extern const char* const HGRAPH_USE_REORDER_KEY;
extern const char* const HGRAPH_GRAPH_KEY;
extern const char* const HGRAPH_BASE_CODES_KEY;
extern const char* const HGRAPH_PRECISE_CODES_KEY;

// IO type
extern const char* const IO_TYPE_KEY;
extern const char* const IO_TYPE_VALUE_MEMORY_IO;
extern const char* const IO_TYPE_VALUE_BLOCK_MEMORY_IO;

// IO param key
extern const char* const IO_PARAMS_KEY;
extern const char* const BLOCK_IO_BLOCK_SIZE_KEY;

// quantization type
extern const char* const QUANTIZATION_TYPE_KEY;
extern const char* const QUANTIZATION_TYPE_VALUE_SQ8;
extern const char* const QUANTIZATION_TYPE_VALUE_FP32;

// quantization param key
extern const char* const QUANTIZATION_PARAMS_KEY;

// graph param key
extern const char* const GRAPH_PARAMS_KEY;

// graph param value
extern const char* const GRAPH_PARAM_MAX_DEGREE;
extern const char* const GRAPH_PARAM_INIT_MAX_CAPACITY;

}  // namespace vsag
