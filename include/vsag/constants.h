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
extern const char* const INDEX_PYRAMID;
extern const char* const INDEX_SPARSE;
extern const char* const INDEX_BRUTE_FORCE;
extern const char* const INDEX_IVF;
extern const char* const DIM;
extern const char* const NUM_ELEMENTS;
extern const char* const IDS;
extern const char* const DISTS;
extern const char* const FLOAT32_VECTORS;
extern const char* const SPARSE_VECTORS;
extern const char* const INT8_VECTORS;
extern const char* const ATTRIBUTE_SETS;
extern const char* const DATASET_PATHS;
extern const char* const EXTRA_INFOS;
extern const char* const EXTRA_INFO_SIZE;

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
extern const char* const DATATYPE_SPARSE;
extern const char* const BLANK_INDEX;

// environment-level-parameters
extern const char* const PREFETCH_STRIDE_VISIT;
extern const char* const PREFETCH_STRIDE_CODE;
extern const char* const PREFETCH_DEPTH_CODE;

// parameters
extern const char* const PARAMETER_DTYPE;
extern const char* const PARAMETER_DIM;
extern const char* const PARAMETER_METRIC_TYPE;
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
extern const char* const DISKANN_PARAMETER_GRAPH_TYPE;
extern const char* const ODESCENT_PARAMETER_ALPHA;
extern const char* const ODESCENT_PARAMETER_GRAPH_ITER_TURN;
extern const char* const ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE;
extern const char* const ODESCENT_PARAMETER_MIN_IN_DEGREE;
extern const char* const ODESCENT_PARAMETER_BUILD_BLOCK_SIZE;
extern const char* const DISKANN_GRAPH_TYPE_VAMANA;
extern const char* const GRAPH_TYPE_ODESCENT;
extern const char* const GRAPH_TYPE_NSW;

extern const char* const DISKANN_PARAMETER_BEAM_SEARCH;
extern const char* const DISKANN_PARAMETER_IO_LIMIT;
extern const char* const DISKANN_PARAMETER_EF_SEARCH;
extern const char* const DISKANN_PARAMETER_REORDER;

extern const char* const HNSW_PARAMETER_EF_RUNTIME;
extern const char* const HNSW_PARAMETER_M;
extern const char* const HNSW_PARAMETER_CONSTRUCTION;
extern const char* const HNSW_PARAMETER_USE_STATIC;
extern const char* const HNSW_PARAMETER_REVERSED_EDGES;
extern const char* const HNSW_PARAMETER_SKIP_RATIO;

extern const char* const PYRAMID_PARAMETER_BASE_CODES;

extern const char* const INDEX_PARAM;

extern const char PART_SLASH;

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

extern const char* const SQ4_UNIFORM_TRUNC_RATE;
extern const char* const RABITQ_PCA_DIM;
extern const char* const RABITQ_BITS_PER_DIM_QUERY;

extern const char* const RABITQ_USE_FHT;

extern const char* const HGRAPH_SUPPORT_REMOVE;
extern const char* const HGRAPH_REMOVE_FLAG_BIT;

// hgraph params
extern const char* const HGRAPH_USE_REORDER;
extern const char* const HGRAPH_USE_ELP_OPTIMIZER;
extern const char* const HGRAPH_IGNORE_REORDER;
extern const char* const HGRAPH_BUILD_BY_BASE_QUANTIZATION;
extern const char* const HGRAPH_BASE_QUANTIZATION_TYPE;
extern const char* const HGRAPH_GRAPH_MAX_DEGREE;
extern const char* const HGRAPH_BUILD_EF_CONSTRUCTION;
extern const char* const HGRAPH_INIT_CAPACITY;
extern const char* const HGRAPH_GRAPH_TYPE;
extern const char* const HGRAPH_GRAPH_STORAGE_TYPE;
extern const char* const HGRAPH_BUILD_THREAD_COUNT;
extern const char* const HGRAPH_PRECISE_QUANTIZATION_TYPE;
extern const char* const HGRAPH_BASE_IO_TYPE;
extern const char* const HGRAPH_BASE_PQ_DIM;
extern const char* const HGRAPH_BASE_FILE_PATH;
extern const char* const HGRAPH_PRECISE_IO_TYPE;
extern const char* const HGRAPH_PRECISE_FILE_PATH;
extern const char* const HGRAPH_PARAMETER_EF_RUNTIME;
extern const char* const HGRAPH_EXTRA_INFO_SIZE;
extern const char* const HGRAPH_USE_EXTRA_INFO_FILTER;
extern const char* const HGRAPH_USE_ATTRIBUTE_FILTER;

extern const char* const BRUTE_FORCE_QUANTIZATION_TYPE;
extern const char* const BRUTE_FORCE_IO_TYPE;

extern const char* const IVF_USE_RESIDUAL;
extern const char* const IVF_USE_REORDER;
extern const char* const IVF_TRAIN_TYPE;
extern const char* const IVF_BUCKETS_COUNT;
extern const char* const IVF_BASE_QUANTIZATION_TYPE;
extern const char* const IVF_BASE_IO_TYPE;
extern const char* const IVF_BASE_PQ_DIM;
extern const char* const IVF_BASE_FILE_PATH;
extern const char* const IVF_PRECISE_QUANTIZATION_TYPE;
extern const char* const IVF_PRECISE_IO_TYPE;
extern const char* const IVF_PRECISE_FILE_PATH;
extern const char* const IVF_USE_ATTRIBUTE_FILTER;
extern const char* const IVF_THREAD_COUNT;

extern const char* const GNO_IMI_FIRST_ORDER_BUCKETS_COUNT;
extern const char* const GNO_IMI_SECOND_ORDER_BUCKETS_COUNT;

// serialization
extern const char* const SERIAL_MAGIC_BEGIN;
extern const char* const SERIAL_MAGIC_END;
extern const char* const SERIAL_META_KEY;

}  // namespace vsag
