
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

#include "vsag/constants.h"

#include "inner_string_params.h"
namespace vsag {

const char* const INDEX_HGRAPH = "hgraph";
const char* const INDEX_DISKANN = "diskann";
const char* const INDEX_HNSW = "hnsw";
const char* const INDEX_FRESH_HNSW = "fresh_hnsw";
const char* const INDEX_PYRAMID = "pyramid";
const char* const INDEX_SPARSE = "sparse_index";
const char* const INDEX_BRUTE_FORCE = "brute_force";
const char* const INDEX_IVF = "ivf";
const char* const INDEX_GNO_IMI = "gno_imi";

const char* const DIM = "dim";
const char* const NUM_ELEMENTS = "num_elements";
const char* const IDS = "ids";
const char* const DISTS = "dists";
const char* const FLOAT32_VECTORS = "f32_vectors";
const char* const SPARSE_VECTORS = "sparse_vectors";
const char* const INT8_VECTORS = "i8_vectors";
const char* const ATTRIBUTE_SETS = "attribute_sets";
const char* const DATASET_PATHS = "paths";
const char* const EXTRA_INFOS = "extra_infos";
const char* const EXTRA_INFO_SIZE = "extra_info_size";

const char* const HNSW_DATA = "hnsw_data";
const char* const CONJUGATE_GRAPH_DATA = "conjugate_graph_data";
const char* const DISKANN_PQ = "diskann_qp";
const char* const DISKANN_COMPRESSED_VECTOR = "diskann_compressed_vector";
const char* const DISKANN_LAYOUT_FILE = "diskann_layout_file";
const char* const DISKANN_TAG_FILE = "diskann_tag_file";
const char* const DISKANN_GRAPH = "diskann_graph";
const char* const SIMPLEFLAT_VECTORS = "simpleflat_vectors";
const char* const SIMPLEFLAT_IDS = "simpleflat_ids";
const char* const METRIC_L2 = "l2";
const char* const METRIC_COSINE = "cosine";
const char* const METRIC_IP = "ip";
const char* const DATATYPE_FLOAT32 = "float32";
const char* const DATATYPE_INT8 = "int8";
const char* const DATATYPE_SPARSE = "sparse";
const char* const BLANK_INDEX = "blank_index";

// environment-level-parameters
const char* const PREFETCH_STRIDE_VISIT = "prefetch_stride_visit";
const char* const PREFETCH_STRIDE_CODE = "prefetch_stride_codes";
const char* const PREFETCH_DEPTH_CODE = "prefetch_depth_codes";

// parameters
const char* const PARAMETER_DTYPE = "dtype";
const char* const PARAMETER_DIM = "dim";
const char* const PARAMETER_METRIC_TYPE = "metric_type";
const char* const PARAMETER_USE_CONJUGATE_GRAPH = "use_conjugate_graph";
const char* const PARAMETER_USE_CONJUGATE_GRAPH_SEARCH = "use_conjugate_graph_search";

const char* const DISKANN_PARAMETER_L = "ef_construction";
const char* const DISKANN_PARAMETER_R = "max_degree";
const char* const DISKANN_PARAMETER_P_VAL = "pq_sample_rate";
const char* const DISKANN_PARAMETER_DISK_PQ_DIMS = "pq_dims";
const char* const DISKANN_PARAMETER_PRELOAD = "use_pq_search";
const char* const DISKANN_PARAMETER_USE_REFERENCE = "use_reference";
const char* const DISKANN_PARAMETER_USE_OPQ = "use_opq";
const char* const DISKANN_PARAMETER_USE_ASYNC_IO = "use_async_io";
const char* const DISKANN_PARAMETER_USE_BSA = "use_bsa";

const char* const DISKANN_PARAMETER_BEAM_SEARCH = "beam_search";
const char* const DISKANN_PARAMETER_IO_LIMIT = "io_limit";
const char* const DISKANN_PARAMETER_EF_SEARCH = "ef_search";
const char* const DISKANN_PARAMETER_REORDER = "use_reorder";
const char* const DISKANN_PARAMETER_GRAPH_TYPE = "graph_type";
const char* const ODESCENT_PARAMETER_ALPHA = "alpha";
const char* const ODESCENT_PARAMETER_GRAPH_ITER_TURN = "graph_iter_turn";
const char* const ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE = "neighbor_sample_rate";
const char* const ODESCENT_PARAMETER_MIN_IN_DEGREE = "min_in_degree";
const char* const ODESCENT_PARAMETER_BUILD_BLOCK_SIZE = "build_block_size";

const char* const DISKANN_GRAPH_TYPE_VAMANA = "vamana";
const char* const GRAPH_TYPE_ODESCENT = "odescent";
const char* const GRAPH_TYPE_NSW = "nsw";

const char* const HNSW_PARAMETER_EF_RUNTIME = "ef_search";
const char* const HNSW_PARAMETER_M = "max_degree";
const char* const HNSW_PARAMETER_CONSTRUCTION = "ef_construction";
const char* const HNSW_PARAMETER_USE_STATIC = "use_static";
const char* const HNSW_PARAMETER_REVERSED_EDGES = "use_reversed_edges";
const char* const HNSW_PARAMETER_SKIP_RATIO = "skip_ratio";

const char* const PYRAMID_PARAMETER_BASE_CODES = "base_codes";

const char* const INDEX_PARAM = "index_param";

const char PART_SLASH = '/';

// statstic key
const char* const STATSTIC_MEMORY = "memory";
const char* const STATSTIC_INDEX_NAME = "index_name";
const char* const STATSTIC_DATA_NUM = "data_num";

const char* const STATSTIC_KNN_TIME = "knn_time";
const char* const STATSTIC_KNN_IO = "knn_io";
const char* const STATSTIC_KNN_HOP = "knn_hop";
const char* const STATSTIC_KNN_IO_TIME = "knn_io_time";
const char* const STATSTIC_KNN_CACHE_HIT = "knn_cache_hit";
const char* const STATSTIC_RANGE_TIME = "range_time";
const char* const STATSTIC_RANGE_IO = "range_io";
const char* const STATSTIC_RANGE_HOP = "range_hop";
const char* const STATSTIC_RANGE_CACHE_HIT = "range_cache_hit";
const char* const STATSTIC_RANGE_IO_TIME = "range_io_time";

//Error message
const char* const MESSAGE_PARAMETER = "invalid parameter";

// Serialize key
const char* const SERIALIZE_MAGIC_NUM = "MAGIC_NUM";
const char* const SERIALIZE_VERSION = "VERSION";

const char* const SQ4_UNIFORM_TRUNC_RATE = "sq4_uniform_trunc_rate";
const char* const RABITQ_PCA_DIM = "rabitq_pca_dim";
const char* const RABITQ_BITS_PER_DIM_QUERY = "rabitq_bits_per_dim_query";

const char* const RABITQ_USE_FHT = "rabitq_use_fht";
const char* const HGRAPH_SUPPORT_REMOVE = "support_remove";
const char* const HGRAPH_REMOVE_FLAG_BIT = "remove_flag_bit";

const char* const HGRAPH_USE_REORDER = HGRAPH_USE_REORDER_KEY;
const char* const HGRAPH_USE_ELP_OPTIMIZER = HGRAPH_USE_ELP_OPTIMIZER_KEY;
const char* const HGRAPH_IGNORE_REORDER = "ignore_reorder";
const char* const HGRAPH_BUILD_BY_BASE_QUANTIZATION = "build_by_base";
const char* const HGRAPH_BASE_QUANTIZATION_TYPE = "base_quantization_type";
const char* const HGRAPH_GRAPH_MAX_DEGREE = "max_degree";
const char* const HGRAPH_BUILD_EF_CONSTRUCTION = "ef_construction";
const char* const HGRAPH_INIT_CAPACITY = "hgraph_init_capacity";
const char* const HGRAPH_GRAPH_TYPE = "graph_type";
const char* const HGRAPH_GRAPH_STORAGE_TYPE = "graph_storage_type";
const char* const HGRAPH_BUILD_THREAD_COUNT = "build_thread_count";
const char* const HGRAPH_PRECISE_QUANTIZATION_TYPE = "precise_quantization_type";
const char* const HGRAPH_BASE_IO_TYPE = "base_io_type";
const char* const HGRAPH_BASE_PQ_DIM = "base_pq_dim";
const char* const HGRAPH_BASE_FILE_PATH = "base_file_path";
const char* const HGRAPH_PRECISE_IO_TYPE = "precise_io_type";
const char* const HGRAPH_PRECISE_FILE_PATH = "precise_file_path";
const char* const HGRAPH_PARAMETER_EF_RUNTIME = "ef_search";
const char* const HGRAPH_EXTRA_INFO_SIZE = "extra_info_size";
const char* const HGRAPH_USE_EXTRA_INFO_FILTER = "use_extra_info_filter";
const char* const HGRAPH_USE_ATTRIBUTE_FILTER = "use_attribute_filter";

const char* const BRUTE_FORCE_QUANTIZATION_TYPE = "quantization_type";
const char* const BRUTE_FORCE_IO_TYPE = "io_type";

const char* const IVF_USE_RESIDUAL = "use_residual";
const char* const IVF_USE_REORDER = "use_reorder";
const char* const IVF_TRAIN_TYPE = "ivf_train_type";
const char* const IVF_BUCKETS_COUNT = "buckets_count";
const char* const IVF_BASE_QUANTIZATION_TYPE = "base_quantization_type";
const char* const IVF_BASE_IO_TYPE = "base_io_type";
const char* const IVF_BASE_PQ_DIM = "base_pq_dim";
const char* const IVF_BASE_FILE_PATH = "base_file_path";

const char* const GNO_IMI_FIRST_ORDER_BUCKETS_COUNT = "first_order_buckets_count";
const char* const GNO_IMI_SECOND_ORDER_BUCKETS_COUNT = "second_order_buckets_count";

const char* const IVF_PRECISE_QUANTIZATION_TYPE = "precise_quantization_type";
const char* const IVF_PRECISE_IO_TYPE = "precise_io_type";
const char* const IVF_PRECISE_FILE_PATH = "precise_file_path";
const char* const IVF_USE_ATTRIBUTE_FILTER = "use_attribute_filter";
const char* const IVF_THREAD_COUNT = "thread_count";

const char* const SERIAL_MAGIC_BEGIN = "vsag0000";
const char* const SERIAL_MAGIC_END = "0000gasv";
const char* const SERIAL_META_KEY = "_meta";

};  // namespace vsag
