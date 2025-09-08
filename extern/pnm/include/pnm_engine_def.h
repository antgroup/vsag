#ifndef PNM_ENGINE_DEF_H_
#define PNM_ENGINE_DEF_H_

#include <stddef.h>
#include <stdint.h>

#define PNM_ENGINE_VERSION "0.1.0"
#define MAX_SEGMENT_NUM 256

enum data_type {
    INT8 = 1,  // not support
    FP16 = 2,
    FP32 = 4,
};
// 搜索参数
struct search_config {
    uint32_t topk;
    uint32_t ef_search;
};

// 创建索引参数
struct build_index_config {
    size_t vecsize;
    uint16_t vecdim;
    uint8_t data_type;
    uint16_t M;
    uint16_t ef_construction;
    uint16_t thread_nums;
    float similar_threshold;
    build_index_config()
        : vecsize(0),
          vecdim(0),
          data_type(FP32),
          M(48),
          ef_construction(500),
          thread_nums(10),
          similar_threshold(0.01){};
};

struct calculate_config {
    void* target_vector;
    uint32_t target_vector_size;
    uint64_t* ids_list;
    uint32_t ids_size;
    void* result_list;
    int hnsw_query_id;
    int level;
    calculate_config()
        : target_vector(nullptr),
          target_vector_size(0),
          ids_list(nullptr),
          ids_size(0),
          result_list(nullptr),
          hnsw_query_id(0),
          level(-1) {
    }
};
// 描述数据库
struct database_context {
    uint64_t database_id;  // 数据库id
    uint64_t offset;       // reserved
    uint64_t length;       // 标志当前数据库已经写入了多少数据
    uint16_t vecdim;       // 向量维度
    uint8_t data_type;     // fp16 or fp32
    int used_segment_ids[MAX_SEGMENT_NUM];
    int segment_nums;
    database_context()
        : database_id(0), offset(0), length(0), vecdim(0), data_type(FP32), segment_nums(0) {
        for (int i = 0; i < MAX_SEGMENT_NUM; i++) {
            used_segment_ids[i] = -1;
        }
    };
};

struct database_info {
    uint64_t database_id;
    uint8_t data_type;
    uint16_t vecdim;
    char database_name[128];  // user-defined name
};

struct pnmesdk_conf {
    uint32_t timeout;  // 超时时间
};

enum {
    PNM_ENGINE_OK = 0,
    PNM_ENGINE_INVALID_PARAMETER = -1,
    PNM_ENGINE_CONFIG_ERROR = -2,
    PNM_ENGINE_DEVICE_INIT_FAILED = -3,
    PNM_ENGINE_DEVICE_NO_SPACE = -4,
    PNM_ENGINE_UNKNOWN_ERROR = -5,
    PNM_ENGINE_POLLING_FAILED = -6,
    PNM_ENGINE_TIMEOUT = -7,
    PNM_ENGINE_VECTOR_LENGTH_NOT_MATCH = -8,
    PNM_ENGINE_FILE_PATH_NOT_EXIST = -9,
    PNM_ENGINE_VEC_DIM_NOT_EQUAL = -10,
    PNM_ENGINE_LIST_DEVICE_DDR_FAILED = -11,
    PNM_ENGINE_NO_FREE_SEGMENT = -12,
    PNM_ENGINE_DATABASE_NOT_EXIST = -13,
    PNM_ENGINE_VECTOR_ID_TO_ADDRESS_FAILED = -14
};

#endif
