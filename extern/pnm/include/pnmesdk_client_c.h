#ifndef PNMSDK_CLIENT_C_H_
#define PNMSDK_CLIENT_C_H_

#include "pnm_engine_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the client
 *
 * @param config: Configuration parameters for the client
 * @return int  return == 0 on success
 *              return < 0  on fail
 */
int
pnmesdk_init(pnmesdk_conf* config);

/**
 * @brief Uninitialize the client
 *
 * @param config: Configuration parameters for the client
 * @return int  return == 0 on success
 *              return < 0  on fail
 */
int
pnmesdk_uninit(pnmesdk_conf* config);

/**
 * @brief Open or create a database
 *
 * @param context: Description context of the database
 * @param database_id: The ID of the database
 * @param database_name: The name of the database
 * @return uint64_t  The ID of the created or opened database.
 *
 */
uint64_t
pnmesdk_db_open(database_context* context, uint64_t database_id, char* database_name);

/**
 * @brief Upload raw data to the device
 *
 * @param context Description context of the database
 * @param data:  Address of the data to be written
 * @param data_size: The size of the data to be written
 * @return int  return == 0 on success
 *              return < 0  on fail
 */
int
pnmesdk_db_storage(database_context* context, char* data, uint64_t data_size);

/**
 * @brief Upload raw data to the device
 *
 * @param context Description context of the database
 * @param file_path: The file path of the data to be written
 * @param offset:  The offset of the data to be written
 * @param length: The length of the data to be uploaded
 * @return int  return == 0 on success
 *              return < 0  on fail
 */
int
pnmesdk_db_file_storage(database_context* context,
                        char* file_path,
                        uint64_t offset,
                        uint64_t length);

/**
 * @brief Get the list of databases
 *
 * @param info:  Pointer to the database list
 * @param n:  Number of databases to be obtained, This parameter tells the function how many database_info structures can be filled
 * @param offset:  Offset of the database list
 * @return int  return > 0 , the value indicates the number of databases obtained.
 *              return = 0 , No databases can be listed, and there are no databases saved on the device.
 */
int
pnmesdk_db_list(database_info* info, uint32_t n, uint32_t offset);

/**
 * @brief hnsw search
 *
 * @param search_config: Search configuration parameters
 * @param target_vector:  Pointer to the query vector
 * @param target_vector_size: Size of the query vector
 * @param result_data: Returns the IDs of the top-k most similar results
 * @return int  return == 0 on success
 *              return < 0  on fail
 */
int
pnmesdk_hnsw_search(database_context* context,
                    search_config* config,
                    void* target_vector,
                    uint32_t target_vector_size,
                    void* result_data);

// 批量距离计算, 单独透传出来硬件向量计算接口
/**
 * @brief handware vector calculation
 *
 * @param context: Description context of the database
 * @param target_vector:  Pointer to the target vector
 * @param target_vector_size: Size of the target vector
 * @param ids_list: List of vectors to be calculated
 * @param ids_size: Number of vectors to be calculated
 * @param result_list: List to store calculation results. The length of the
 * result_list must match the ids_size, ensuring there is enough space to store
 * one result per id.
 * @return int  return == 0 on success
 *              return < 0  on fail
 */
int
database_context_cal(database_context* context, calculate_config* cal_config);

int
database_context_cal_metric(database_context* context,
                            void* target_vector,
                            uint32_t target_vector_size,
                            uint64_t* ids_list,
                            uint32_t ids_size,
                            void* result_list,
                            int hnsw_query_id,
                            int level);

int
database_context_get_raw_data(database_context* context,
                              uint32_t vector_size,
                              uint64_t* ids_list,
                              uint32_t ids_size,
                              void* raw_data);

bool
pnmesdk_upload_non_level0_data(database_context* context,
                               uint32_t id_nums,
                               uint64_t* ids_list,
                               int level,
                               uint32_t cache_size);

int
pnme_get_search_query_id();
void
pnme_hnsw_search_end(int query_id);

int
pnmesdk_db_del(database_context* context);

/**
 * @brief Build index
 *
 * @param config: Configuration parameters for building the index
 * @param intput_data_path: The path to the input data file; the source data file needs to be in fvecs format.
 * @param output_index_path: The path where the index file will be saved
 * @return int  return == 0 on success
 *              return < 0  on fail
 */
int
pnmesdk_build_index(build_index_config* config, char* intput_data_path, char* output_index_path);

/**
 * @brief Build index for data deduplication
 *
 * @param config: Configuration parameters for building the index or offline data deduplication
 * @param file_path: The Address of the data file
 * @return int  return == 0 on success
 *              return < 0  on fail
 */
int
hnsw_build_index_from_memory(build_index_config* config, float* base_data);
#ifdef __cplusplus
}
#endif

#endif
