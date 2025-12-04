
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

extern "C" {
#include <stdint.h>

#define VSAG_SUCCESS 0
#define VSAG_UNKNOWN_ERROR -1
#define VSAG_INTERNAL_ERROR -2
#define VSAG_INVALID_ARGUMENT -3
#define VSAG_WRONG_STATUS -4
#define VSAG_BUILD_TWICE -5
#define VSAG_INDEX_NOT_EMPTY -6
#define VSAG_UNSUPPORTED_INDEX -7
#define VSAG_UNSUPPORTED_INDEX_OPERATION -8
#define VSAG_DIMENSION_NOT_EQUAL -9
#define VSAG_INDEX_EMPTY -10
#define VSAG_NO_ENOUGH_MEMORY -11
#define VSAG_READ_ERROR -12
#define VSAG_MISSING_FILE -13
#define VSAG_INVALID_BINARY -14

typedef struct Error {
    int code;
    char message[1024];
} Error_t; /** The error code. */

typedef void* vsag_index_t; /** The vsag index handle. */

typedef bool (*FilterFunc_t)(int64_t id);
typedef struct SearchResult {
    float* dists;       /** The distances of the results. */
    int64_t* ids;       /** The ids of the results. */
    int64_t count;      /** The count of the results. */
    void* other_result; /** The other result of the search. */
} SearchResult_t;       /** The search result. */

/**
 * @brief Create a index factory object.
 *
 * @param index_name The name of the index.
 * @param index_param The parameter of the index.
 * @return vsag_index_t The vsag index handle.
 */
vsag_index_t
vsag_index_factory(const char* index_name, const char* index_param);

/**
 * @brief Destroy the index factory object.
 *
 * @param index The vsag index handle.
 * @return Error_t The error code.
 */
Error_t
vsag_index_destroy(vsag_index_t index);

/**
 * @brief Build the index, include train and add.
 *
 * @param index The vsag index handle.
 * @param data The data to build the index.
 * @param ids The ids of the data.
 * @param dim The dimension of the data.
 * @param count The count of the data.
 * @return Error_t The error code.
 */
Error_t
vsag_index_build(
    vsag_index_t index, const float* data, const int64_t* ids, uint64_t dim, uint64_t count);

/**
 * @brief Add more data to the index, suppose the index is trained.
 *
 * @param index The vsag index handle.
 * @param data The data to add to the index.
 * @param ids The ids of the data.
 * @param dim The dimension of the data.
 * @param count The count of the data.
 * @return Error_t The error code.
 */
Error_t
vsag_index_add(vsag_index_t index,
               const float* data,
               const int64_t* ids,
               const uint64_t dim,
               const uint64_t count);

/**
 * @brief Train the index, no data is added to the index.
 *
 * @param index The vsag index handle.
 * @param data The data to train the index.
 * @param ids The ids of the data.
 * @param dim The dimension of the data.
 * @param count The count of the data.
 * @return Error_t The error code.
 */
Error_t
vsag_index_train(vsag_index_t index,
                 const float* data,
                 const int64_t* ids,
                 const uint64_t dim,
                 const uint64_t count);

/**
 * @brief Knn Search the index.
 *
 * @param index The vsag index handle.
 * @param query The query data.
 * @param dim The dimension of the query data.
 * @param k The top k results.
 * @param parameters The parameters of the search.
 * @param result The result of the search.
 * @return Error_t The error code.
 */
Error_t
vsag_index_knn_search(vsag_index_t index,
                      const float* query,
                      uint64_t dim,
                      int64_t k,
                      const char* parameters,
                      SearchResult_t* search_result);

/**
 * @brief Knn Search the index with filter.
 *
 * @param index The vsag index handle.
 * @param query The query data.
 * @param dim The dimension of the query data.
 * @param k The top k results.
 * @param parameters The parameters of the search.
 * @param filter The filter function.
 * @param result The result of the search.
 * @return Error_t The error code.
 */
Error_t
vsag_index_knn_search_with_filter(vsag_index_t index,
                                  const float* query,
                                  uint64_t dim,
                                  int64_t k,
                                  const char* parameters,
                                  FilterFunc_t filter,
                                  SearchResult_t* search_result);

/**
 * @brief Range Search the index.
 *
 * @param index The vsag index handle.
 * @param query The query data.
 * @param dim The dimension of the query data.
 * @param radius The radius of the search.
 * @param parameters The parameters of the search.
 * @param result The result of the search.
 * @return Error_t The error code.
 */
Error_t
vsag_index_range_search(vsag_index_t index,
                        const float* query,
                        uint64_t dim,
                        float radius,
                        const char* parameters,
                        SearchResult_t* search_result);

/**
 * @brief Range Search the index with filter.
 *
 * @param index The vsag index handle.
 * @param query The query data.
 * @param dim The dimension of the query data.
 * @param radius The radius of the search.
 * @param parameters The parameters of the search.
 * @param filter The filter function.
 * @param result The result of the search.
 * @return Error_t The error code.
 */
Error_t
vsag_index_range_search_with_filter(vsag_index_t index,
                                    const float* query,
                                    uint64_t dim,
                                    float radius,
                                    const char* parameters,
                                    FilterFunc_t filter,
                                    SearchResult_t* search_result);

/**
 * @brief Clone the index.
 *
 * @param index The vsag index handle.
 * @param clone_index The output cloned vsag index handle's pointer.
 * @return Error_t The error code.
 */
Error_t
vsag_index_clone(const vsag_index_t index, vsag_index_t* clone_index);

/**
 * @brief Export the model index.
 *
 * @param index The vsag index handle.
 * @param model_index The output vsag index handle's pointer(empty index as model).
 * @return Error_t The error code.
 */
Error_t
vsag_index_export_model(const vsag_index_t index, vsag_index_t* model_index);

/**
 * @brief Calculate the distance between query and ids.
 *
 * @param index The vsag index handle.
 * @param query The query data.
 * @param dim The dimension of the query data.
 * @param ids The ids of the data.
 * @param count The count of the ids.
 * @param dists The output distances of the ids.
 * @return Error_t The error code.
 */
Error_t
vsag_index_calculate_distance_by_ids(const vsag_index_t index,
                                     const float* query,
                                     const uint64_t dim,
                                     const int64_t* ids,
                                     const uint64_t count,
                                     float* dists);

/**
 * @brief Update the ids of the data.
 *
 * @param index The vsag index handle.
 * @param old_ids The old ids of the data.
 * @param new_ids The new ids of the data.
 * @param dim The dimension of the data.
 * @param count The count of the data.
 * @return Error_t The error code.
 */
Error_t
vsag_index_update_ids(vsag_index_t index,
                      const int64_t* old_ids,
                      const int64_t* new_ids,
                      const uint64_t dim,
                      const uint64_t count);
/**
 * @brief Update the vector of the data, replace the origin vector with new vector.
 *        No suppose the origin and new vector are similar.
 *
 * @param index The vsag index handle.
 * @param id The id of the data.
 * @param new_data The new vector of the data.
 * @param dim The dimension of the data.
 * @return Error_t The error code.
 */
Error_t
vsag_index_update_vector(vsag_index_t index,
                         const int64_t id,
                         const float* new_data,
                         const uint64_t dim);

/**
 * @brief Update the vector of the data, force replace. 
 * Suppose the origin and new vector are similar.
 *
 * @param index The vsag index handle.
 * @param id The id of the data.
 * @param new_data The new vector of the data.
 * @param dim The dimension of the data.
 * @return Error_t The error code.
 */
Error_t
vsag_index_update_vector_force(vsag_index_t index,
                               const int64_t id,
                               const float* new_data,
                               const uint64_t dim);
/**
 * @brief Get the vectors of the data by ids.
 *
 * @param index The vsag index handle.
 * @param ids The ids of the data.
 * @param count The count of the ids.
 * @param vectors The output vectors of the data.
 * @return Error_t The error code.
 */
Error_t
vsag_index_get_vector_by_ids(vsag_index_t index,
                             const int64_t* ids,
                             const uint64_t count,
                             float* vectors);

/**
 * @brief Serialize the index to a file.
 *
 * @param index The vsag index handle.
 * @param file_path The file path to serialize the index.
 * @return Error_t The error code.
 */
Error_t
vsag_serialize_file(vsag_index_t index, const char* file_path);

Error_t
vsag_deserialize_file(vsag_index_t index, const char* file_path);

typedef uint64_t OffsetType;
typedef uint64_t SizeType;

Error_t
vsag_serialize_write_func(vsag_index_t index,
                          void (*write_func)(OffsetType offset, SizeType size, const void* data));

Error_t
vsag_deserialize_read_func(vsag_index_t index,
                           void (*read_func)(OffsetType offset, SizeType size, void* data),
                           SizeType (*size_func)());
}
