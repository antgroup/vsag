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

#include <stdint.h>

/*
 * The VSAG C ABI. It is the base for language bindings (Python, Java, Go, Rust, Node.js) and
 * for hosts that embed VSAG without a C++ toolchain.
 *
 * Conventions:
 *  - Every object is an opaque handle with exactly one `_destroy` function. Destroying NULL is
 *    a no-op.
 *  - Every fallible function returns `vsag_error_t*`: NULL on success, otherwise an error
 *    object the caller must release with vsag_error_destroy(). No exception crosses this
 *    boundary and no error state is kept per thread.
 *  - Input buffers are borrowed, never copied or freed. Unless stated otherwise they only have
 *    to stay valid for the duration of the call they are passed to. Buffers attached to a
 *    dataset or a bitmap filter must stay valid for as long as that handle is used.
 *  - Pointers returned by getters stay valid until the handle they were read from is destroyed.
 *    Owned strings returned by analysis must be released with vsag_string_destroy().
 *  - Signatures use fixed-width integers only; flags are int32_t (0 is false).
 *  - Index and search parameters are JSON strings, identical to the C++ API.
 *  - Thread safety follows the C++ library: an index can be searched concurrently. A dataset,
 *    filter or search result handle may be read concurrently but must not be modified or
 *    destroyed while another thread uses it.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define VSAG_C_API_VERSION 1

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

#define VSAG_LOG_TRACE 0
#define VSAG_LOG_DEBUG 1
#define VSAG_LOG_INFO 2
#define VSAG_LOG_WARN 3
#define VSAG_LOG_ERROR 4
#define VSAG_LOG_CRITICAL 5
#define VSAG_LOG_OFF 6

#define VSAG_FILTER_ALLOW 0
#define VSAG_FILTER_DENY 1

typedef struct vsag_error vsag_error_t;
typedef struct vsag_dataset vsag_dataset_t;
typedef struct vsag_engine vsag_engine_t;
typedef struct vsag_engine_options vsag_engine_options_t;
typedef struct vsag_filter vsag_filter_t;
typedef struct vsag_index vsag_index_t;
typedef struct vsag_search_result vsag_search_result_t;

/** The value of VSAG_C_API_VERSION the library was built with. */
uint32_t
vsag_c_api_version(void);

/* ---------------------------------------------------------------------------------------------
 * Errors
 * ------------------------------------------------------------------------------------------ */

/** One of the VSAG_* error codes; VSAG_SUCCESS for NULL. */
int32_t
vsag_error_code(const vsag_error_t* error);

/** A NUL-terminated message; an empty string for NULL. */
const char*
vsag_error_message(const vsag_error_t* error);

void
vsag_error_destroy(vsag_error_t* error);

/* ---------------------------------------------------------------------------------------------
 * Logging (host extension point)
 * ------------------------------------------------------------------------------------------ */

/**
 * Receives one log record. It may be called concurrently from any thread, including threads
 * created by the library, so it must be thread-safe. It must not throw or unwind into the
 * library and must not keep `message` after it returns.
 */
typedef void (*vsag_log_fn)(void* user_data, int32_t level, const char* message);

/** Sets the minimum level (VSAG_LOG_*) of records that are emitted. */
vsag_error_t*
vsag_set_log_level(int32_t level);

/**
 * Routes library log records to `log_fn`. Passing NULL restores the built-in logger.
 * `user_data` must stay valid until the logger is replaced or the process exits.
 *
 * The first call with a non-NULL callback installs a forwarding logger; do it before other
 * threads start logging. Later calls only swap the callback, so they are safe while other
 * threads log, and a record that is being delivered may still reach the previous callback.
 */
vsag_error_t*
vsag_set_logger(vsag_log_fn log_fn, void* user_data);

/* ---------------------------------------------------------------------------------------------
 * Dataset: a view over caller-owned vectors, used for both base data and queries
 * ------------------------------------------------------------------------------------------ */

vsag_error_t*
vsag_dataset_create(vsag_dataset_t** out_dataset);

void
vsag_dataset_destroy(vsag_dataset_t* dataset);

vsag_error_t*
vsag_dataset_set_num_elements(vsag_dataset_t* dataset, int64_t num_elements);

vsag_error_t*
vsag_dataset_set_dim(vsag_dataset_t* dataset, int64_t dim);

/** `ids` holds num_elements entries. */
vsag_error_t*
vsag_dataset_set_ids(vsag_dataset_t* dataset, const int64_t* ids);

/** `vectors` holds num_elements * dim values in row-major order. */
vsag_error_t*
vsag_dataset_set_float32_vectors(vsag_dataset_t* dataset, const float* vectors);

/** 16-bit vectors as raw bits, for indexes created with dtype float16 or bfloat16. */
vsag_error_t*
vsag_dataset_set_float16_vectors(vsag_dataset_t* dataset, const uint16_t* vectors);

vsag_error_t*
vsag_dataset_set_int8_vectors(vsag_dataset_t* dataset, const int8_t* vectors);

/**
 * Sparse vectors in CSR form: row i owns indices[index_pointers[i] .. index_pointers[i + 1])
 * and the matching values. `index_pointers` holds num_elements + 1 entries and starts at 0.
 */
vsag_error_t*
vsag_dataset_set_sparse_vectors(vsag_dataset_t* dataset,
                                const uint32_t* index_pointers,
                                const uint32_t* indices,
                                const float* values);

/* ---------------------------------------------------------------------------------------------
 * Filters: data only, so no call crosses the boundary per candidate
 * ------------------------------------------------------------------------------------------ */

/**
 * Keeps (VSAG_FILTER_ALLOW) or drops (VSAG_FILTER_DENY) the listed ids. The list is copied.
 */
vsag_error_t*
vsag_filter_create_id_list(const int64_t* ids,
                           uint64_t count,
                           int32_t mode,
                           vsag_filter_t** out_filter);

/**
 * Keeps id i when bit i is set: (bitmap[i / 8] >> (i % 8)) & 1. Ids outside [0, num_ids) are
 * dropped. The bitmap is borrowed and must outlive the filter.
 */
vsag_error_t*
vsag_filter_create_bitmap(const uint8_t* bitmap, uint64_t num_ids, vsag_filter_t** out_filter);

/** Keeps ids in the inclusive range [min_id, max_id]. */
vsag_error_t*
vsag_filter_create_id_range(int64_t min_id, int64_t max_id, vsag_filter_t** out_filter);

void
vsag_filter_destroy(vsag_filter_t* filter);

/**
 * Host extension point: decides per candidate. Returns non-zero to keep `id` and 0 to drop it.
 * It is called for every candidate the search considers, possibly concurrently from any thread
 * the library uses, so it must be thread-safe, cheap, must not throw or unwind into the library,
 * and must not call back into VSAG. `user_data` must stay valid while the filter is in use.
 * Prefer the data filters above when the set of valid ids is known up front.
 */
typedef int32_t (*vsag_filter_fn)(void* user_data, int64_t id);

vsag_error_t*
vsag_filter_create_callback(vsag_filter_fn filter_fn, void* user_data, vsag_filter_t** out_filter);

/* ---------------------------------------------------------------------------------------------
 * Engine: indexes that use host-provided memory and threads (host extension point)
 *
 * An engine bundles an allocator and a thread pool supplied by the host and creates indexes that
 * use them. Every callback below may be called concurrently from any thread, must not throw or
 * unwind into the library, and receives the `user_data` it was registered with. `user_data` and
 * whatever it refers to must stay valid until the engine and every index created from it have
 * been destroyed; the library keeps the resources alive until then.
 * ------------------------------------------------------------------------------------------ */

/** Returns `size` bytes, or NULL on failure; the library then reports VSAG_NO_ENOUGH_MEMORY. */
typedef void* (*vsag_allocate_fn)(void* user_data, uint64_t size);

/** Releases a block returned by allocate or reallocate. */
typedef void (*vsag_deallocate_fn)(void* user_data, void* ptr);

/** Resizes a block, preserving its contents, and returns the new block or NULL on failure. */
typedef void* (*vsag_reallocate_fn)(void* user_data, void* ptr, uint64_t size);

/** A unit of work the library hands to the host pool; run it exactly once on any thread. */
typedef void (*vsag_task_fn)(void* task_data);

/**
 * Schedules `task(task_data)` on a host thread and returns 0, or returns non-zero when the task
 * cannot be accepted, in which case the library runs it on the calling thread. The pool is
 * owned by the host, so the library does not resize it.
 */
typedef int32_t (*vsag_submit_fn)(void* user_data, vsag_task_fn task, void* task_data);

vsag_error_t*
vsag_engine_options_create(vsag_engine_options_t** out_options);

void
vsag_engine_options_destroy(vsag_engine_options_t* options);

/** Uses the host allocator; `name` is only used in diagnostics and is copied. */
vsag_error_t*
vsag_engine_options_set_allocator(vsag_engine_options_t* options,
                                  const char* name,
                                  vsag_allocate_fn allocate_fn,
                                  vsag_deallocate_fn deallocate_fn,
                                  vsag_reallocate_fn reallocate_fn,
                                  void* user_data);

/** Runs parallel work on the host pool. */
vsag_error_t*
vsag_engine_options_set_thread_pool(vsag_engine_options_t* options,
                                    vsag_submit_fn submit_fn,
                                    void* user_data);

/** `options` may be NULL for the built-in allocator and pool; it is copied. */
vsag_error_t*
vsag_engine_create(const vsag_engine_options_t* options, vsag_engine_t** out_engine);

/** Releasing the handle does not invalidate indexes created from the engine. */
void
vsag_engine_destroy(vsag_engine_t* engine);

vsag_error_t*
vsag_engine_create_index(vsag_engine_t* engine,
                         const char* index_name,
                         const char* index_param,
                         vsag_index_t** out_index);

/* ---------------------------------------------------------------------------------------------
 * Search results
 * ------------------------------------------------------------------------------------------ */

int64_t
vsag_search_result_count(const vsag_search_result_t* result);

/** `count` ids, nearest first. */
const int64_t*
vsag_search_result_ids(const vsag_search_result_t* result);

/** `count` distances, matching vsag_search_result_ids(). */
const float*
vsag_search_result_distances(const vsag_search_result_t* result);

/** The search statistics as a JSON string. */
const char*
vsag_search_result_statistics(const vsag_search_result_t* result);

void
vsag_search_result_destroy(vsag_search_result_t* result);

/* ---------------------------------------------------------------------------------------------
 * Index
 * ------------------------------------------------------------------------------------------ */

vsag_error_t*
vsag_index_create(const char* index_name, const char* index_param, vsag_index_t** out_index);

void
vsag_index_destroy(vsag_index_t* index);

vsag_error_t*
vsag_index_build(vsag_index_t* index, const vsag_dataset_t* base);

vsag_error_t*
vsag_index_train(vsag_index_t* index, const vsag_dataset_t* data);

/** `out_failed_count` (may be NULL) receives how many vectors were not inserted. */
vsag_error_t*
vsag_index_add(vsag_index_t* index, const vsag_dataset_t* base, uint64_t* out_failed_count);

vsag_error_t*
vsag_index_remove(vsag_index_t* index,
                  const int64_t* ids,
                  uint64_t count,
                  uint64_t* out_removed_count);

/** Renames `count` ids; stops at the first failure. */
vsag_error_t*
vsag_index_update_ids(vsag_index_t* index,
                      const int64_t* old_ids,
                      const int64_t* new_ids,
                      uint64_t count);

/** Replaces the vector of `id` with the single vector in `vector`. */
vsag_error_t*
vsag_index_update_vector(vsag_index_t* index,
                         int64_t id,
                         const vsag_dataset_t* vector,
                         int32_t force_update);

/**
 * Searches the k nearest neighbors of the single query in `query`. `filter` may be NULL.
 * The caller destroys `*out_result`.
 */
vsag_error_t*
vsag_index_knn_search(const vsag_index_t* index,
                      const vsag_dataset_t* query,
                      int64_t k,
                      const char* parameters,
                      const vsag_filter_t* filter,
                      vsag_search_result_t** out_result);

/**
 * Returns the neighbors of the single query in `query` within `radius`. `limit` caps the
 * number of results; -1 means no cap. `filter` may be NULL. The caller destroys `*out_result`.
 */
vsag_error_t*
vsag_index_range_search(const vsag_index_t* index,
                        const vsag_dataset_t* query,
                        float radius,
                        int64_t limit,
                        const char* parameters,
                        const vsag_filter_t* filter,
                        vsag_search_result_t** out_result);

/**
 * Hot path for dense float32 queries: one call, no handles and no allocation by the caller.
 * Runs `num_queries` searches one after another on the calling thread. `queries` holds
 * num_queries * dim values. `out_ids` and `out_distances` hold num_queries * k entries; row i
 * receives the results of query i, and slots beyond `out_counts[i]` are left untouched.
 * `filter` and `out_counts` may be NULL.
 */
vsag_error_t*
vsag_index_knn_search_float32(const vsag_index_t* index,
                              const float* queries,
                              uint64_t num_queries,
                              uint64_t dim,
                              int64_t k,
                              const char* parameters,
                              const vsag_filter_t* filter,
                              int64_t* out_ids,
                              float* out_distances,
                              int64_t* out_counts);

/** Writes the distances between the single query in `query` and `count` stored vectors. */
vsag_error_t*
vsag_index_calc_distances_by_ids(const vsag_index_t* index,
                                 const vsag_dataset_t* query,
                                 const int64_t* ids,
                                 uint64_t count,
                                 float* out_distances);

/** Copies `count` stored float32 vectors into `out_vectors` (count * dim values). */
vsag_error_t*
vsag_index_get_vectors_by_ids(const vsag_index_t* index,
                              const int64_t* ids,
                              uint64_t count,
                              float* out_vectors);

vsag_error_t*
vsag_index_clone(const vsag_index_t* index, vsag_index_t** out_index);

vsag_error_t*
vsag_index_export_model(const vsag_index_t* index, vsag_index_t** out_index);

vsag_error_t*
vsag_index_serialize_file(const vsag_index_t* index, const char* file_path);

vsag_error_t*
vsag_index_deserialize_file(vsag_index_t* index, const char* file_path);

/* ---------------------------------------------------------------------------------------------
 * Serialization through host storage (host extension point)
 *
 * The callbacks return 0 on success and any other value on failure; a failure makes the call
 * return an error. They are invoked synchronously on the calling thread, must not throw or
 * unwind into the library, and must not keep the pointers they receive after returning.
 * `user_data` is passed through unchanged and only has to stay valid for the duration of the
 * call.
 * ------------------------------------------------------------------------------------------ */

/** Stores `size` bytes of `data` at `offset` of the output. */
typedef int32_t (*vsag_write_fn)(void* user_data, uint64_t offset, uint64_t size, const void* data);

/** Fills `dest` with `size` bytes that start at `offset` of the input. */
typedef int32_t (*vsag_read_fn)(void* user_data, uint64_t offset, uint64_t size, void* dest);

/** The total size of the input in bytes. */
typedef uint64_t (*vsag_size_fn)(void* user_data);

/**
 * Serializes the index by passing its bytes to `write_fn`. A failing `write_fn` is reported as
 * VSAG_INTERNAL_ERROR, because there is no dedicated write error code; a failing read callback
 * during deserialization is reported as VSAG_READ_ERROR.
 */
vsag_error_t*
vsag_index_serialize_writer(const vsag_index_t* index, vsag_write_fn write_fn, void* user_data);

/** Deserializes the index by reading its bytes through `read_fn`; `size_fn` is required. */
vsag_error_t*
vsag_index_deserialize_reader(vsag_index_t* index,
                              vsag_read_fn read_fn,
                              vsag_size_fn size_fn,
                              void* user_data);

vsag_error_t*
vsag_index_get_num_elements(const vsag_index_t* index, int64_t* out_num_elements);

vsag_error_t*
vsag_index_get_memory_usage(const vsag_index_t* index, int64_t* out_memory_usage);

vsag_error_t*
vsag_index_check_id_exist(const vsag_index_t* index, int64_t id, int32_t* out_exist);

vsag_error_t*
vsag_index_get_min_max_id(const vsag_index_t* index, int64_t* out_min_id, int64_t* out_max_id);

/* ---------------------------------------------------------------------------------------------
 * Index analysis
 * ------------------------------------------------------------------------------------------ */

/**
 * Analyzes a serialized index and returns its JSON report. All string arguments are required;
 * use an empty string for an unused optional path. "default" reuses embedded index/search
 * parameters, and query_data_type accepts "auto", "dense" or "sparse".
 *
 * On success, `*out_json` is an owned NUL-terminated string that the caller must release with
 * vsag_string_destroy(). On failure, `*out_json` is NULL.
 */
vsag_error_t*
vsag_analyze_index(const char* index_path,
                   const char* build_parameter,
                   const char* query_path,
                   const char* query_data_type,
                   const char* base_path,
                   const char* groundtruth_path,
                   const char* save_groundtruth_path,
                   const char* search_parameter,
                   int64_t topk,
                   char** out_json);

/** Releases a string returned by vsag_analyze_index(). Destroying NULL is a no-op. */
void
vsag_string_destroy(char* string);

#ifdef __cplusplus
}
#endif
