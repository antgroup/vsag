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

#include "vsag/vsag_c_api.h"

/*
 * The evaluation extension of the VSAG C ABI. It exposes the measurement core of the
 * `eval_performance` tool (build duration/TPS/memory, search QPS/latency/recall) for in-memory
 * float32 datasets, so language bindings such as `pyvsag.eval` can reuse the exact same
 * measurement code as the C++ tool.
 *
 * It follows the conventions of vsag_c_api.h and adds:
 *  - Configuration and results are JSON strings. The configuration is a flat JSON object keyed by
 *    the `eval_performance` option names (for example `top_k`, `search_param`,
 *    `num_threads_searching`, `enable_recall`, `use_id_based_recall`); unknown keys are rejected.
 *  - A returned result string is owned by the caller and must be released with
 *    vsag_string_destroy().
 *  - Dataset buffers are borrowed and only have to stay valid for the duration of the call.
 */

#ifdef __cplusplus
extern "C" {
#endif

/** Releases a string returned by a vsag_eval_* function. Destroying NULL is a no-op. */
void
vsag_string_destroy(char* str);

/**
 * Sets the process-wide number of threads used to build indexes (1..200) and optionally returns the
 * previous value. Indexes read it when they are created, so call it before creating the index to
 * evaluate; vsag_eval_build() only applies `num_threads_building` of its configuration while it
 * runs and restores the previous value afterwards.
 */
vsag_error_t*
vsag_eval_set_num_threads_building(int32_t num_threads, int32_t* out_previous);

/**
 * Builds `index` from `base` (dense float32 vectors, optional ids) while measuring it.
 *
 * `metric_type` is one of "l2", "ip" or "cosine". `config_json` may be NULL for the defaults.
 * `out_result_json` receives a JSON object such as
 * `{"action": "build", "duration(s)": ..., "tps": ..., "index_memory(B)": ...}`.
 */
vsag_error_t*
vsag_eval_build(vsag_index_t* index,
                const vsag_dataset_t* base,
                const char* metric_type,
                const char* config_json,
                char** out_result_json);

/**
 * Searches `index` with every vector of `queries` (dense float32) while measuring it.
 *
 * `gt_ids` is an optional row-major [num_queries x gt_k] matrix of ground-truth ids; it is required
 * when recall is enabled and `gt_k` must be at least `top_k`. `base` is optional: it is only
 * needed for distance-based recall (the default); pass NULL together with
 * `"use_id_based_recall": true` to compare ids only. `out_result_json` receives a JSON object such
 * as `{"action": "search", "qps": ..., "recall_avg": ..., "latency_avg(ms)": ...}`.
 */
vsag_error_t*
vsag_eval_search(vsag_index_t* index,
                 const vsag_dataset_t* base,
                 const vsag_dataset_t* queries,
                 const int64_t* gt_ids,
                 int64_t gt_k,
                 const char* metric_type,
                 const char* config_json,
                 char** out_result_json);

#ifdef __cplusplus
}
#endif
