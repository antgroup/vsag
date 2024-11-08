// Copyright 2023 Greptime Team
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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define VSAG_MAX_ERROR_MESSAGE_LENGTH 256

struct CError {
  int type_;
  char message[VSAG_MAX_ERROR_MESSAGE_LENGTH];
};

CError *new_error(int type_, const char *msg);
void free_error(const CError *);

const CError *create_index(const char *in_index_type, const char *in_parameters,

                           void **out_index_ptr);

const CError *build_index(void *in_index_ptr, size_t in_num_vectors,
                          size_t in_dim, const int64_t *in_ids,
                          const float *in_vectors,

                          const int64_t **out_failed_ids,
                          size_t *out_num_failed);

const CError *knn_search_index(void *in_index_ptr, size_t in_dim,
                               const float *in_query_vector, size_t in_k,
                               const char *in_search_parameters,

                               const int64_t **out_ids,
                               const float **out_distances,
                               size_t *out_num_results);

const CError *dump_index(void *in_index_ptr, const char *in_file_path);

const CError *load_index(const char *in_file_path, const char *in_index_type,
                         const char *in_parameters,

                         void **out_index_ptr);

void free_index(void *index_ptr);
void free_i64_vector(int64_t *vector);
void free_f32_vector(float *vector);

#ifdef __cplusplus
} // extern "C"
#endif
