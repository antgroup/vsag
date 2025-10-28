
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
typedef void* vsag_index_t;
vsag_index_t
vsag_index_factory(const char* index_name, const char* index_param);

bool
vsag_index_destroy(vsag_index_t index);

bool
vsag_index_build(
    vsag_index_t index, const float* data, const int64_t* ids, uint64_t dim, uint64_t count);

bool
vsag_index_knn_search(vsag_index_t index,
                      const float* query,
                      uint64_t dim,
                      int64_t k,
                      const char* parameters,
                      float* results,
                      int64_t* ids);

bool
vsag_serialize_file(vsag_index_t index, const char* file_path);

bool
vsag_deserialize_file(vsag_index_t index, const char* file_path);
}
