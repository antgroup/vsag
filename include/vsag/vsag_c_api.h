
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
} Error_t;

typedef void* vsag_index_t;
vsag_index_t
vsag_index_factory(const char* index_name, const char* index_param);

Error_t
vsag_index_destroy(vsag_index_t index);

Error_t
vsag_index_build(
    vsag_index_t index, const float* data, const int64_t* ids, uint64_t dim, uint64_t count);

Error_t
vsag_index_knn_search(vsag_index_t index,
                      const float* query,
                      uint64_t dim,
                      int64_t k,
                      const char* parameters,
                      float* results,
                      int64_t* ids);

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
