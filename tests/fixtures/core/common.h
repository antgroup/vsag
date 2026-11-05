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

/**
 * @file common.h
 * @brief Common utilities and helper functions for tests.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.h"
#include "typing.h"

namespace fixtures {

extern const int RABITQ_MIN_RACALL_DIM;

std::vector<int>
get_common_used_dims(uint64_t count = -1, int seed = 369, int limited_dim = -1);

std::vector<int>
get_index_test_dims(uint64_t count = -1, int seed = 369, int limited_dim = -1);

bool
is_path_belong_to(const std::string& a, const std::string& b);

std::string
create_random_string(bool is_full);

uint64_t
GetFileSize(const std::string& filename);

std::vector<std::string>
SplitString(const std::string& s, char delimiter);

std::vector<IOItem>
GenTestItems(uint64_t count, uint64_t max_length, uint64_t max_index = 10000);

template <typename T>
T*
CopyVector(const std::vector<T>& vec) {
    auto result = new T[vec.size()];
    memcpy(result, vec.data(), vec.size() * sizeof(T));
    return result;
}

template <typename T>
T*
DuplicateCopyVector(const std::vector<T>& vec) {
    auto result = new T[vec.size()];
    if (vec.size() % 2 != 0) {
        throw std::runtime_error("Vector size must be even for duplication.");
    }
    memcpy(result, vec.data(), (vec.size() / 2) * sizeof(T));
    memcpy(result + vec.size() / 2, vec.data(), (vec.size() / 2) * sizeof(T));
    return result;
}

template <typename T>
T*
CopyVector(const vsag::Vector<T>& vec, vsag::Allocator* allocator) {
    T* result;
    if (allocator) {
        result = (T*)allocator->Allocate(sizeof(T) * vec.size());
    } else {
        result = new T[vec.size()];
    }
    memcpy(result, vec.data(), vec.size() * sizeof(T));
    return result;
}

template <typename T>
T*
CopyVector(const std::vector<T>& vec, vsag::Allocator* allocator) {
    T* result;
    if (allocator) {
        result = (T*)allocator->Allocate(sizeof(T) * vec.size());
    } else {
        result = new T[vec.size()];
    }
    memcpy(result, vec.data(), vec.size() * sizeof(T));
    return result;
}

}  // namespace fixtures
