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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <tuple>

#include "container_types.h"
#include "vsag/allocator.h"
#include "vsag/dataset.h"

namespace vsag {

inline uint32_t
read_sparse_code_id(const uint8_t* ids, uint32_t index) {
    uint32_t id = 0;
    std::memcpy(&id, ids + index * sizeof(id), sizeof(id));
    return id;
}

inline float
read_sparse_code_val(const uint8_t* vals, uint32_t index) {
    float val = 0.0F;
    std::memcpy(&val, vals + index * sizeof(val), sizeof(val));
    return val;
}

inline std::tuple<Vector<uint32_t>, Vector<float>>
sort_sparse_vector(const SparseVector& vector, Allocator* allocator) {
    Vector<uint32_t> indices(vector.len_, allocator);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](uint32_t left, uint32_t right) {
        return vector.ids_[left] < vector.ids_[right];
    });
    Vector<uint32_t> sorted_ids(vector.len_, allocator);
    Vector<float> sorted_vals(vector.len_, allocator);
    for (uint64_t index = 0; index < vector.len_; ++index) {
        sorted_ids[index] = vector.ids_[indices[index]];
        sorted_vals[index] = vector.vals_[indices[index]];
    }
    return std::make_tuple(sorted_ids, sorted_vals);
}

inline float
get_distance(uint32_t len1,
             const uint32_t* ids1,
             const float* vals1,
             uint32_t len2,
             const uint32_t* ids2,
             const float* vals2) {
    float sum = 0.0F;
    uint32_t i = 0;
    uint32_t j = 0;

    while (i < len1 && j < len2) {
        if (ids1[i] < ids2[j]) {
            i++;
        } else if (ids1[i] > ids2[j]) {
            j++;
        } else {
            sum += vals1[i] * vals2[j];
            i++;
            j++;
        }
    }

    return 1 - sum;
}

inline float
get_distance_from_sparse_code(uint32_t len1,
                              const uint32_t* ids1,
                              const float* vals1,
                              const uint8_t* codes2) {
    uint32_t len2 = 0;
    std::memcpy(&len2, codes2, sizeof(len2));
    const auto* ids2 = codes2 + sizeof(uint32_t);
    const auto* vals2 = codes2 + sizeof(uint32_t) + len2 * sizeof(uint32_t);

    float sum = 0.0F;
    uint32_t idx1 = 0;
    uint32_t idx2 = 0;
    while (idx1 < len1 && idx2 < len2) {
        auto id2 = read_sparse_code_id(ids2, idx2);
        if (ids1[idx1] < id2) {
            idx1++;
        } else if (ids1[idx1] > id2) {
            idx2++;
        } else {
            sum += vals1[idx1] * read_sparse_code_val(vals2, idx2);
            idx1++;
            idx2++;
        }
    }

    return 1 - sum;
}

}  // namespace vsag