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
#include <numeric>
#include <tuple>

#include "container_types.h"
#include "vsag/allocator.h"
#include "vsag/dataset.h"

namespace vsag {

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

// Returns the sparse inner-product distance used by SINDI: 1 - dot(vector1, vector2).
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

}  // namespace vsag
