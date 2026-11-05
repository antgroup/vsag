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

#include <cstdint>

// PQ distance kernel: for each of the 256 float centers, compute
//   result[i] += (centers[i] - query_val)^2
//
// This is the core of PQ distance table construction. Parameterized
// on SimdTraits<Tag> which must expose: FloatVec, Width, set1, load,
// sub, mul, add, store.

namespace vsag::simd {

using PQDistFallback = void (*)(const void*, float, void*);

template <typename T>
inline void
PQDistanceFloat256Impl(const void* single_dim_centers,
                       float single_dim_val,
                       void* result,
                       PQDistFallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    const auto* float_centers = static_cast<const float*>(single_dim_centers);
    auto* float_result = static_cast<float*>(result);

    V v_query = T::set1(single_dim_val);

    for (uint64_t idx = 0; idx < 256; idx += W) {
        V v_centers = T::load(float_centers + idx);
        V v_diff = T::sub(v_centers, v_query);
        V v_diff_sq = T::mul(v_diff, v_diff);
        V v_chunk = T::load(float_result + idx);
        v_chunk = T::add(v_chunk, v_diff_sq);
        T::store(float_result + idx, v_chunk);
    }
}

}  // namespace vsag::simd
