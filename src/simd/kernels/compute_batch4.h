
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

// Batch-of-4 IP / L2 kernel: one query vector against four code vectors.
// Results are accumulated into result1..result4 (the caller must initialise
// them before invocation, e.g. to 0). Matches the existing semantics of
// FP32ComputeIPBatch4 / FP32ComputeL2SqrBatch4: the four accumulators
// share the same query load, so we get 4x reuse of every q-cacheline.

#include <cstdint>

#include "simd/simd_marco.h"

namespace vsag::simd {

using Batch4Fallback = void (*)(const float* RESTRICT query,
                                uint64_t dim,
                                const float* RESTRICT c1,
                                const float* RESTRICT c2,
                                const float* RESTRICT c3,
                                const float* RESTRICT c4,
                                float& r1,
                                float& r2,
                                float& r3,
                                float& r4);

enum class Batch4Kind { IP, L2 };

template <typename T, Batch4Kind Kind>
inline __attribute__((always_inline)) typename T::FloatVec
batch4_accumulate(typename T::FloatVec q, typename T::FloatVec c, typename T::FloatVec acc) {
    if constexpr (Kind == Batch4Kind::IP) {
        return T::fmadd(q, c, acc);
    } else {
        typename T::FloatVec d = T::sub(q, c);
        return T::fmadd(d, d, acc);
    }
}

template <typename T, Batch4Kind Kind>
inline void
ComputeBatch4Impl(const float* RESTRICT query,
                  uint64_t dim,
                  const float* RESTRICT c1,
                  const float* RESTRICT c2,
                  const float* RESTRICT c3,
                  const float* RESTRICT c4,
                  float& r1,
                  float& r2,
                  float& r3,
                  float& r4,
                  Batch4Fallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            fallback(query, dim, c1, c2, c3, c4, r1, r2, r3, r4);
            return;
        }
    }

    V s1 = T::zero();
    V s2 = T::zero();
    V s3 = T::zero();
    V s4 = T::zero();

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V q = T::load(query + i);
        s1 = batch4_accumulate<T, Kind>(q, T::load(c1 + i), s1);
        s2 = batch4_accumulate<T, Kind>(q, T::load(c2 + i), s2);
        s3 = batch4_accumulate<T, Kind>(q, T::load(c3 + i), s3);
        s4 = batch4_accumulate<T, Kind>(q, T::load(c4 + i), s4);
    }
    r1 += T::reduce_add(s1);
    r2 += T::reduce_add(s2);
    r3 += T::reduce_add(s3);
    r4 += T::reduce_add(s4);

    if constexpr (W > 1) {
        if (dim > i) {
            fallback(query + i, dim - i, c1 + i, c2 + i, c3 + i, c4 + i, r1, r2, r3, r4);
        }
    }
}

}  // namespace vsag::simd
