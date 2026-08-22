
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
                  [[maybe_unused]] Batch4Fallback fallback = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    V s1 = T::zero();
    V s2 = T::zero();
    V s3 = T::zero();
    V s4 = T::zero();

    uint64_t i = 0;
    // Four vector chunks per iteration to maximize the number of independent
    // code loads in flight; accumulation order per code is unchanged, so
    // results stay bit-identical.
    for (; i + 4 * W <= dim; i += 4 * W) {
        const V qa = T::load(query + i);
        const V qb = T::load(query + i + W);
        const V qc = T::load(query + i + 2 * W);
        const V qd = T::load(query + i + 3 * W);
        s1 = batch4_accumulate<T, Kind>(qa, T::load(c1 + i), s1);
        s2 = batch4_accumulate<T, Kind>(qa, T::load(c2 + i), s2);
        s3 = batch4_accumulate<T, Kind>(qa, T::load(c3 + i), s3);
        s4 = batch4_accumulate<T, Kind>(qa, T::load(c4 + i), s4);
        s1 = batch4_accumulate<T, Kind>(qb, T::load(c1 + i + W), s1);
        s2 = batch4_accumulate<T, Kind>(qb, T::load(c2 + i + W), s2);
        s3 = batch4_accumulate<T, Kind>(qb, T::load(c3 + i + W), s3);
        s4 = batch4_accumulate<T, Kind>(qb, T::load(c4 + i + W), s4);
        s1 = batch4_accumulate<T, Kind>(qc, T::load(c1 + i + 2 * W), s1);
        s2 = batch4_accumulate<T, Kind>(qc, T::load(c2 + i + 2 * W), s2);
        s3 = batch4_accumulate<T, Kind>(qc, T::load(c3 + i + 2 * W), s3);
        s4 = batch4_accumulate<T, Kind>(qc, T::load(c4 + i + 2 * W), s4);
        s1 = batch4_accumulate<T, Kind>(qd, T::load(c1 + i + 3 * W), s1);
        s2 = batch4_accumulate<T, Kind>(qd, T::load(c2 + i + 3 * W), s2);
        s3 = batch4_accumulate<T, Kind>(qd, T::load(c3 + i + 3 * W), s3);
        s4 = batch4_accumulate<T, Kind>(qd, T::load(c4 + i + 3 * W), s4);
    }
    // Two vector chunks per iteration: the eight code loads in flight overlap
    // their latencies instead of serializing behind one query load.
    for (; i + 2 * W <= dim; i += 2 * W) {
        const V qa = T::load(query + i);
        const V qb = T::load(query + i + W);
        s1 = batch4_accumulate<T, Kind>(qa, T::load(c1 + i), s1);
        s2 = batch4_accumulate<T, Kind>(qa, T::load(c2 + i), s2);
        s3 = batch4_accumulate<T, Kind>(qa, T::load(c3 + i), s3);
        s4 = batch4_accumulate<T, Kind>(qa, T::load(c4 + i), s4);
        s1 = batch4_accumulate<T, Kind>(qb, T::load(c1 + i + W), s1);
        s2 = batch4_accumulate<T, Kind>(qb, T::load(c2 + i + W), s2);
        s3 = batch4_accumulate<T, Kind>(qb, T::load(c3 + i + W), s3);
        s4 = batch4_accumulate<T, Kind>(qb, T::load(c4 + i + W), s4);
    }
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

    // The dim % W remainder is handled inline: falling back to narrower SIMD
    // tiers costs two extra out-of-line calls per invocation on the search hot
    // path (e.g. dim = 100 spilled a 4-wide tail from AVX512 down to SSE).
    if constexpr (Kind == Batch4Kind::IP) {
        for (; i < dim; ++i) {
            r1 += query[i] * c1[i];
            r2 += query[i] * c2[i];
            r3 += query[i] * c3[i];
            r4 += query[i] * c4[i];
        }
    } else {
        for (; i < dim; ++i) {
            const float d1 = query[i] - c1[i];
            const float d2 = query[i] - c2[i];
            const float d3 = query[i] - c3[i];
            const float d4 = query[i] - c4[i];
            r1 += d1 * d1;
            r2 += d2 * d2;
            r3 += d3 * d3;
            r4 += d4 * d4;
        }
    }
}

}  // namespace vsag::simd
