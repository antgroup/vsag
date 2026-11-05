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

namespace vsag::simd {

template <typename T>
inline float
SQ4ComputeIPImpl(const float* query,
                 const uint8_t* codes,
                 const float* lower_bound,
                 const float* diff,
                 uint64_t dim,
                 float (*fallback)(const float*,
                                   const uint8_t*,
                                   const float*,
                                   const float*,
                                   uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;
    constexpr int STEP = 2 * W;
    if (dim < static_cast<uint64_t>(STEP)) {
        return fallback ? fallback(query, codes, lower_bound, diff, dim) : 0.0f;
    }
    V sum0 = T::zero();
    V sum1 = T::zero();
    uint64_t d = 0;
    for (; d + STEP <= dim; d += STEP) {
        V dec0, dec1;
        T::load_nibbles_2x_as_float(codes + (d >> 1), dec0, dec1);
        V val0 = T::fmadd(dec0, T::load(diff + d), T::load(lower_bound + d));
        V val1 = T::fmadd(dec1, T::load(diff + d + W), T::load(lower_bound + d + W));
        sum0 = T::fmadd(T::load(query + d), val0, sum0);
        sum1 = T::fmadd(T::load(query + d + W), val1, sum1);
    }
    float result = T::reduce_add(T::add(sum0, sum1));
    if (d < dim) {
        if (fallback)
            result += fallback(query + d, codes + (d >> 1), lower_bound + d, diff + d, dim - d);
    }
    return result;
}

template <typename T>
inline float
SQ4ComputeL2SqrImpl(const float* query,
                    const uint8_t* codes,
                    const float* lower_bound,
                    const float* diff,
                    uint64_t dim,
                    float (*fallback)(const float*,
                                      const uint8_t*,
                                      const float*,
                                      const float*,
                                      uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;
    constexpr int STEP = 2 * W;
    if (dim < static_cast<uint64_t>(STEP)) {
        return fallback ? fallback(query, codes, lower_bound, diff, dim) : 0.0f;
    }
    V sum0 = T::zero();
    V sum1 = T::zero();
    uint64_t d = 0;
    for (; d + STEP <= dim; d += STEP) {
        V dec0, dec1;
        T::load_nibbles_2x_as_float(codes + (d >> 1), dec0, dec1);
        V val0 = T::fmadd(dec0, T::load(diff + d), T::load(lower_bound + d));
        V val1 = T::fmadd(dec1, T::load(diff + d + W), T::load(lower_bound + d + W));
        V d0 = T::sub(T::load(query + d), val0);
        V d1 = T::sub(T::load(query + d + W), val1);
        sum0 = T::fmadd(d0, d0, sum0);
        sum1 = T::fmadd(d1, d1, sum1);
    }
    float result = T::reduce_add(T::add(sum0, sum1));
    if (d < dim) {
        if (fallback)
            result += fallback(query + d, codes + (d >> 1), lower_bound + d, diff + d, dim - d);
    }
    return result;
}

template <typename T>
inline float
SQ4ComputeCodesIPImpl(const uint8_t* codes1,
                      const uint8_t* codes2,
                      const float* lower_bound,
                      const float* diff,
                      uint64_t dim,
                      float (*fallback)(const uint8_t*,
                                        const uint8_t*,
                                        const float*,
                                        const float*,
                                        uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;
    constexpr int STEP = 2 * W;
    if (dim < static_cast<uint64_t>(STEP)) {
        return fallback ? fallback(codes1, codes2, lower_bound, diff, dim) : 0.0f;
    }
    V sum0 = T::zero();
    V sum1 = T::zero();
    uint64_t d = 0;
    for (; d + STEP <= dim; d += STEP) {
        V dec1_0, dec1_1, dec2_0, dec2_1;
        T::load_nibbles_2x_as_float(codes1 + (d >> 1), dec1_0, dec1_1);
        T::load_nibbles_2x_as_float(codes2 + (d >> 1), dec2_0, dec2_1);
        V diff0 = T::load(diff + d);
        V diff1 = T::load(diff + d + W);
        V lb0 = T::load(lower_bound + d);
        V lb1 = T::load(lower_bound + d + W);
        V a0 = T::fmadd(dec1_0, diff0, lb0);
        V a1 = T::fmadd(dec1_1, diff1, lb1);
        V b0 = T::fmadd(dec2_0, diff0, lb0);
        V b1 = T::fmadd(dec2_1, diff1, lb1);
        sum0 = T::fmadd(a0, b0, sum0);
        sum1 = T::fmadd(a1, b1, sum1);
    }
    float result = T::reduce_add(T::add(sum0, sum1));
    if (d < dim) {
        if (fallback)
            result +=
                fallback(codes1 + (d >> 1), codes2 + (d >> 1), lower_bound + d, diff + d, dim - d);
    }
    return result;
}

template <typename T>
inline float
SQ4ComputeCodesL2SqrImpl(const uint8_t* codes1,
                         const uint8_t* codes2,
                         const float* lower_bound,
                         const float* diff,
                         uint64_t dim,
                         float (*fallback)(const uint8_t*,
                                           const uint8_t*,
                                           const float*,
                                           const float*,
                                           uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;
    constexpr int STEP = 2 * W;
    if (dim < static_cast<uint64_t>(STEP)) {
        return fallback ? fallback(codes1, codes2, lower_bound, diff, dim) : 0.0f;
    }
    V sum0 = T::zero();
    V sum1 = T::zero();
    uint64_t d = 0;
    for (; d + STEP <= dim; d += STEP) {
        V dec1_0, dec1_1, dec2_0, dec2_1;
        T::load_nibbles_2x_as_float(codes1 + (d >> 1), dec1_0, dec1_1);
        T::load_nibbles_2x_as_float(codes2 + (d >> 1), dec2_0, dec2_1);
        V diff0 = T::load(diff + d);
        V diff1 = T::load(diff + d + W);
        V lb0 = T::load(lower_bound + d);
        V lb1 = T::load(lower_bound + d + W);
        V a0 = T::fmadd(dec1_0, diff0, lb0);
        V a1 = T::fmadd(dec1_1, diff1, lb1);
        V b0 = T::fmadd(dec2_0, diff0, lb0);
        V b1 = T::fmadd(dec2_1, diff1, lb1);
        V e0 = T::sub(a0, b0);
        V e1 = T::sub(a1, b1);
        sum0 = T::fmadd(e0, e0, sum0);
        sum1 = T::fmadd(e1, e1, sum1);
    }
    float result = T::reduce_add(T::add(sum0, sum1));
    if (d < dim) {
        if (fallback)
            result +=
                fallback(codes1 + (d >> 1), codes2 + (d >> 1), lower_bound + d, diff + d, dim - d);
    }
    return result;
}

}  // namespace vsag::simd
