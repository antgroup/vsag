
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

// AVX traits (Width = 8, __m256). MUST only be included from a TU
// compiled with -mavx.

#include <immintrin.h>

#include "simd_traits_generic.h"

namespace vsag::simd {

struct AVX_Tag {};

template <>
struct SimdTraits<AVX_Tag> {
    using FloatVec = __m256;
    static constexpr int Width = 8;

    static inline __attribute__((always_inline)) FloatVec
    zero() {
        return _mm256_setzero_ps();
    }

    static inline __attribute__((always_inline)) FloatVec
    load(const float* p) {
        return _mm256_loadu_ps(p);
    }

    static inline __attribute__((always_inline)) FloatVec
    mul(FloatVec a, FloatVec b) {
        return _mm256_mul_ps(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    add(FloatVec a, FloatVec b) {
        return _mm256_add_ps(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    sub(FloatVec a, FloatVec b) {
        return _mm256_sub_ps(a, b);
    }

    // Plain AVX lacks FMA; emulate to match the original avx.cpp.
    static inline __attribute__((always_inline)) FloatVec
    fmadd(FloatVec a, FloatVec b, FloatVec c) {
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
    }

    static inline __attribute__((always_inline)) FloatVec
    div(FloatVec a, FloatVec b) {
        return _mm256_div_ps(a, b);
    }

    static inline __attribute__((always_inline)) void
    store(float* p, FloatVec v) {
        _mm256_storeu_ps(p, v);
    }

    static inline __attribute__((always_inline)) float
    reduce_add(FloatVec v) {
        // Byte-for-byte match avx_reduce_add_ps in avx.cpp.
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, v);
        return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    }
};

}  // namespace vsag::simd
