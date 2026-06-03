
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

// AVX512 traits (Width = 16, __m512). MUST only be included from a TU
// compiled with -mavx512f (and friends).

#include <immintrin.h>

#include "simd_traits_generic.h"

namespace vsag::simd {

struct AVX512_Tag {};

template <>
struct SimdTraits<AVX512_Tag> {
    using FloatVec = __m512;
    static constexpr int Width = 16;

    static inline __attribute__((always_inline)) FloatVec
    zero() {
        return _mm512_setzero_ps();
    }

    static inline __attribute__((always_inline)) FloatVec
    load(const float* p) {
        return _mm512_loadu_ps(p);
    }

    static inline __attribute__((always_inline)) FloatVec
    mul(FloatVec a, FloatVec b) {
        return _mm512_mul_ps(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    add(FloatVec a, FloatVec b) {
        return _mm512_add_ps(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    sub(FloatVec a, FloatVec b) {
        return _mm512_sub_ps(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    fmadd(FloatVec a, FloatVec b, FloatVec c) {
        return _mm512_fmadd_ps(a, b, c);
    }

    static inline __attribute__((always_inline)) FloatVec
    div(FloatVec a, FloatVec b) {
        return _mm512_div_ps(a, b);
    }

    static inline __attribute__((always_inline)) void
    store(float* p, FloatVec v) {
        _mm512_storeu_ps(p, v);
    }

    static inline __attribute__((always_inline)) float
    reduce_add(FloatVec v) {
        return _mm512_reduce_add_ps(v);
    }
};

}  // namespace vsag::simd
