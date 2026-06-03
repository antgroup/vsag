
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

// SSE traits (Width = 4, __m128).
//
// IMPORTANT: This header uses SSE intrinsics. It MUST only be included
// from translation units compiled with -msse (and friends). Including
// it from sse.cpp is correct; including it from generic.cpp or any
// other ISA TU will trigger "target specific option mismatch" errors.

#include <immintrin.h>

#include "simd_traits_generic.h"

namespace vsag::simd {

struct SSE_Tag {};

template <>
struct SimdTraits<SSE_Tag> {
    using FloatVec = __m128;
    static constexpr int Width = 4;

    static inline __attribute__((always_inline)) FloatVec
    zero() {
        return _mm_setzero_ps();
    }

    static inline __attribute__((always_inline)) FloatVec
    load(const float* p) {
        return _mm_loadu_ps(p);
    }

    static inline __attribute__((always_inline)) FloatVec
    mul(FloatVec a, FloatVec b) {
        return _mm_mul_ps(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    add(FloatVec a, FloatVec b) {
        return _mm_add_ps(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    sub(FloatVec a, FloatVec b) {
        return _mm_sub_ps(a, b);
    }

    // SSE has no FMA; emulate as add(mul(a, b), c). Matches the
    // original sse.cpp implementation byte-for-byte.
    static inline __attribute__((always_inline)) FloatVec
    fmadd(FloatVec a, FloatVec b, FloatVec c) {
        return _mm_add_ps(_mm_mul_ps(a, b), c);
    }

    static inline __attribute__((always_inline)) FloatVec
    div(FloatVec a, FloatVec b) {
        return _mm_div_ps(a, b);
    }

    static inline __attribute__((always_inline)) void
    store(float* p, FloatVec v) {
        _mm_storeu_ps(p, v);
    }

    static inline __attribute__((always_inline)) float
    reduce_add(FloatVec v) {
        // Match the original sse FP32ReduceAdd / FP32ComputeIP shape:
        // the existing sse.cpp uses two different reductions (hadd for
        // FP32ReduceAdd, scalar-sum for FP32ComputeIP). To keep numerical
        // results bit-stable against the FP32ComputeIP baseline we use
        // the scalar-sum form here, which matches the original IP/L2
        // tail. FP32ReduceAdd's hadd path differs only in last-bit
        // ordering and is allowed to diverge.
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, v);
        return tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
};

}  // namespace vsag::simd
