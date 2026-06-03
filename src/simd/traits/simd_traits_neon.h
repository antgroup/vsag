
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

// NEON traits (Width = 4, float32x4_t). MUST only be included from a
// TU compiled with -march=armv8-a (i.e. neon.cpp).

#include <arm_neon.h>

#include "simd_traits_generic.h"

namespace vsag::simd {

struct NEON_Tag {};

template <>
struct SimdTraits<NEON_Tag> {
    using FloatVec = float32x4_t;
    static constexpr int Width = 4;

    static inline __attribute__((always_inline)) FloatVec
    zero() {
        return vdupq_n_f32(0.0f);
    }

    static inline __attribute__((always_inline)) FloatVec
    load(const float* p) {
        return vld1q_f32(p);
    }

    static inline __attribute__((always_inline)) FloatVec
    mul(FloatVec a, FloatVec b) {
        return vmulq_f32(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    add(FloatVec a, FloatVec b) {
        return vaddq_f32(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    sub(FloatVec a, FloatVec b) {
        return vsubq_f32(a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    fmadd(FloatVec a, FloatVec b, FloatVec c) {
        return vfmaq_f32(c, a, b);
    }

    static inline __attribute__((always_inline)) FloatVec
    div(FloatVec a, FloatVec b) {
        return vdivq_f32(a, b);
    }

    static inline __attribute__((always_inline)) void
    store(float* p, FloatVec v) {
        vst1q_f32(p, v);
    }

    static inline __attribute__((always_inline)) float
    reduce_add(FloatVec v) {
        return vaddvq_f32(v);
    }
};

}  // namespace vsag::simd
