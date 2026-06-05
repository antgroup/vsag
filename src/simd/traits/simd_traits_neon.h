
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

    static inline __attribute__((always_inline)) FloatVec
    set1(float v) {
        return vdupq_n_f32(v);
    }
};

// --- Int8 traits for NEON ---
struct NEON_Int8_Tag {};
template <>
struct Int8Traits<NEON_Int8_Tag> {
    using Int8HalfVec = int8x8_t;
    using Int16Vec = int16x8_t;
    using Int32Vec = int32x4_t;
    static constexpr int BatchSize = 8;

    static inline __attribute__((always_inline)) Int8HalfVec
    load_i8(const int8_t* p) {
        return vld1_s8(p);
    }
    static inline __attribute__((always_inline)) Int16Vec
    cvt_i8_to_i16(Int8HalfVec v) {
        return vmovl_s8(v);
    }
    // Emulate x86 madd_epi16: pairwise (a[2i]*b[2i] + a[2i+1]*b[2i+1])
    static inline __attribute__((always_inline)) Int32Vec
    madd_i16(Int16Vec a, Int16Vec b) {
        int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        int32x4_t hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
        return vaddq_s32(vpaddq_s32(lo, hi), vdupq_n_s32(0));
    }
    static inline __attribute__((always_inline)) Int16Vec
    sub_i16(Int16Vec a, Int16Vec b) {
        return vsubq_s16(a, b);
    }
    static inline __attribute__((always_inline)) Int32Vec
    add_i32(Int32Vec a, Int32Vec b) {
        return vaddq_s32(a, b);
    }
    static inline __attribute__((always_inline)) Int32Vec
    zero_i32() {
        return vdupq_n_s32(0);
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_i32(Int32Vec v) {
        return vaddvq_s32(v);
    }
};

// --- Bit operation traits for NEON ---
struct NEON_Bit_Tag {};
template <>
struct BitTraits<NEON_Bit_Tag> {
    using IntVec = uint8x16_t;
    static constexpr int ByteWidth = 16;

    static inline __attribute__((always_inline)) IntVec
    load(const uint8_t* p) {
        return vld1q_u8(p);
    }
    static inline __attribute__((always_inline)) void
    store(uint8_t* p, IntVec v) {
        vst1q_u8(p, v);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_and(IntVec a, IntVec b) {
        return vandq_u8(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_or(IntVec a, IntVec b) {
        return vorrq_u8(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_xor(IntVec a, IntVec b) {
        return veorq_u8(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_not(IntVec a) {
        return vmvnq_u8(a);
    }
};

// --- SQ8 traits for NEON ---
struct NEON_SQ8_Tag {};
template <>
struct SQ8Traits<NEON_SQ8_Tag> : SimdTraits<NEON_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_u8_as_float(const uint8_t* p) {
        uint32x4_t v32 = {p[0], p[1], p[2], p[3]};
        return vcvtq_f32_u32(v32);
    }
};

// --- SQ4Traits for NEON (Width=4, STEP=8 nibbles = 4 bytes per call) ---
struct NEON_SQ4_Tag {};
template <>
struct SQ4Traits<NEON_SQ4_Tag> : SimdTraits<NEON_Tag> {
    static inline __attribute__((always_inline)) void
    load_nibbles_2x_as_float(const uint8_t* p, FloatVec& out0, FloatVec& out1) {
        constexpr float inv15 = 1.0f / 15.0f;
        float32x4_t v_inv15 = vdupq_n_f32(inv15);
        float32x4_t lo = {
            static_cast<float>(p[0] & 0x0f),
            static_cast<float>(p[0] >> 4),
            static_cast<float>(p[1] & 0x0f),
            static_cast<float>(p[1] >> 4),
        };
        float32x4_t hi = {
            static_cast<float>(p[2] & 0x0f),
            static_cast<float>(p[2] >> 4),
            static_cast<float>(p[3] & 0x0f),
            static_cast<float>(p[3] >> 4),
        };
        out0 = vmulq_f32(lo, v_inv15);
        out1 = vmulq_f32(hi, v_inv15);
    }
};

// --- BF16Traits for NEON ---
struct NEON_BF16_Tag {};
template <>
struct BF16Traits<NEON_BF16_Tag> : SimdTraits<NEON_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        uint16x4_t bf16 = vld1_u16(p);
        uint32x4_t bf32 = vshll_n_u16(bf16, 16);
        return vreinterpretq_f32_u32(bf32);
    }
};

// --- FP16Traits for NEON ---
struct NEON_FP16_Tag {};
template <>
struct FP16Traits<NEON_FP16_Tag> : SimdTraits<NEON_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        float16x4_t fp16 = vld1_f16(reinterpret_cast<const __fp16*>(p));
        return vcvt_f32_f16(fp16);
    }
};

}  // namespace vsag::simd
