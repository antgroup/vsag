
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

    static inline __attribute__((always_inline)) FloatVec
    set1(float v) {
        return _mm_set1_ps(v);
    }
};

// --- Int8 traits for SSE ---
struct SSE_Int8_Tag {};
template <>
struct Int8Traits<SSE_Int8_Tag> {
    using Int8HalfVec = __m128i;  // only lower 64 bits used
    using Int16Vec = __m128i;
    using Int32Vec = __m128i;
    static constexpr int BatchSize = 8;

    static inline __attribute__((always_inline)) Int8HalfVec
    load_i8(const int8_t* p) {
        return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    }
    static inline __attribute__((always_inline)) Int16Vec
    cvt_i8_to_i16(Int8HalfVec v) {
        return _mm_cvtepi8_epi16(v);
    }
    static inline __attribute__((always_inline)) Int32Vec
    madd_i16(Int16Vec a, Int16Vec b) {
        return _mm_madd_epi16(a, b);
    }
    static inline __attribute__((always_inline)) Int16Vec
    sub_i16(Int16Vec a, Int16Vec b) {
        return _mm_sub_epi16(a, b);
    }
    static inline __attribute__((always_inline)) Int32Vec
    add_i32(Int32Vec a, Int32Vec b) {
        return _mm_add_epi32(a, b);
    }
    static inline __attribute__((always_inline)) Int32Vec
    zero_i32() {
        return _mm_setzero_si128();
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_i32(Int32Vec v) {
        alignas(16) int32_t tmp[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(tmp), v);
        return tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
};

// --- Bit operation traits for SSE ---
struct SSE_Bit_Tag {};
template <>
struct BitTraits<SSE_Bit_Tag> {
    using IntVec = __m128i;
    static constexpr int ByteWidth = 16;

    static inline __attribute__((always_inline)) IntVec
    load(const uint8_t* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    static inline __attribute__((always_inline)) void
    store(uint8_t* p, IntVec v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_and(IntVec a, IntVec b) {
        return _mm_and_si128(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_or(IntVec a, IntVec b) {
        return _mm_or_si128(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_xor(IntVec a, IntVec b) {
        return _mm_xor_si128(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_not(IntVec a) {
        return _mm_xor_si128(a, _mm_set1_epi8(static_cast<char>(0xFF)));
    }
};

// --- SQ8 traits for SSE ---
struct SSE_SQ8_Tag {};
template <>
struct SQ8Traits<SSE_SQ8_Tag> : SimdTraits<SSE_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_u8_as_float(const uint8_t* p) {
        __m128i v = _mm_set_epi8(0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 static_cast<char>(p[3]),
                                 static_cast<char>(p[2]),
                                 static_cast<char>(p[1]),
                                 static_cast<char>(p[0]));
        return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(v));
    }
};

// --- SQ4Traits for SSE (Width=4, STEP=8 nibbles = 4 bytes per call) ---
struct SSE_SQ4_Tag {};
template <>
struct SQ4Traits<SSE_SQ4_Tag> : SimdTraits<SSE_Tag> {
    static inline __attribute__((always_inline)) void
    load_nibbles_2x_as_float(const uint8_t* p, FloatVec& out0, FloatVec& out1) {
        __m128i code_vec = _mm_set_epi8(0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        static_cast<char>(p[3]),
                                        static_cast<char>(p[2]),
                                        static_cast<char>(p[1]),
                                        static_cast<char>(p[0]));
        __m128i lo = _mm_and_si128(code_vec, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(code_vec, 4), _mm_set1_epi8(0x0F));
        __m128i interleaved = _mm_unpacklo_epi8(lo, hi);
        __m128i v0 = _mm_cvtepu8_epi32(interleaved);
        __m128i v1 = _mm_cvtepu8_epi32(_mm_srli_si128(interleaved, 4));
        const __m128 inv15 = _mm_set1_ps(1.0f / 15.0f);
        out0 = _mm_mul_ps(_mm_cvtepi32_ps(v0), inv15);
        out1 = _mm_mul_ps(_mm_cvtepi32_ps(v1), inv15);
    }
};

// --- BF16Traits for SSE ---
struct SSE_BF16_Tag {};
template <>
struct BF16Traits<SSE_BF16_Tag> : SimdTraits<SSE_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        __m128i v = _mm_set_epi16(static_cast<short>(p[3]),
                                  0,
                                  static_cast<short>(p[2]),
                                  0,
                                  static_cast<short>(p[1]),
                                  0,
                                  static_cast<short>(p[0]),
                                  0);
        return _mm_castsi128_ps(v);
    }
};

// --- UniformCodeTraits for SSE (16-byte integer vectors) ---
struct SSE_Uniform_Tag {};
template <>
struct UniformCodeTraits<SSE_Uniform_Tag> {
    using IntVec = __m128i;
    static constexpr int ByteWidth = 16;

    static inline __attribute__((always_inline)) IntVec
    loadu(const uint8_t* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    static inline __attribute__((always_inline)) IntVec
    zero() {
        return _mm_setzero_si128();
    }
    static inline __attribute__((always_inline)) IntVec
    set1_epi8(int8_t v) {
        return _mm_set1_epi8(v);
    }
    static inline __attribute__((always_inline)) IntVec
    set1_epi16(int16_t v) {
        return _mm_set1_epi16(v);
    }
    static inline __attribute__((always_inline)) IntVec
    and_si(IntVec a, IntVec b) {
        return _mm_and_si128(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    srli_epi16(IntVec a, int imm) {
        return _mm_srli_epi16(a, imm);
    }
    static inline __attribute__((always_inline)) IntVec
    maddubs_epi16(IntVec a, IntVec b) {
        return _mm_maddubs_epi16(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    madd_epi16(IntVec a, IntVec b) {
        return _mm_madd_epi16(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    add_epi16(IntVec a, IntVec b) {
        return _mm_add_epi16(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    add_epi32(IntVec a, IntVec b) {
        return _mm_add_epi32(a, b);
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_epi16(IntVec v) {
        alignas(16) int16_t tmp[8];
        _mm_store_si128(reinterpret_cast<__m128i*>(tmp), v);
        int32_t sum = 0;
        for (int i = 0; i < 8; ++i) sum += tmp[i];
        return sum;
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_epi32(IntVec v) {
        alignas(16) int32_t tmp[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(tmp), v);
        return tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
};

}  // namespace vsag::simd
