
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

    static inline __attribute__((always_inline)) FloatVec
    set1(float v) {
        return _mm512_set1_ps(v);
    }
};

// --- Int8 traits for AVX512 ---
struct AVX512_Int8_Tag {};
template <>
struct Int8Traits<AVX512_Int8_Tag> {
    using Int8HalfVec = __m256i;
    using Int16Vec = __m512i;
    using Int32Vec = __m512i;
    static constexpr int BatchSize = 32;

    static inline __attribute__((always_inline)) Int8HalfVec
    load_i8(const int8_t* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static inline __attribute__((always_inline)) Int16Vec
    cvt_i8_to_i16(Int8HalfVec v) {
        return _mm512_cvtepi8_epi16(v);
    }
    static inline __attribute__((always_inline)) Int32Vec
    madd_i16(Int16Vec a, Int16Vec b) {
        return _mm512_madd_epi16(a, b);
    }
    static inline __attribute__((always_inline)) Int16Vec
    sub_i16(Int16Vec a, Int16Vec b) {
        return _mm512_sub_epi16(a, b);
    }
    static inline __attribute__((always_inline)) Int32Vec
    add_i32(Int32Vec a, Int32Vec b) {
        return _mm512_add_epi32(a, b);
    }
    static inline __attribute__((always_inline)) Int32Vec
    zero_i32() {
        return _mm512_setzero_si512();
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_i32(Int32Vec v) {
        return _mm512_reduce_add_epi32(v);
    }
};

// --- Bit operation traits for AVX512 ---
struct AVX512_Bit_Tag {};
template <>
struct BitTraits<AVX512_Bit_Tag> {
    using IntVec = __m512i;
    static constexpr int ByteWidth = 64;

    static inline __attribute__((always_inline)) IntVec
    load(const uint8_t* p) {
        return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p));
    }
    static inline __attribute__((always_inline)) void
    store(uint8_t* p, IntVec v) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(p), v);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_and(IntVec a, IntVec b) {
        return _mm512_and_si512(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_or(IntVec a, IntVec b) {
        return _mm512_or_si512(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_xor(IntVec a, IntVec b) {
        return _mm512_xor_si512(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_not(IntVec a) {
        return _mm512_xor_si512(a, _mm512_set1_epi8(static_cast<char>(0xFF)));
    }
};

// --- SQ8 traits for AVX512 ---
struct AVX512_SQ8_Tag {};
template <>
struct SQ8Traits<AVX512_SQ8_Tag> : SimdTraits<AVX512_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_u8_as_float(const uint8_t* p) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v));
    }
};

// --- SQ4Traits for AVX512 (Width=16, STEP=32 nibbles = 16 bytes per call) ---
struct AVX512_SQ4_Tag {};
template <>
struct SQ4Traits<AVX512_SQ4_Tag> : SimdTraits<AVX512_Tag> {
    static inline __attribute__((always_inline)) void
    load_nibbles_2x_as_float(const uint8_t* p, FloatVec& out0, FloatVec& out1) {
        __m128i code = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        __m128i mask = _mm_set1_epi8(0x0F);
        __m128i lo = _mm_and_si128(code, mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(code, 4), mask);
        __m128i il_lo = _mm_unpacklo_epi8(lo, hi);
        __m128i il_hi = _mm_unpackhi_epi8(lo, hi);
        __m512i e0 = _mm512_cvtepu8_epi32(il_lo);
        __m512i e1 = _mm512_cvtepu8_epi32(il_hi);
        const __m512 inv15 = _mm512_set1_ps(1.0f / 15.0f);
        out0 = _mm512_mul_ps(_mm512_cvtepi32_ps(e0), inv15);
        out1 = _mm512_mul_ps(_mm512_cvtepi32_ps(e1), inv15);
    }
};

// --- BF16Traits for AVX512 ---
struct AVX512_BF16_Tag {};
template <>
struct BF16Traits<AVX512_BF16_Tag> : SimdTraits<AVX512_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        __m256i bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        __m512i bf32 = _mm512_cvtepu16_epi32(bf16);
        return _mm512_castsi512_ps(_mm512_slli_epi32(bf32, 16));
    }
};

// --- FP16Traits for AVX512 ---
struct AVX512_FP16_Tag {};
template <>
struct FP16Traits<AVX512_FP16_Tag> : SimdTraits<AVX512_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        __m256i fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        return _mm512_cvtph_ps(fp16);
    }
};

// --- UniformCodeTraits for AVX512 (64-byte integer vectors) ---
struct AVX512_Uniform_Tag {};
template <>
struct UniformCodeTraits<AVX512_Uniform_Tag> {
    using IntVec = __m512i;
    static constexpr int ByteWidth = 64;

    static inline __attribute__((always_inline)) IntVec
    loadu(const uint8_t* p) {
        return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p));
    }
    static inline __attribute__((always_inline)) IntVec
    zero() {
        return _mm512_setzero_si512();
    }
    static inline __attribute__((always_inline)) IntVec
    set1_epi8(int8_t v) {
        return _mm512_set1_epi8(v);
    }
    static inline __attribute__((always_inline)) IntVec
    set1_epi16(int16_t v) {
        return _mm512_set1_epi16(v);
    }
    static inline __attribute__((always_inline)) IntVec
    and_si(IntVec a, IntVec b) {
        return _mm512_and_si512(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    srli_epi16(IntVec a, int imm) {
        return _mm512_srli_epi16(a, imm);
    }
    static inline __attribute__((always_inline)) IntVec
    maddubs_epi16(IntVec a, IntVec b) {
        return _mm512_maddubs_epi16(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    madd_epi16(IntVec a, IntVec b) {
        return _mm512_madd_epi16(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    add_epi16(IntVec a, IntVec b) {
        return _mm512_add_epi16(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    add_epi32(IntVec a, IntVec b) {
        return _mm512_add_epi32(a, b);
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_epi16(IntVec v) {
        auto lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v));
        auto hi = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(v, 1));
        return _mm512_reduce_add_epi32(lo) + _mm512_reduce_add_epi32(hi);
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_epi32(IntVec v) {
        return _mm512_reduce_add_epi32(v);
    }
};

// --- RaBitQTraits for AVX512 (16-wide, uses mask registers) ---
struct AVX512_RaBitQ_Tag {};
template <>
struct RaBitQTraits<AVX512_RaBitQ_Tag> : SimdTraits<AVX512_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    bits_to_signed(const uint8_t* bit_ptr, FloatVec pos, FloatVec neg) {
        auto mask = static_cast<__mmask16>(static_cast<uint16_t>(bit_ptr[0]) |
                                           (static_cast<uint16_t>(bit_ptr[1]) << 8));
        return _mm512_mask_blend_ps(mask, neg, pos);
    }

    static inline __attribute__((always_inline)) FloatVec
    bits_select(const uint8_t* bit_ptr, FloatVec weight) {
        auto mask = static_cast<__mmask16>(static_cast<uint16_t>(bit_ptr[0]) |
                                           (static_cast<uint16_t>(bit_ptr[1]) << 8));
        return _mm512_mask_add_ps(_mm512_setzero_ps(), mask, _mm512_setzero_ps(), weight);
    }
};

}  // namespace vsag::simd
