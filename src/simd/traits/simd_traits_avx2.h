
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

// AVX2 traits (Width = 8, __m256, with FMA). MUST only be included from
// a TU compiled with -mavx2 -mfma. Inherits SimdTraits<AVX_Tag> and
// overrides fmadd to use the real FMA instruction.

#include "simd_traits_avx.h"

namespace vsag::simd {

struct AVX2_Tag {};

template <>
struct SimdTraits<AVX2_Tag> : SimdTraits<AVX_Tag> {
    // Override: AVX2 + FMA gives us a single fused mul-add.
    static inline __attribute__((always_inline)) FloatVec
    fmadd(FloatVec a, FloatVec b, FloatVec c) {
        return _mm256_fmadd_ps(a, b, c);
    }
};

// --- Int8 traits for AVX2 ---
struct AVX2_Int8_Tag {};
template <>
struct Int8Traits<AVX2_Int8_Tag> {
    using Int8HalfVec = __m128i;
    using Int16Vec = __m256i;
    using Int32Vec = __m256i;
    static constexpr int BatchSize = 16;

    static inline __attribute__((always_inline)) Int8HalfVec
    load_i8(const int8_t* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    static inline __attribute__((always_inline)) Int16Vec
    cvt_i8_to_i16(Int8HalfVec v) {
        return _mm256_cvtepi8_epi16(v);
    }
    static inline __attribute__((always_inline)) Int32Vec
    madd_i16(Int16Vec a, Int16Vec b) {
        return _mm256_madd_epi16(a, b);
    }
    static inline __attribute__((always_inline)) Int16Vec
    sub_i16(Int16Vec a, Int16Vec b) {
        return _mm256_sub_epi16(a, b);
    }
    static inline __attribute__((always_inline)) Int32Vec
    add_i32(Int32Vec a, Int32Vec b) {
        return _mm256_add_epi32(a, b);
    }
    static inline __attribute__((always_inline)) Int32Vec
    zero_i32() {
        return _mm256_setzero_si256();
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_i32(Int32Vec v) {
        alignas(32) int32_t tmp[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v);
        int32_t sum = 0;
        for (int i = 0; i < 8; ++i) sum += tmp[i];
        return sum;
    }
};

// --- Bit operation traits for AVX2 ---
struct AVX2_Bit_Tag {};
template <>
struct BitTraits<AVX2_Bit_Tag> {
    using IntVec = __m256i;
    static constexpr int ByteWidth = 32;

    static inline __attribute__((always_inline)) IntVec
    load(const uint8_t* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static inline __attribute__((always_inline)) void
    store(uint8_t* p, IntVec v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_and(IntVec a, IntVec b) {
        return _mm256_and_si256(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_or(IntVec a, IntVec b) {
        return _mm256_or_si256(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_xor(IntVec a, IntVec b) {
        return _mm256_xor_si256(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_not(IntVec a) {
        return _mm256_xor_si256(a, _mm256_set1_epi8(static_cast<char>(0xFF)));
    }
};

// --- SQ8 traits for AVX2 ---
struct AVX2_SQ8_Tag {};
template <>
struct SQ8Traits<AVX2_SQ8_Tag> : SimdTraits<AVX2_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_u8_as_float(const uint8_t* p) {
        __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
    }
};

// --- SQ4Traits for AVX2 (Width=8, STEP=16 nibbles = 8 bytes per call) ---
struct AVX2_SQ4_Tag {};
template <>
struct SQ4Traits<AVX2_SQ4_Tag> : SimdTraits<AVX2_Tag> {
    static inline __attribute__((always_inline)) void
    load_nibbles_2x_as_float(const uint8_t* p, FloatVec& out0, FloatVec& out1) {
        __m128i code = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        __m128i mask = _mm_set1_epi8(0x0F);
        __m128i lo = _mm_and_si128(code, mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(code, 4), mask);
        __m128i il = _mm_unpacklo_epi8(lo, hi);
        __m256i e0 = _mm256_cvtepu8_epi32(il);
        __m256i e1 = _mm256_cvtepu8_epi32(_mm_srli_si128(il, 8));
        const __m256 inv15 = _mm256_set1_ps(1.0f / 15.0f);
        out0 = _mm256_mul_ps(_mm256_cvtepi32_ps(e0), inv15);
        out1 = _mm256_mul_ps(_mm256_cvtepi32_ps(e1), inv15);
    }
};

// --- BF16Traits for AVX2 ---
struct AVX2_BF16_Tag {};
template <>
struct BF16Traits<AVX2_BF16_Tag> : SimdTraits<AVX2_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        __m128i bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        __m256i bf32 = _mm256_cvtepu16_epi32(bf16);
        return _mm256_castsi256_ps(_mm256_slli_epi32(bf32, 16));
    }
};

// --- FP16Traits for AVX2 ---
struct AVX2_FP16_Tag {};
template <>
struct FP16Traits<AVX2_FP16_Tag> : SimdTraits<AVX2_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        __m128i fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        return _mm256_cvtph_ps(fp16);
    }
};

// --- UniformCodeTraits for AVX2 (32-byte integer vectors) ---
struct AVX2_Uniform_Tag {};
template <>
struct UniformCodeTraits<AVX2_Uniform_Tag> {
    using IntVec = __m256i;
    static constexpr int ByteWidth = 32;

    static inline __attribute__((always_inline)) IntVec
    loadu(const uint8_t* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static inline __attribute__((always_inline)) IntVec
    zero() {
        return _mm256_setzero_si256();
    }
    static inline __attribute__((always_inline)) IntVec
    set1_epi8(int8_t v) {
        return _mm256_set1_epi8(v);
    }
    static inline __attribute__((always_inline)) IntVec
    set1_epi16(int16_t v) {
        return _mm256_set1_epi16(v);
    }
    static inline __attribute__((always_inline)) IntVec
    and_si(IntVec a, IntVec b) {
        return _mm256_and_si256(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    srli_epi16(IntVec a, int imm) {
        return _mm256_srli_epi16(a, imm);
    }
    static inline __attribute__((always_inline)) IntVec
    maddubs_epi16(IntVec a, IntVec b) {
        return _mm256_maddubs_epi16(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    madd_epi16(IntVec a, IntVec b) {
        return _mm256_madd_epi16(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    add_epi16(IntVec a, IntVec b) {
        return _mm256_add_epi16(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    add_epi32(IntVec a, IntVec b) {
        return _mm256_add_epi32(a, b);
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_epi16(IntVec v) {
        alignas(32) int16_t tmp[16];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v);
        int32_t sum = 0;
        for (int i = 0; i < 16; ++i) sum += tmp[i];
        return sum;
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_epi32(IntVec v) {
        alignas(32) int32_t tmp[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v);
        int32_t sum = 0;
        for (int i = 0; i < 8; ++i) sum += tmp[i];
        return sum;
    }
};

// --- RaBitQTraits for AVX2 (8-wide, uses AVX2 integer ops for bit decode) ---
struct AVX2_RaBitQ_Tag {};
template <>
struct RaBitQTraits<AVX2_RaBitQ_Tag> : SimdTraits<AVX2_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    bits_to_signed(const uint8_t* bit_ptr, FloatVec pos, FloatVec neg) {
        const __m256i bit_masks = _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
        const __m256i all_ones = _mm256_set1_epi32(-1);
        const __m256i zero_i = _mm256_setzero_si256();
        __m256i mask = _mm256_set1_epi32(static_cast<int>(bit_ptr[0]));
        mask = _mm256_and_si256(mask, bit_masks);
        mask = _mm256_cmpeq_epi32(mask, zero_i);
        mask = _mm256_andnot_si256(mask, all_ones);
        return _mm256_blendv_ps(neg, pos, _mm256_castsi256_ps(mask));
    }

    static inline __attribute__((always_inline)) FloatVec
    bits_select(const uint8_t* bit_ptr, FloatVec weight) {
        const __m256i bit_masks = _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
        const __m256i all_ones = _mm256_set1_epi32(-1);
        const __m256i zero_i = _mm256_setzero_si256();
        __m256i mask = _mm256_set1_epi32(static_cast<int>(bit_ptr[0]));
        mask = _mm256_and_si256(mask, bit_masks);
        mask = _mm256_cmpeq_epi32(mask, zero_i);
        mask = _mm256_andnot_si256(mask, all_ones);
        return _mm256_blendv_ps(_mm256_setzero_ps(), weight, _mm256_castsi256_ps(mask));
    }
};

}  // namespace vsag::simd
