
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

#include <cstring>

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

    static inline __attribute__((always_inline)) FloatVec
    set1(float v) {
        return _mm256_set1_ps(v);
    }
};

// --- Bit operation traits for AVX (uses float ops since AVX has no 256-bit integer) ---
struct AVX_Bit_Tag {};
template <>
struct BitTraits<AVX_Bit_Tag> {
    using IntVec = __m256;  // reuse float vector for bitwise ops
    static constexpr int ByteWidth = 32;

    static inline __attribute__((always_inline)) IntVec
    load(const uint8_t* p) {
        return _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
    }
    static inline __attribute__((always_inline)) void
    store(uint8_t* p, IntVec v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), _mm256_castps_si256(v));
    }
    static inline __attribute__((always_inline)) IntVec
    bit_and(IntVec a, IntVec b) {
        return _mm256_and_ps(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_or(IntVec a, IntVec b) {
        return _mm256_or_ps(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_xor(IntVec a, IntVec b) {
        return _mm256_xor_ps(a, b);
    }
    static inline __attribute__((always_inline)) IntVec
    bit_not(IntVec a) {
        return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }
};

// --- SQ8 traits for AVX ---
struct AVX_SQ8_Tag {};
template <>
struct SQ8Traits<AVX_SQ8_Tag> : SimdTraits<AVX_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_u8_as_float(const uint8_t* p) {
        int32_t lo_val, hi_val;
        std::memcpy(&lo_val, p, 4);
        std::memcpy(&hi_val, p + 4, 4);
        __m128i lo = _mm_cvtepu8_epi32(_mm_set_epi32(0, 0, 0, lo_val));
        __m128i hi = _mm_cvtepu8_epi32(_mm_set_epi32(0, 0, 0, hi_val));
        return _mm256_cvtepi32_ps(_mm256_set_m128i(hi, lo));
    }
};

// --- SQ4Traits for AVX (Width=8, STEP=16 nibbles = 8 bytes per call) ---
struct AVX_SQ4_Tag {};
template <>
struct SQ4Traits<AVX_SQ4_Tag> : SimdTraits<AVX_Tag> {
    static inline __attribute__((always_inline)) void
    load_nibbles_2x_as_float(const uint8_t* p, FloatVec& out0, FloatVec& out1) {
        __m128i code0 = _mm_set_epi8(0,
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
        __m128i code1 = _mm_set_epi8(0,
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
                                     static_cast<char>(p[7]),
                                     static_cast<char>(p[6]),
                                     static_cast<char>(p[5]),
                                     static_cast<char>(p[4]));
        const __m128i mask = _mm_set1_epi8(0x0F);
        __m128i lo0 = _mm_and_si128(code0, mask);
        __m128i hi0 = _mm_and_si128(_mm_srli_epi16(code0, 4), mask);
        __m128i lo1 = _mm_and_si128(code1, mask);
        __m128i hi1 = _mm_and_si128(_mm_srli_epi16(code1, 4), mask);
        __m128i il0 = _mm_unpacklo_epi8(lo0, hi0);
        __m128i il1 = _mm_unpacklo_epi8(lo1, hi1);
        __m128 v00 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(il0));
        __m128 v01 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(il0, 4)));
        __m128 v10 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(il1));
        __m128 v11 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(il1, 4)));
        const __m256 inv15 = _mm256_set1_ps(1.0f / 15.0f);
        out0 = _mm256_mul_ps(_mm256_set_m128(v01, v00), inv15);
        out1 = _mm256_mul_ps(_mm256_set_m128(v11, v10), inv15);
    }
};

// --- BF16Traits for AVX ---
struct AVX_BF16_Tag {};
template <>
struct BF16Traits<AVX_BF16_Tag> : SimdTraits<AVX_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        __m256i v = _mm256_set_epi16(static_cast<short>(p[7]),
                                     0,
                                     static_cast<short>(p[6]),
                                     0,
                                     static_cast<short>(p[5]),
                                     0,
                                     static_cast<short>(p[4]),
                                     0,
                                     static_cast<short>(p[3]),
                                     0,
                                     static_cast<short>(p[2]),
                                     0,
                                     static_cast<short>(p[1]),
                                     0,
                                     static_cast<short>(p[0]),
                                     0);
        return _mm256_castsi256_ps(v);
    }
};

// --- FP16Traits for AVX (uses F16C: _mm256_cvtph_ps) ---
struct AVX_FP16_Tag {};
template <>
struct FP16Traits<AVX_FP16_Tag> : SimdTraits<AVX_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        __m128i fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        return _mm256_cvtph_ps(fp16);
    }
};

// --- RaBitQTraits for AVX (8-wide, no AVX2 integer ops) ---
struct AVX_RaBitQ_Tag {};
template <>
struct RaBitQTraits<AVX_RaBitQ_Tag> : SimdTraits<AVX_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    bits_to_signed(const uint8_t* bit_ptr, FloatVec pos, FloatVec neg) {
        uint8_t byte = bit_ptr[0];
        float p = _mm_cvtss_f32(_mm256_castps256_ps128(pos));
        float n = _mm_cvtss_f32(_mm256_castps256_ps128(neg));
        return _mm256_set_ps(((byte >> 7) & 1) ? p : n,
                             ((byte >> 6) & 1) ? p : n,
                             ((byte >> 5) & 1) ? p : n,
                             ((byte >> 4) & 1) ? p : n,
                             ((byte >> 3) & 1) ? p : n,
                             ((byte >> 2) & 1) ? p : n,
                             ((byte >> 1) & 1) ? p : n,
                             ((byte >> 0) & 1) ? p : n);
    }

    static inline __attribute__((always_inline)) FloatVec
    bits_select(const uint8_t* bit_ptr, FloatVec weight) {
        uint8_t byte = bit_ptr[0];
        float w = _mm_cvtss_f32(_mm256_castps256_ps128(weight));
        return _mm256_set_ps(((byte >> 7) & 1) ? w : 0.0f,
                             ((byte >> 6) & 1) ? w : 0.0f,
                             ((byte >> 5) & 1) ? w : 0.0f,
                             ((byte >> 4) & 1) ? w : 0.0f,
                             ((byte >> 3) & 1) ? w : 0.0f,
                             ((byte >> 2) & 1) ? w : 0.0f,
                             ((byte >> 1) & 1) ? w : 0.0f,
                             ((byte >> 0) & 1) ? w : 0.0f);
    }
};

}  // namespace vsag::simd
