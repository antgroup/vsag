
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

// Scalar "pseudo-SIMD" traits for the generic backend.
//
// IMPORTANT (compile-strategy invariant):
//   - This header MUST only be included by ISA translation units whose
//     COMPILE_FLAGS are compatible with the intrinsics it exposes.
//   - The generic backend uses no intrinsics, so this header is safe in
//     any TU; the convention nevertheless holds for the AVX/NEON/SVE
//     traits headers that will be added in later phases.
//   - Cross-ISA inclusion of a higher-tier traits header from a lower-tier
//     TU (e.g. avx512 traits in sse.cpp) will trigger
//     "target specific option mismatch" errors. Do not break this rule.

namespace vsag::simd {

struct Generic_Tag {};

// Primary template; specializations live in sibling headers.
template <typename ISA>
struct SimdTraits;

template <typename ISA>
struct Int8Traits;

template <>
struct SimdTraits<Generic_Tag> {
    using FloatVec = float;
    static constexpr int Width = 1;

    static inline __attribute__((always_inline)) FloatVec
    zero() {
        return 0.0f;
    }

    static inline __attribute__((always_inline)) FloatVec
    load(const float* p) {
        return *p;
    }

    static inline __attribute__((always_inline)) FloatVec
    mul(FloatVec a, FloatVec b) {
        return a * b;
    }

    static inline __attribute__((always_inline)) FloatVec
    add(FloatVec a, FloatVec b) {
        return a + b;
    }

    static inline __attribute__((always_inline)) FloatVec
    sub(FloatVec a, FloatVec b) {
        return a - b;
    }

    static inline __attribute__((always_inline)) FloatVec
    fmadd(FloatVec a, FloatVec b, FloatVec c) {
        return a * b + c;
    }

    static inline __attribute__((always_inline)) FloatVec
    div(FloatVec a, FloatVec b) {
        return a / b;
    }

    static inline __attribute__((always_inline)) void
    store(float* p, FloatVec v) {
        *p = v;
    }

    static inline __attribute__((always_inline)) float
    reduce_add(FloatVec v) {
        return v;
    }

    static inline __attribute__((always_inline)) FloatVec
    set1(float v) {
        return v;
    }
};

// --- Int8 traits for generic (scalar) ---
struct Generic_Int8_Tag {};
template <>
struct Int8Traits<Generic_Int8_Tag> {
    using Int8HalfVec = int8_t;
    using Int16Vec = int16_t;
    using Int32Vec = int32_t;
    static constexpr int BatchSize = 1;

    static inline __attribute__((always_inline)) Int8HalfVec
    load_i8(const int8_t* p) {
        return *p;
    }
    static inline __attribute__((always_inline)) Int16Vec
    cvt_i8_to_i16(Int8HalfVec v) {
        return static_cast<int16_t>(v);
    }
    static inline __attribute__((always_inline)) Int32Vec
    madd_i16(Int16Vec a, Int16Vec b) {
        return static_cast<int32_t>(a) * b;
    }
    static inline __attribute__((always_inline)) Int16Vec
    sub_i16(Int16Vec a, Int16Vec b) {
        return a - b;
    }
    static inline __attribute__((always_inline)) Int32Vec
    add_i32(Int32Vec a, Int32Vec b) {
        return a + b;
    }
    static inline __attribute__((always_inline)) Int32Vec
    zero_i32() {
        return 0;
    }
    static inline __attribute__((always_inline)) int32_t
    reduce_add_i32(Int32Vec v) {
        return v;
    }
};

// --- BitTraits primary template ---
template <typename Tag>
struct BitTraits;

// Generic scalar bit operations (ByteWidth = 1, process byte by byte)
struct Generic_Bit_Tag {};
template <>
struct BitTraits<Generic_Bit_Tag> {
    using IntVec = uint8_t;
    static constexpr int ByteWidth = 1;

    static inline __attribute__((always_inline)) IntVec
    load(const uint8_t* p) {
        return *p;
    }
    static inline __attribute__((always_inline)) void
    store(uint8_t* p, IntVec v) {
        *p = v;
    }
    static inline __attribute__((always_inline)) IntVec
    bit_and(IntVec a, IntVec b) {
        return a & b;
    }
    static inline __attribute__((always_inline)) IntVec
    bit_or(IntVec a, IntVec b) {
        return a | b;
    }
    static inline __attribute__((always_inline)) IntVec
    bit_xor(IntVec a, IntVec b) {
        return a ^ b;
    }
    static inline __attribute__((always_inline)) IntVec
    bit_not(IntVec a) {
        return ~a;
    }
};

// --- SQ8Traits primary template ---
template <typename Tag>
struct SQ8Traits;

// Generic SQ8 traits (scalar, Width=1)
struct Generic_SQ8_Tag {};
template <>
struct SQ8Traits<Generic_SQ8_Tag> : SimdTraits<Generic_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_u8_as_float(const uint8_t* p) {
        return static_cast<float>(p[0]);
    }
};

// --- SQ4Traits primary template ---
template <typename Tag>
struct SQ4Traits;

// Generic SQ4 (scalar, Width=1, STEP=2 nibbles per iter)
struct Generic_SQ4_Tag {};
template <>
struct SQ4Traits<Generic_SQ4_Tag> : SimdTraits<Generic_Tag> {
    static inline __attribute__((always_inline)) void
    load_nibbles_2x_as_float(const uint8_t* p, FloatVec& out0, FloatVec& out1) {
        constexpr float inv15 = 1.0f / 15.0f;
        out0 = (p[0] & 0x0f) * inv15;
        out1 = (p[0] >> 4) * inv15;
    }
};

// --- BF16Traits primary template ---
template <typename Tag>
struct BF16Traits;

struct Generic_BF16_Tag {};
template <>
struct BF16Traits<Generic_BF16_Tag> : SimdTraits<Generic_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        uint32_t bits = static_cast<uint32_t>(*p) << 16;
        float f;
        __builtin_memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

// --- RaBitQTraits primary template ---
template <typename Tag>
struct RaBitQTraits;

// --- UniformCodeTraits primary template ---
// Provides integer vector ops for SQ4/SQ8 uniform code IP computation.
template <typename Tag>
struct UniformCodeTraits;

// --- FP16Traits primary template ---
template <typename Tag>
struct FP16Traits;

struct Generic_FP16_Tag {};
template <>
struct FP16Traits<Generic_FP16_Tag> : SimdTraits<Generic_Tag> {
    static inline __attribute__((always_inline)) FloatVec
    load_half(const uint16_t* p) {
        uint32_t h = *p;
        uint32_t sign = (h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FFu;
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign;
            } else {
                exp = 1;
                while (!(mant & 0x400)) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x3FF;
                f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f = sign | 0x7F800000u | (mant << 13);
        } else {
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float result;
        __builtin_memcpy(&result, &f, sizeof(result));
        return result;
    }
};

}  // namespace vsag::simd
