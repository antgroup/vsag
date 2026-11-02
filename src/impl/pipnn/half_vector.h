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

#include <cstdint>
#include <cstring>

#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "utils/half_input.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define VSAG_PIPNN_HALF_AVX2 1
#else
#define VSAG_PIPNN_HALF_AVX2 0
#endif

namespace vsag::pipnn_half {

enum class Format { FP16, BF16 };

// IEEE binary16/bfloat16 decode, including signed zero, subnormals, infinities and NaNs.
// Deliberately independent of generic::FP16ToFloat, which only rebiases exponents.
inline float
Decode(uint16_t value, Format format) {
    if (format == Format::FP16) {
        return half_input::FP16ToFloat(value);
    }
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

namespace detail {

// Select the format once per row, including on targets without x86 intrinsics.
template <Format format>
inline void
DecodeRowScalar(const uint16_t* source, float* destination, uint64_t dim) {
    for (uint64_t k = 0; k < dim; ++k) {
        destination[k] = Decode(source[k], format);
    }
}

#if VSAG_PIPNN_HALF_AVX2
__attribute__((target("avx2,f16c"))) inline void
DecodeRowFP16(const uint16_t* source, float* destination, uint64_t dim) {
    uint64_t k = 0;
    for (; dim - k >= 8; k += 8) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + k));
        const __m128i exponent = _mm_set1_epi16(0x7c00);
        const __m128i nonfinite = _mm_cmpeq_epi16(_mm_and_si128(raw, exponent), exponent);
        // F16C quiets signaling NaNs. Do not convert exceptional lanes: retain the
        // scalar decoder's exact sign/payload bits, without raising invalid for sNaNs.
        const __m128i finite = _mm_andnot_si128(nonfinite, raw);
        _mm256_storeu_ps(destination + k, _mm256_cvtph_ps(finite));
        if (_mm_movemask_epi8(nonfinite) != 0) {
            for (uint64_t lane = 0; lane < 8; ++lane) {
                if ((source[k + lane] & 0x7c00U) == 0x7c00U) {
                    destination[k + lane] = Decode(source[k + lane], Format::FP16);
                }
            }
        }
    }
    DecodeRowScalar<Format::FP16>(source + k, destination + k, dim - k);
}

__attribute__((target("avx2"))) inline void
DecodeRowBF16(const uint16_t* source, float* destination, uint64_t dim) {
    uint64_t k = 0;
    for (; dim - k >= 8; k += 8) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + k));
        const __m256i bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
        _mm256_storeu_ps(destination + k, _mm256_castsi256_ps(bits));
    }
    DecodeRowScalar<Format::BF16>(source + k, destination + k, dim - k);
}

// Four-lane accumulation matches the baseline FP32 preparation reduction on
// baseline x86 builds. Widen only live registers; no FP32 input arrays.
template <bool square, Format format>
__attribute__((target("avx2,f16c,no-fma"))) inline float
PreparationDot(const uint16_t* a, const float* b, uint64_t dim) {
    __m128 sum = _mm_setzero_ps();
    uint64_t k = 0;
    for (; dim - k >= 4; k += 4) {
        uint64_t bits;
        std::memcpy(&bits, a + k, sizeof(bits));
        const __m128i raw = _mm_cvtsi64_si128(static_cast<int64_t>(bits));
        __m128 av;
        if constexpr (format == Format::FP16) {
            av = _mm_cvtph_ps(raw);
        } else {
            av = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(raw), 16));
        }
        const __m128 bv = square ? av : _mm_loadu_ps(b + k);
        sum = _mm_add_ps(sum, _mm_mul_ps(av, bv));
    }
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    float result = _mm_cvtss_f32(sum);
    for (; k < dim; ++k) {
        const float av = Decode(a[k], format);
        result += av * (square ? av : b[k]);
    }
    return result;
}
#endif

// Keep contraction control local to this fallback, never to the including TU.
// GCC function options are retained even when the caller uses fast-math.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
inline float
PreparationScalar(const uint16_t* a, const float* b, uint64_t dim, Format format) {
    // Preserve four independent lanes and separate multiply/add on every platform.
    const auto product = [a, b, format](uint64_t k) {
        const float av = Decode(a[k], format);
#if defined(__GNUC__) && !defined(__clang__)
        return av * (b == nullptr ? av : b[k]);
#else
        // Clang fast-math can contract across lexical contract(off) pragmas.
        // Retain the portable barrier until equivalent codegen is proven.
        volatile float value = av * (b == nullptr ? av : b[k]);
        return value;
#endif
    };
    float s0 = 0.0F;
    float s1 = 0.0F;
    float s2 = 0.0F;
    float s3 = 0.0F;
    uint64_t k = 0;
    for (; dim - k >= 4; k += 4) {
        s0 += product(k);
        s1 += product(k + 1);
        s2 += product(k + 2);
        s3 += product(k + 3);
    }
    float result = (s0 + s2) + (s1 + s3);
    for (; k < dim; ++k) {
        result += product(k);
    }
    return result;
}

}  // namespace detail

// Decode a nonoverlapping row into dim writable floats. Pointers need only their
// element alignment, not SIMD alignment; dim == 0 permits null pointers. No
// padding is read or written, and all output bits match Decode, including sNaNs.
inline void
DecodeRow(const uint16_t* source, float* destination, uint64_t dim, Format format) {
    if (dim == 0) {
        return;
    }
#if VSAG_PIPNN_HALF_AVX2
    static const bool avx2 = __builtin_cpu_supports("avx2");
    static const bool f16c = __builtin_cpu_supports("f16c");
    if (avx2) {
        if (format == Format::BF16) {
            detail::DecodeRowBF16(source, destination, dim);
            return;
        }
        if (f16c) {
            detail::DecodeRowFP16(source, destination, dim);
            return;
        }
    }
#endif
    if (format == Format::FP16) {
        detail::DecodeRowScalar<Format::FP16>(source, destination, dim);
    } else {
        detail::DecodeRowScalar<Format::BF16>(source, destination, dim);
    }
}

// Mixed half/FP32 projection and squared norm for preparation. This avoids
// scalar decode in every arithmetic operation while retaining FP32 accumulation.
inline float
Preparation(const uint16_t* a, const float* b, uint64_t dim, Format format) {
#if VSAG_PIPNN_HALF_AVX2
    static const bool supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
    if (supported) {
        if (b == nullptr) {
            return format == Format::FP16 ? detail::PreparationDot<true, Format::FP16>(a, b, dim)
                                          : detail::PreparationDot<true, Format::BF16>(a, b, dim);
        }
        return format == Format::FP16 ? detail::PreparationDot<false, Format::FP16>(a, b, dim)
                                      : detail::PreparationDot<false, Format::BF16>(a, b, dim);
    }
#endif
    return detail::PreparationScalar(a, b, dim, format);
}

// FP32 accumulation throughout. Vector reduction/FMA can differ in rounding
// from scalar summation. No half-precision accumulating instructions are used.
inline float
Dot(const uint16_t* a, const uint16_t* b, uint64_t dim, Format format) {
    if (dim == 0) {
        return 0.0F;
    }
    const auto* a_bytes = reinterpret_cast<const uint8_t*>(a);
    const auto* b_bytes = reinterpret_cast<const uint8_t*>(b);
    // Dispatched kernels use IEEE-correct half traits, also for scalar tails.
    // They do not use generic::FP16ToFloat.
    return format == Format::FP16 ? FP16ComputeIP(a_bytes, b_bytes, dim)
                                  : BF16ComputeIP(a_bytes, b_bytes, dim);
}

}  // namespace vsag::pipnn_half

#undef VSAG_PIPNN_HALF_AVX2
