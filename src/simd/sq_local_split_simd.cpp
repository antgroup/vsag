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

#include "sq_local_split_simd.h"

#include <cstring>

#include "simd_status.h"

#if defined(__x86_64__) or defined(_M_X64)
#include <immintrin.h>
#define VSAG_HAS_X86_INTRINSICS 1
#endif

namespace vsag {

namespace {

inline uint64_t
load_u64(const uint8_t* p) {
    uint64_t value = 0;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

inline uint32_t
load_bits(const uint8_t* p, uint64_t bit_offset, uint64_t bits) {
    uint32_t raw = 0;
    std::memcpy(&raw, p + (bit_offset >> 3), sizeof(uint16_t));
    return (raw >> (bit_offset & 7U)) & static_cast<uint32_t>((1ULL << bits) - 1ULL);
}

}  // namespace

namespace generic {

float
FloatPackedIP(
    const float* q, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y) {
    const uint64_t mask_hi = (1ULL << x) - 1ULL;
    const uint64_t mask_lo = (1ULL << y) - 1ULL;
    float sum = 0.0F;
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const uint64_t hv = load_u64(hi + (i * x) / 8);
        const uint64_t lv = load_u64(lo + (i * y) / 8);
        for (uint64_t k = 0; k < 8; ++k) {
            const auto h = static_cast<uint32_t>((hv >> (k * x)) & mask_hi);
            const auto l = static_cast<uint32_t>((lv >> (k * y)) & mask_lo);
            sum += q[i + k] * static_cast<float>((h << y) | l);
        }
    }
    for (; i < dim; ++i) {
        const auto h = load_bits(hi, i * x, x);
        const auto l = load_bits(lo, i * y, y);
        sum += q[i] * static_cast<float>((h << y) | l);
    }
    return sum;
}

float
BytePackedIP(
    const uint8_t* qb, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y) {
    const uint64_t mask_hi = (1ULL << x) - 1ULL;
    const uint64_t mask_lo = (1ULL << y) - 1ULL;
    float sum = 0.0F;
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const uint64_t hv = load_u64(hi + (i * x) / 8);
        const uint64_t lv = load_u64(lo + (i * y) / 8);
        uint64_t qv = 0;
        std::memcpy(&qv, qb + i, sizeof(qv));
        for (uint64_t k = 0; k < 8; ++k) {
            const auto h = static_cast<uint32_t>((hv >> (k * x)) & mask_hi);
            const auto l = static_cast<uint32_t>((lv >> (k * y)) & mask_lo);
            const auto qk = static_cast<uint32_t>((qv >> (k * 8)) & 0xFFULL);
            sum += static_cast<float>(qk) * static_cast<float>((h << y) | l);
        }
    }
    for (; i < dim; ++i) {
        const auto h = load_bits(hi, i * x, x);
        const auto l = load_bits(lo, i * y, y);
        sum += static_cast<float>(qb[i]) * static_cast<float>((h << y) | l);
    }
    return sum;
}

float
PackedPackedIP(const uint8_t* hi1,
               const uint8_t* lo1,
               const uint8_t* hi2,
               const uint8_t* lo2,
               uint64_t dim,
               uint64_t x,
               uint64_t y) {
    const uint64_t mask_hi = (1ULL << x) - 1ULL;
    const uint64_t mask_lo = (1ULL << y) - 1ULL;
    float sum = 0.0F;
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const uint64_t h1 = load_u64(hi1 + (i * x) / 8);
        const uint64_t l1 = load_u64(lo1 + (i * y) / 8);
        const uint64_t h2 = load_u64(hi2 + (i * x) / 8);
        const uint64_t l2 = load_u64(lo2 + (i * y) / 8);
        for (uint64_t k = 0; k < 8; ++k) {
            const uint32_t a = static_cast<uint32_t>(((h1 >> (k * x)) & mask_hi) << y) |
                               static_cast<uint32_t>((l1 >> (k * y)) & mask_lo);
            const uint32_t b = static_cast<uint32_t>(((h2 >> (k * x)) & mask_hi) << y) |
                               static_cast<uint32_t>((l2 >> (k * y)) & mask_lo);
            sum += static_cast<float>(a) * static_cast<float>(b);
        }
    }
    for (; i < dim; ++i) {
        const uint32_t a = (load_bits(hi1, i * x, x) << y) | load_bits(lo1, i * y, y);
        const uint32_t b = (load_bits(hi2, i * x, x) << y) | load_bits(lo2, i * y, y);
        sum += static_cast<float>(a) * static_cast<float>(b);
    }
    return sum;
}

float
FloatByteIP(const float* q, const uint8_t* codes, uint64_t dim) {
    float sum = 0.0F;
    for (uint64_t i = 0; i < dim; ++i) {
        sum += q[i] * static_cast<float>(codes[i]);
    }
    return sum;
}

float
ByteByteIP(const uint8_t* a, const uint8_t* b, uint64_t dim) {
    float sum = 0.0F;
    for (uint64_t i = 0; i < dim; ++i) {
        sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return sum;
}

}  // namespace generic

#if defined(VSAG_HAS_X86_INTRINSICS)
#define VSAG_SQ_SPLIT_AVX512_TARGET \
    __attribute__((target("avx2,fma,avx512f,avx512cd,avx512vl,avx512bw,avx512dq")))

namespace avx512 {

VSAG_SQ_SPLIT_AVX512_TARGET
float
FloatPackedIP(
    const float* q, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y) {
    __m256 acc = _mm256_setzero_ps();
    const __m512i shift_hi = _mm512_set_epi64(7 * x, 6 * x, 5 * x, 4 * x, 3 * x, 2 * x, x, 0);
    const __m512i shift_lo = _mm512_set_epi64(7 * y, 6 * y, 5 * y, 4 * y, 3 * y, 2 * y, y, 0);
    const __m512i mask_hi = _mm512_set1_epi64(static_cast<int64_t>((1ULL << x) - 1ULL));
    const __m512i mask_lo = _mm512_set1_epi64(static_cast<int64_t>((1ULL << y) - 1ULL));
    const __m512i shift_y = _mm512_set1_epi64(static_cast<int64_t>(y));
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m512i hv = _mm512_set1_epi64(static_cast<int64_t>(load_u64(hi + (i * x) / 8)));
        const __m512i lv = _mm512_set1_epi64(static_cast<int64_t>(load_u64(lo + (i * y) / 8)));
        const __m512i h = _mm512_and_si512(_mm512_srlv_epi64(hv, shift_hi), mask_hi);
        const __m512i l = _mm512_and_si512(_mm512_srlv_epi64(lv, shift_lo), mask_lo);
        const __m512i codes = _mm512_or_si512(_mm512_sllv_epi64(h, shift_y), l);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm512_cvtepi64_ps(codes), acc);
    }
    const __m128 low_lane = _mm256_castps256_ps128(acc);
    const __m128 high_lane = _mm256_extractf128_ps(acc, 1);
    __m128 reduced = _mm_add_ps(low_lane, high_lane);
    reduced = _mm_hadd_ps(reduced, reduced);
    reduced = _mm_hadd_ps(reduced, reduced);
    float sum = _mm_cvtss_f32(reduced);
    for (; i < dim; ++i) {
        sum += q[i] * static_cast<float>((load_bits(hi, i * x, x) << y) | load_bits(lo, i * y, y));
    }
    return sum;
}

VSAG_SQ_SPLIT_AVX512_TARGET
float
BytePackedIP(
    const uint8_t* qb, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y) {
    __m256 acc = _mm256_setzero_ps();
    const __m512i shift_hi = _mm512_set_epi64(7 * x, 6 * x, 5 * x, 4 * x, 3 * x, 2 * x, x, 0);
    const __m512i shift_lo = _mm512_set_epi64(7 * y, 6 * y, 5 * y, 4 * y, 3 * y, 2 * y, y, 0);
    const __m512i mask_hi = _mm512_set1_epi64(static_cast<int64_t>((1ULL << x) - 1ULL));
    const __m512i mask_lo = _mm512_set1_epi64(static_cast<int64_t>((1ULL << y) - 1ULL));
    const __m512i shift_y = _mm512_set1_epi64(static_cast<int64_t>(y));
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m512i hv = _mm512_set1_epi64(static_cast<int64_t>(load_u64(hi + (i * x) / 8)));
        const __m512i lv = _mm512_set1_epi64(static_cast<int64_t>(load_u64(lo + (i * y) / 8)));
        const __m512i h = _mm512_and_si512(_mm512_srlv_epi64(hv, shift_hi), mask_hi);
        const __m512i l = _mm512_and_si512(_mm512_srlv_epi64(lv, shift_lo), mask_lo);
        const __m512i codes = _mm512_or_si512(_mm512_sllv_epi64(h, shift_y), l);
        const __m512i qv =
            _mm512_cvtepu8_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(qb + i)));
        acc = _mm256_fmadd_ps(_mm512_cvtepi64_ps(qv), _mm512_cvtepi64_ps(codes), acc);
    }
    const __m128 low_lane = _mm256_castps256_ps128(acc);
    const __m128 high_lane = _mm256_extractf128_ps(acc, 1);
    __m128 reduced = _mm_add_ps(low_lane, high_lane);
    reduced = _mm_hadd_ps(reduced, reduced);
    reduced = _mm_hadd_ps(reduced, reduced);
    float sum = _mm_cvtss_f32(reduced);
    for (; i < dim; ++i) {
        sum += static_cast<float>(qb[i]) *
               static_cast<float>((load_bits(hi, i * x, x) << y) | load_bits(lo, i * y, y));
    }
    return sum;
}

VSAG_SQ_SPLIT_AVX512_TARGET
float
PackedPackedIP(const uint8_t* hi1,
               const uint8_t* lo1,
               const uint8_t* hi2,
               const uint8_t* lo2,
               uint64_t dim,
               uint64_t x,
               uint64_t y) {
    __m256 acc = _mm256_setzero_ps();
    const __m512i shift_hi = _mm512_set_epi64(7 * x, 6 * x, 5 * x, 4 * x, 3 * x, 2 * x, x, 0);
    const __m512i shift_lo = _mm512_set_epi64(7 * y, 6 * y, 5 * y, 4 * y, 3 * y, 2 * y, y, 0);
    const __m512i mask_hi = _mm512_set1_epi64(static_cast<int64_t>((1ULL << x) - 1ULL));
    const __m512i mask_lo = _mm512_set1_epi64(static_cast<int64_t>((1ULL << y) - 1ULL));
    const __m512i shift_y = _mm512_set1_epi64(static_cast<int64_t>(y));
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m512i h1 = _mm512_set1_epi64(static_cast<int64_t>(load_u64(hi1 + (i * x) / 8)));
        const __m512i l1 = _mm512_set1_epi64(static_cast<int64_t>(load_u64(lo1 + (i * y) / 8)));
        const __m512i h2 = _mm512_set1_epi64(static_cast<int64_t>(load_u64(hi2 + (i * x) / 8)));
        const __m512i l2 = _mm512_set1_epi64(static_cast<int64_t>(load_u64(lo2 + (i * y) / 8)));
        const __m512i c1 = _mm512_or_si512(
            _mm512_sllv_epi64(_mm512_and_si512(_mm512_srlv_epi64(h1, shift_hi), mask_hi), shift_y),
            _mm512_and_si512(_mm512_srlv_epi64(l1, shift_lo), mask_lo));
        const __m512i c2 = _mm512_or_si512(
            _mm512_sllv_epi64(_mm512_and_si512(_mm512_srlv_epi64(h2, shift_hi), mask_hi), shift_y),
            _mm512_and_si512(_mm512_srlv_epi64(l2, shift_lo), mask_lo));
        acc = _mm256_fmadd_ps(_mm512_cvtepi64_ps(c1), _mm512_cvtepi64_ps(c2), acc);
    }
    const __m128 low_lane = _mm256_castps256_ps128(acc);
    const __m128 high_lane = _mm256_extractf128_ps(acc, 1);
    __m128 reduced = _mm_add_ps(low_lane, high_lane);
    reduced = _mm_hadd_ps(reduced, reduced);
    reduced = _mm_hadd_ps(reduced, reduced);
    float sum = _mm_cvtss_f32(reduced);
    for (; i < dim; ++i) {
        const uint32_t a = (load_bits(hi1, i * x, x) << y) | load_bits(lo1, i * y, y);
        const uint32_t b = (load_bits(hi2, i * x, x) << y) | load_bits(lo2, i * y, y);
        sum += static_cast<float>(a) * static_cast<float>(b);
    }
    return sum;
}

VSAG_SQ_SPLIT_AVX512_TARGET
float
ByteByteIP(const uint8_t* a, const uint8_t* b, uint64_t dim) {
    __m256 acc = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256i av =
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i)));
        const __m256i bv =
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i)));
        acc = _mm256_add_ps(acc, _mm256_cvtepi32_ps(_mm256_mullo_epi32(av, bv)));
    }
    const __m128 low_lane = _mm256_castps256_ps128(acc);
    const __m128 high_lane = _mm256_extractf128_ps(acc, 1);
    __m128 reduced = _mm_add_ps(low_lane, high_lane);
    reduced = _mm_hadd_ps(reduced, reduced);
    reduced = _mm_hadd_ps(reduced, reduced);
    float sum = _mm_cvtss_f32(reduced);
    for (; i < dim; ++i) {
        sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return sum;
}

VSAG_SQ_SPLIT_AVX512_TARGET
float
FloatByteIP(const float* q, const uint8_t* codes, uint64_t dim) {
    __m256 acc = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m512i cv =
            _mm512_cvtepu8_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i)));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm512_cvtepi64_ps(cv), acc);
    }
    const __m128 low_lane = _mm256_castps256_ps128(acc);
    const __m128 high_lane = _mm256_extractf128_ps(acc, 1);
    __m128 reduced = _mm_add_ps(low_lane, high_lane);
    reduced = _mm_hadd_ps(reduced, reduced);
    reduced = _mm_hadd_ps(reduced, reduced);
    float sum = _mm_cvtss_f32(reduced);
    for (; i < dim; ++i) {
        sum += q[i] * static_cast<float>(codes[i]);
    }
    return sum;
}

}  // namespace avx512
#else
namespace avx512 {
float
FloatPackedIP(
    const float* q, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y) {
    return generic::FloatPackedIP(q, hi, lo, dim, x, y);
}
float
BytePackedIP(
    const uint8_t* qb, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y) {
    return generic::BytePackedIP(qb, hi, lo, dim, x, y);
}
float
PackedPackedIP(const uint8_t* hi1,
               const uint8_t* lo1,
               const uint8_t* hi2,
               const uint8_t* lo2,
               uint64_t dim,
               uint64_t x,
               uint64_t y) {
    return generic::PackedPackedIP(hi1, lo1, hi2, lo2, dim, x, y);
}
float
FloatByteIP(const float* q, const uint8_t* codes, uint64_t dim) {
    return generic::FloatByteIP(q, codes, dim);
}
float
ByteByteIP(const uint8_t* a, const uint8_t* b, uint64_t dim) {
    return generic::ByteByteIP(a, b, dim);
}
}  // namespace avx512
#endif

// Runtime dispatch: AVX-512 when available, generic otherwise. The pointers are initialized once.
SQFloatPackedIPType SQFloatPackedIP = nullptr;
SQBytePackedIPType SQBytePackedIP = nullptr;
SQPackedPackedIPType SQPackedPackedIP = nullptr;
SQFloatByteIPType SQFloatByteIP = nullptr;
SQByteByteIPType SQByteByteIP = nullptr;

namespace {
const bool kSQDispatchReady = []() {
    const bool avx512 = SimdStatus::SupportAVX512();
    SQFloatPackedIP = avx512 ? avx512::FloatPackedIP : generic::FloatPackedIP;
    SQBytePackedIP = avx512 ? avx512::BytePackedIP : generic::BytePackedIP;
    SQPackedPackedIP = avx512 ? avx512::PackedPackedIP : generic::PackedPackedIP;
    SQFloatByteIP = avx512 ? avx512::FloatByteIP : generic::FloatByteIP;
    SQByteByteIP = avx512 ? avx512::ByteByteIP : generic::ByteByteIP;
    return true;
}();
}  // namespace

}  // namespace vsag
