
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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "simd.h"
#include "simd/int8_simd.h"
#include "simd/kernels/rabitq_pack.h"
#include "vsag/attribute.h"

#if defined(ENABLE_AVX2)
#include <immintrin.h>

#include "simd/kernels/kernels.h"
#include "simd/traits/simd_traits_avx2.h"

inline float
avx2_reduce_add_ps(__m256 a) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, a);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}
#define AVX2_REDUCE_ADD_PS(a) avx2_reduce_add_ps(a)
#endif

namespace vsag::avx2 {

namespace {

#if defined(ENABLE_AVX2)
constexpr std::array<uint64_t, 256>
MakeBitExpandLut() {
    std::array<uint64_t, 256> result{};
    for (uint64_t value = 0; value < result.size(); ++value) {
        for (uint64_t lane = 0; lane < 8; ++lane) {
            result[value] |= ((value >> lane) & 1ULL) << (lane * 8);
        }
    }
    return result;
}

constexpr auto kBitExpandLut = MakeBitExpandLut();
static_assert(kBitExpandLut[0x01] == 0x0000000000000001ULL);
static_assert(kBitExpandLut[0x80] == 0x0100000000000000ULL);
static_assert(kBitExpandLut[0xA5] == 0x0100010000010001ULL);

inline __m128i
ExpandCompactSupplementTopBits(const uint8_t* plane, uint64_t group, uint32_t bit) {
    const uint64_t low = kBitExpandLut[plane[group * 2U]] << bit;
    const uint64_t high = kBitExpandLut[plane[group * 2U + 1U]] << bit;
    return _mm_set_epi64x(static_cast<int64_t>(high), static_cast<int64_t>(low));
}

inline void
AccumulateCompactSupplement16(__m128i codes, const float* vector, __m256& sum) {
    const auto low_codes = _mm256_cvtepu8_epi32(codes);
    const auto high_codes = _mm256_cvtepu8_epi32(_mm_srli_si128(codes, 8));
    sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(low_codes), _mm256_loadu_ps(vector), sum);
    sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(high_codes), _mm256_loadu_ps(vector + 8), sum);
}

inline void
DecodeCompactSupplementChunk(const uint8_t* packed, uint32_t bits, __m128i* values) {
    const auto low_four_mask = _mm_set1_epi8(0x0F);
    const auto low_six_mask = _mm_set1_epi8(0x3F);
    const auto part0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const auto part1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + 16U));
    if (bits == 5U) {
        values[0] = _mm_and_si128(part0, low_four_mask);
        values[1] = _mm_and_si128(_mm_srli_epi16(part0, 4), low_four_mask);
        values[2] = _mm_and_si128(part1, low_four_mask);
        values[3] = _mm_and_si128(_mm_srli_epi16(part1, 4), low_four_mask);
        for (uint64_t group = 0; group < 4U; ++group) {
            values[group] = _mm_or_si128(values[group],
                                         ExpandCompactSupplementTopBits(packed + 32U, group, 4U));
        }
        return;
    }

    const auto part2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + 32U));
    values[0] = _mm_and_si128(part0, low_six_mask);
    values[1] = _mm_and_si128(part1, low_six_mask);
    values[2] = _mm_and_si128(part2, low_six_mask);
    values[3] =
        _mm_or_si128(_mm_and_si128(_mm_srli_epi16(part0, 6), _mm_set1_epi8(0x03)),
                     _mm_or_si128(_mm_and_si128(_mm_srli_epi16(part1, 4), _mm_set1_epi8(0x0C)),
                                  _mm_and_si128(_mm_srli_epi16(part2, 2), _mm_set1_epi8(0x30))));
    if (bits == 7U) {
        for (uint64_t group = 0; group < 4U; ++group) {
            values[group] = _mm_or_si128(values[group],
                                         ExpandCompactSupplementTopBits(packed + 48U, group, 6U));
        }
    }
}

inline void
PackCompactSupplementChunk(const uint8_t* scalar_codes, uint8_t* packed, uint32_t bits) {
    __m128i values[4];
    for (uint64_t group = 0; group < 4U; ++group) {
        values[group] =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(scalar_codes + group * 16U));
    }

    if (bits == 5U) {
        const auto low_mask = _mm_set1_epi8(0x0F);
        const auto part0 = _mm_or_si128(_mm_and_si128(values[0], low_mask),
                                        _mm_slli_epi16(_mm_and_si128(values[1], low_mask), 4));
        const auto part1 = _mm_or_si128(_mm_and_si128(values[2], low_mask),
                                        _mm_slli_epi16(_mm_and_si128(values[3], low_mask), 4));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(packed), part0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(packed + 16U), part1);
    } else {
        const auto low_mask = _mm_set1_epi8(0x3F);
        const auto part0 =
            _mm_or_si128(_mm_and_si128(values[0], low_mask),
                         _mm_slli_epi16(_mm_and_si128(values[3], _mm_set1_epi8(0x03)), 6));
        const auto part1 =
            _mm_or_si128(_mm_and_si128(values[1], low_mask),
                         _mm_slli_epi16(_mm_and_si128(values[3], _mm_set1_epi8(0x0C)), 4));
        const auto part2 =
            _mm_or_si128(_mm_and_si128(values[2], low_mask),
                         _mm_slli_epi16(_mm_and_si128(values[3], _mm_set1_epi8(0x30)), 2));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(packed), part0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(packed + 16U), part1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(packed + 32U), part2);
    }

    if (bits == 5U or bits == 7U) {
        const uint32_t top_bit = bits - 1U;
        const auto bit_mask =
            _mm256_set1_epi8(static_cast<char>(static_cast<uint8_t>(1U << top_bit)));
        const auto zero = _mm256_setzero_si256();
        const auto low_values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scalar_codes));
        const auto high_values =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scalar_codes + 32U));
        const uint32_t low = ~static_cast<uint32_t>(
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(low_values, bit_mask), zero)));
        const uint32_t high = ~static_cast<uint32_t>(
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(high_values, bit_mask), zero)));
        const uint64_t top_plane =
            static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32U);
        std::memcpy(packed + (bits == 5U ? 32U : 48U), &top_plane, sizeof(top_plane));
    }
}
#endif

}  // namespace

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx2::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx2::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0F - avx2::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx2::INT8ComputeL2Sqr(pVect1, pVect2, qty);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx2::INT8ComputeIP(pVect1, pVect2, qty);
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -avx2::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_AVX2)
    simd::PQDistanceFloat256Impl<simd::SimdTraits<simd::Avx2Tag>>(
        single_dim_centers, single_dim_val, result, &avx::PQDistanceFloat256);
#else
    return avx::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

void
Prefetch(const void* data) {
    avx::Prefetch(data);
}

#if defined(ENABLE_AVX2)
__inline __m128i __attribute__((__always_inline__)) load_8_char(const uint8_t* data) {
    return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(data));
}
#endif

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::ComputeIPImpl<simd::SimdTraits<simd::Avx2Tag>>(
        query, codes, dim, &sse::FP32ComputeIP);
#else
    return avx::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::Avx2Tag>>(
        query, codes, dim, &sse::FP32ComputeL2Sqr);
#else
    return avx::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP32SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const float* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
#if defined(ENABLE_AVX2)
    __m256 q_vec = _mm256_set1_ps(query_val);
    uint32_t i = 0;
    for (; i + 8 <= num; i += 8) {
        __m128i id_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
        __m256i idx_vec = _mm256_cvtepu16_epi32(id_vec);
        __m256 val_vec = _mm256_loadu_ps(vals + i);

        __m256 dist_vec = _mm256_i32gather_ps(dists, idx_vec, 4);
        dist_vec = _mm256_fmadd_ps(val_vec, q_vec, dist_vec);

        alignas(32) float res[8];
        alignas(32) int32_t indices[8];
        _mm256_store_ps(res, dist_vec);
        _mm256_store_si256(reinterpret_cast<__m256i*>(indices), idx_vec);

        for (int k = 0; k < 8; ++k) {
            dists[indices[k]] = res[k];
        }
    }
    for (; i < num; ++i) {
        dists[ids[i]] += vals[i] * query_val;
    }
#else
    return avx::FP32SparseAccumulate(dists, ids, vals, query_val, num);
#endif
}

void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4) {
#if defined(ENABLE_AVX2)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::Avx2Tag>, simd::Batch4Kind::IP>(
        query,
        dim,
        codes1,
        codes2,
        codes3,
        codes4,
        result1,
        result2,
        result3,
        result4,
        &avx::FP32ComputeIPBatch4);
#else
    return avx::FP32ComputeIPBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4) {
#if defined(ENABLE_AVX2)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::Avx2Tag>, simd::Batch4Kind::L2>(
        query,
        dim,
        codes1,
        codes2,
        codes3,
        codes4,
        result1,
        result2,
        result3,
        result4,
        &avx::FP32ComputeL2SqrBatch4);
#else
    return avx::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX2)
    simd::BinaryOpImpl<simd::SimdTraits<simd::Avx2Tag>, simd::BinaryOp::Sub>(
        x, y, z, dim, &sse::FP32Sub);
#else
    return sse::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX2)
    simd::BinaryOpImpl<simd::SimdTraits<simd::Avx2Tag>, simd::BinaryOp::Add>(
        x, y, z, dim, &sse::FP32Add);
#else
    return sse::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX2)
    simd::BinaryOpImpl<simd::SimdTraits<simd::Avx2Tag>, simd::BinaryOp::Mul>(
        x, y, z, dim, &sse::FP32Mul);
#else
    return sse::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX2)
    simd::BinaryOpImpl<simd::SimdTraits<simd::Avx2Tag>, simd::BinaryOp::Div>(
        x, y, z, dim, &sse::FP32Div);
#else
    return sse::FP32Div(x, y, z, dim);
#endif
}
float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::ReduceAddImpl<simd::SimdTraits<simd::Avx2Tag>>(x, dim, &sse::FP32ReduceAdd);
#else
    return sse::FP32ReduceAdd(x, dim);
#endif
}

#if defined(ENABLE_AVX2)
__inline __m256i __attribute__((__always_inline__)) load_8_short(const uint16_t* data) {
    __m128i bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
    __m256i bf32 = _mm256_cvtepu16_epi32(bf16);
    return _mm256_slli_epi32(bf32, 16);
}
#endif

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::HalfComputeIPImpl<simd::BF16Traits<simd::Avx2BF16Tag>>(
        query, codes, dim, &avx::BF16ComputeIP);
#else
    return avx::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::HalfComputeL2SqrImpl<simd::BF16Traits<simd::Avx2BF16Tag>>(
        query, codes, dim, &avx::BF16ComputeL2Sqr);
#else
    return avx::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::HalfComputeIPImpl<simd::FP16Traits<simd::Avx2FP16Tag>>(
        query, codes, dim, &avx::FP16ComputeIP);
#else
    return avx::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::HalfComputeL2SqrImpl<simd::FP16Traits<simd::Avx2FP16Tag>>(
        query, codes, dim, &avx::FP16ComputeL2Sqr);
#else
    return avx::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP16SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const uint16_t* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
#if defined(ENABLE_AVX2)
    __m256 q_vec = _mm256_set1_ps(query_val);
    uint32_t i = 0;
    for (; i + 8 <= num; i += 8) {
        __m128i id_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
        __m256i idx_vec = _mm256_cvtepu16_epi32(id_vec);
        __m128i val_half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vals + i));
        __m256 val_vec = _mm256_cvtph_ps(val_half);

        __m256 dist_vec = _mm256_i32gather_ps(dists, idx_vec, 4);
        dist_vec = _mm256_fmadd_ps(val_vec, q_vec, dist_vec);

        alignas(32) float res[8];
        alignas(32) int32_t indices[8];
        _mm256_store_ps(res, dist_vec);
        _mm256_store_si256(reinterpret_cast<__m256i*>(indices), idx_vec);

        for (int k = 0; k < 8; ++k) {
            dists[indices[k]] = res[k];
        }
    }
    for (; i < num; ++i) {
        dists[ids[i]] += generic::FP16ToFloat(vals[i]) * query_val;
    }
#else
    return avx::FP16SparseAccumulate(dists, ids, vals, query_val, num);
#endif
}

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::Int8ComputeL2SqrImpl<simd::Int8Traits<simd::Avx2Int8Tag>>(
        query, codes, dim, &avx::INT8ComputeL2Sqr);
#else
    return sse::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::Int8ComputeIPImpl<simd::Int8Traits<simd::Avx2Int8Tag>>(
        query, codes, dim, &avx::INT8ComputeIP);
#else
    return sse::INT8ComputeIP(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ8ComputeIPImpl<simd::SQ8Traits<simd::Avx2SQ8Tag>>(
        query, codes, lower_bound, diff, dim, &avx::SQ8ComputeIP);
#else
    return avx::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ8ComputeL2SqrImpl<simd::SQ8Traits<simd::Avx2SQ8Tag>>(
        query, codes, lower_bound, diff, dim, &avx::SQ8ComputeL2Sqr);
#else
    return avx::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ8ComputeCodesIPImpl<simd::SQ8Traits<simd::Avx2SQ8Tag>>(
        codes1, codes2, lower_bound, diff, dim, &avx::SQ8ComputeCodesIP);
#else
    return avx::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ8ComputeCodesL2SqrImpl<simd::SQ8Traits<simd::Avx2SQ8Tag>>(
        codes1, codes2, lower_bound, diff, dim, &avx::SQ8ComputeCodesL2Sqr);
#else
    return avx::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

void
SQ8SparseAccumulate(float* RESTRICT dists,
                    const uint16_t* RESTRICT ids,
                    const uint8_t* RESTRICT vals,
                    float query_val,
                    uint32_t num) {
#if defined(ENABLE_AVX2)
    __m256 q_vec = _mm256_set1_ps(query_val);
    uint32_t i = 0;
    for (; i + 8 <= num; i += 8) {
        __m128i id_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
        __m256i idx_vec = _mm256_cvtepu16_epi32(id_vec);
        __m256i val_i32 = _mm256_cvtepu8_epi32(load_8_char(vals + i));
        __m256 val_vec = _mm256_cvtepi32_ps(val_i32);

        __m256 dist_vec = _mm256_i32gather_ps(dists, idx_vec, 4);
        dist_vec = _mm256_fmadd_ps(val_vec, q_vec, dist_vec);

        alignas(32) float res[8];
        alignas(32) int32_t indices[8];
        _mm256_store_ps(res, dist_vec);
        _mm256_store_si256(reinterpret_cast<__m256i*>(indices), idx_vec);

        for (int k = 0; k < 8; ++k) {
            dists[indices[k]] = res[k];
        }
    }
    for (; i < num; ++i) {
        dists[ids[i]] += static_cast<float>(vals[i]) * query_val;
    }
#else
    return avx::SQ8SparseAccumulate(dists, ids, vals, query_val, num);
#endif
}

#if defined(ENABLE_AVX2)

__inline __m128i __attribute__((__always_inline__)) load_4_char(const uint8_t* data) {
    return _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, data[3], data[2], data[1], data[0]);
}

__inline void __attribute__((__always_inline__)) SQ4Decode16Values(const uint8_t* codes,
                                                                   uint64_t offset,
                                                                   __m256& values01,
                                                                   __m256& values23,
                                                                   const float* lower_bound,
                                                                   const float* diff) {
    // Load 8 bytes (16 4-bit values)
    __m128i code_vec = load_4_char(codes + (offset >> 1));
    __m128i code_vec2 = load_4_char(codes + (offset >> 1) + 4);

    // Extract low nibbles (values 0,2,4,6,8,10,12,14) - even indices
    __m128i low_nibbles1 = _mm_and_si128(code_vec, _mm_set1_epi8(0x0F));
    __m128i low_nibbles2 = _mm_and_si128(code_vec2, _mm_set1_epi8(0x0F));

    // Extract high nibbles (values 1,3,5,7,9,11,13,15) - odd indices
    __m128i high_nibbles1 = _mm_and_si128(_mm_srli_epi16(code_vec, 4), _mm_set1_epi8(0x0F));
    __m128i high_nibbles2 = _mm_and_si128(_mm_srli_epi16(code_vec2, 4), _mm_set1_epi8(0x0F));

    // Interleave low and high nibbles to get correct order
    __m128i interleaved1 = _mm_unpacklo_epi8(low_nibbles1, high_nibbles1);
    __m128i interleaved2 = _mm_unpacklo_epi8(low_nibbles2, high_nibbles2);

    // Convert to float and scale - first 8 values
    __m128i low_part1 = _mm_cvtepu8_epi32(interleaved1);
    __m128i high_part1 = _mm_cvtepu8_epi32(_mm_srli_si128(interleaved1, 4));
    __m128 values0 = _mm_cvtepi32_ps(low_part1);
    __m128 values1 = _mm_cvtepi32_ps(high_part1);

    // Convert to float and scale - next 8 values
    __m128i low_part2 = _mm_cvtepu8_epi32(interleaved2);
    __m128i high_part2 = _mm_cvtepu8_epi32(_mm_srli_si128(interleaved2, 4));
    __m128 values2 = _mm_cvtepi32_ps(low_part2);
    __m128 values3 = _mm_cvtepi32_ps(high_part2);

    // Combine into AVX vectors
    values01 = _mm256_set_m128(values1, values0);
    values23 = _mm256_set_m128(values3, values2);

    // Scale by 1/15.0
    __m256 scale = _mm256_set1_ps(1.0F / 15.0F);
    values01 = _mm256_mul_ps(values01, scale);
    values23 = _mm256_mul_ps(values23, scale);

    // Apply diff and lower_bound
    __m256 diff_vec0 = _mm256_loadu_ps(diff + offset);
    __m256 diff_vec1 = _mm256_loadu_ps(diff + offset + 8);
    __m256 lb_vec0 = _mm256_loadu_ps(lower_bound + offset);
    __m256 lb_vec1 = _mm256_loadu_ps(lower_bound + offset + 8);

    values01 = _mm256_fmadd_ps(values01, diff_vec0, lb_vec0);
    values23 = _mm256_fmadd_ps(values23, diff_vec1, lb_vec1);
}
#endif
float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ4ComputeIPImpl<simd::SQ4Traits<simd::Avx2SQ4Tag>>(
        query, codes, lower_bound, diff, dim, &sse::SQ4ComputeIP);
#else
    return sse::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ4ComputeL2SqrImpl<simd::SQ4Traits<simd::Avx2SQ4Tag>>(
        query, codes, lower_bound, diff, dim, &sse::SQ4ComputeL2Sqr);
#else
    return sse::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ4ComputeCodesIPImpl<simd::SQ4Traits<simd::Avx2SQ4Tag>>(
        codes1, codes2, lower_bound, diff, dim, &sse::SQ4ComputeCodesIP);
#else
    return sse::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ4ComputeCodesL2SqrImpl<simd::SQ4Traits<simd::Avx2SQ4Tag>>(
        codes1, codes2, lower_bound, diff, dim, &sse::SQ4ComputeCodesL2Sqr);
#else
    return sse::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ4UniformComputeCodesIPImpl<simd::UniformCodeTraits<simd::Avx2UniformTag>>(
        codes1, codes2, dim, &avx::SQ4UniformComputeCodesIP);
#else
    return avx::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::SQ8UniformComputeCodesIPImpl<simd::UniformCodeTraits<simd::Avx2UniformTag>>(
        codes1, codes2, dim, &avx::SQ8UniformComputeCodesIP);
#else
    return avx::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

void
SQ8UniformComputeCodesIPBatch(const uint8_t* RESTRICT query,
                              const uint8_t* RESTRICT codes,
                              uint64_t dim,
                              uint64_t n_codes,
                              uint64_t code_stride,
                              float* RESTRICT out) {
    for (uint64_t i = 0; i < n_codes; ++i) {
        out[i] = avx2::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
}

float
RaBitQFloatSQIP(const float* vector, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::RaBitQFloatScalarIPImpl<simd::SQ8Traits<simd::Avx2SQ8Tag>>(
        vector, codes, dim, &generic::RaBitQFloatSQIP);
#else
    return generic::RaBitQFloatSQIP(vector, codes, dim);
#endif
}

uint64_t
RaBitQCodeCodeIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::RaBitQScalarCodesIPImpl<simd::UniformCodeTraits<simd::Avx2UniformTag>>(
        codes1, codes2, dim, &generic::RaBitQCodeCodeIP);
#else
    return generic::RaBitQCodeCodeIP(codes1, codes2, dim);
#endif
}
void
RaBitQPackScalarToSplitPlanes(const uint8_t* scalar_codes,
                              uint8_t* filter_planes,
                              uint8_t* supplement_planes,
                              uint64_t dim,
                              uint32_t total_bits,
                              uint32_t filter_bits) {
#if defined(ENABLE_AVX2)
    const uint64_t plane_bytes = (dim + 7) / 8;
    uint64_t d = 0;
    const __m256i zero = _mm256_setzero_si256();
    for (; d + 32 <= dim; d += 32) {
        const __m256i values =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scalar_codes + d));
        const uint64_t byte_index = d / 8;
        for (uint32_t bit = 0; bit < total_bits; ++bit) {
            const __m256i bit_mask =
                _mm256_set1_epi8(static_cast<char>(static_cast<uint8_t>(1U << bit)));
            const __m256i masked = _mm256_and_si256(values, bit_mask);
            const __m256i is_zero = _mm256_cmpeq_epi8(masked, zero);
            const uint32_t packed = ~static_cast<uint32_t>(_mm256_movemask_epi8(is_zero));
            auto* plane = simd::RaBitQGetSplitPlane(
                filter_planes, supplement_planes, plane_bytes, total_bits, filter_bits, bit);
            plane[byte_index] = static_cast<uint8_t>(packed);
            plane[byte_index + 1] = static_cast<uint8_t>(packed >> 8);
            plane[byte_index + 2] = static_cast<uint8_t>(packed >> 16);
            plane[byte_index + 3] = static_cast<uint8_t>(packed >> 24);
        }
    }
    simd::RaBitQPackScalarToSplitPlanesTail(
        scalar_codes, filter_planes, supplement_planes, dim, total_bits, filter_bits, d);
#else
    generic::RaBitQPackScalarToSplitPlanes(
        scalar_codes, filter_planes, supplement_planes, dim, total_bits, filter_bits);
#endif
}

void
RaBitQPackScalarToSplitCode(const uint8_t* scalar_codes,
                            uint8_t* filter_planes,
                            uint8_t* packed_supplement_code,
                            uint64_t dim,
                            uint32_t total_bits,
                            uint32_t filter_bits) {
#if defined(ENABLE_AVX2)
    const uint64_t plane_bytes = (dim + 7U) / 8U;
    if (plane_bytes != 0U and filter_bits != 0U) {
        std::memset(filter_planes, 0, plane_bytes * static_cast<uint64_t>(filter_bits));
    }
    uint64_t d = 0;
    const auto zero = _mm256_setzero_si256();
    for (; d + 32U <= dim; d += 32U) {
        const auto values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scalar_codes + d));
        for (uint32_t plane = 0; plane < filter_bits; ++plane) {
            const uint32_t logical_bit = total_bits - plane - 1U;
            const auto bit_mask =
                _mm256_set1_epi8(static_cast<char>(static_cast<uint8_t>(1U << logical_bit)));
            const auto is_zero = _mm256_cmpeq_epi8(_mm256_and_si256(values, bit_mask), zero);
            const uint32_t packed = ~static_cast<uint32_t>(_mm256_movemask_epi8(is_zero));
            std::memcpy(filter_planes + static_cast<uint64_t>(plane) * plane_bytes + d / 8U,
                        &packed,
                        sizeof(packed));
        }
    }
    simd::RaBitQPackScalarFilterPlanesTail(
        scalar_codes, filter_planes, dim, total_bits, filter_bits, d);
    const uint32_t supplement_bits = total_bits - filter_bits;
    const simd::RaBitQPackedSupplementLayout layout{dim, supplement_bits};
    if (layout.PackedSize() != 0U) {
        std::memset(packed_supplement_code, 0, layout.PackedSize());
        if (layout.UsesCompactChunks()) {
            for (uint64_t chunk = 0; chunk < layout.FullChunkCount(); ++chunk) {
                PackCompactSupplementChunk(
                    scalar_codes + chunk * simd::RaBitQPackedSupplementLayout::CHUNK_DIM,
                    packed_supplement_code + chunk * layout.ChunkBytes(),
                    supplement_bits);
            }
            simd::RaBitQPackScalarSupplementTail(scalar_codes, packed_supplement_code, layout);
        } else {
            simd::RaBitQPackScalarSupplementCode(
                scalar_codes, packed_supplement_code, dim, supplement_bits);
        }
    }
#else
    generic::RaBitQPackScalarToSplitCode(
        scalar_codes, filter_planes, packed_supplement_code, dim, total_bits, filter_bits);
#endif
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_AVX2)
    return simd::RaBitQFloatBinaryIPImpl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        vector, bits, dim, inv_sqrt_d, &avx::RaBitQFloatBinaryIP);
#else
    return avx::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
#endif
}

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results) {
#if defined(ENABLE_AVX2)
    simd::RaBitQFloatBinaryIPBatch4Impl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        vector,
        bits1,
        bits2,
        bits3,
        bits4,
        dim,
        inv_sqrt_d,
        results,
        &generic::RaBitQFloatBinaryIPBatch4);
#else
    avx::RaBitQFloatBinaryIPBatch4(vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
#endif
}

void
RaBitQFloatThreeBitIPBatch4(const float* vector,
                            const uint8_t* bits1,
                            const uint8_t* bits2,
                            const uint8_t* bits3,
                            const uint8_t* bits4,
                            uint64_t dim,
                            uint32_t reorder_bits,
                            float* results) {
#if defined(ENABLE_AVX2)
    results[0] = 0.0F;
    results[1] = 0.0F;
    results[2] = 0.0F;
    results[3] = 0.0F;
    if (dim == 0) {
        return;
    }
    if (dim < 8) {
        generic::RaBitQFloatThreeBitIPBatch4(
            vector, bits1, bits2, bits3, bits4, dim, reorder_bits, results);
        return;
    }

    const uint64_t plane_bytes = (dim + 7) / 8;
    const __m256 zero = _mm256_setzero_ps();
    const __m256i bit_masks = _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i zero_i = _mm256_setzero_si256();
    const __m256 weights[3] = {_mm256_set1_ps(static_cast<float>(1U << (reorder_bits + 2))),
                               _mm256_set1_ps(static_cast<float>(1U << (reorder_bits + 1))),
                               _mm256_set1_ps(static_cast<float>(1U << reorder_bits))};
    const uint8_t* codes[4] = {bits1, bits2, bits3, bits4};
    __m256 sums[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};

    uint64_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        const uint64_t byte_idx = d >> 3;
        const __m256 vec = _mm256_loadu_ps(vector + d);
        for (uint32_t i = 0; i < 4; ++i) {
            __m256 code = _mm256_setzero_ps();
            for (uint32_t bit = 0; bit < 3; ++bit) {
                const auto* plane = codes[i] + static_cast<uint64_t>(bit) * plane_bytes;
                __m256i mask = _mm256_set1_epi32(static_cast<int>(plane[byte_idx]));
                mask = _mm256_and_si256(mask, bit_masks);
                mask = _mm256_cmpeq_epi32(mask, zero_i);
                mask = _mm256_andnot_si256(mask, all_ones);
                code = _mm256_add_ps(
                    code, _mm256_blendv_ps(zero, weights[bit], _mm256_castsi256_ps(mask)));
            }
            sums[i] = _mm256_fmadd_ps(code, vec, sums[i]);
        }
    }

    alignas(32) float lanes[8];
    for (uint32_t i = 0; i < 4; ++i) {
        _mm256_store_ps(lanes, sums[i]);
        results[i] =
            lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    }
    for (; d < dim; ++d) {
        const uint64_t byte_idx = d >> 3;
        const uint8_t bit_mask = static_cast<uint8_t>(1U << (d & 7));
        const float value = vector[d];
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t code = 0;
            for (uint32_t bit = 0; bit < 3; ++bit) {
                const auto* plane = codes[i] + static_cast<uint64_t>(bit) * plane_bytes;
                if ((plane[byte_idx] & bit_mask) != 0U) {
                    code += 1U << (reorder_bits + 2 - bit);
                }
            }
            results[i] += value * static_cast<float>(code);
        }
    }
#else
    avx::RaBitQFloatThreeBitIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, reorder_bits, results);
#endif
}

float
RaBitQFloatTwoBitCenteredIP(const float* vector, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::RaBitQFloatTwoBitCenteredIPImpl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        vector, bits, dim, &generic::RaBitQFloatTwoBitCenteredIP);
#else
    return generic::RaBitQFloatTwoBitCenteredIP(vector, bits, dim);
#endif
}

void
RaBitQFloatTwoBitCenteredIPBatch4(const float* vector,
                                  const uint8_t* bits1,
                                  const uint8_t* bits2,
                                  const uint8_t* bits3,
                                  const uint8_t* bits4,
                                  uint64_t dim,
                                  float* results) {
#if defined(ENABLE_AVX2)
    simd::RaBitQFloatTwoBitCenteredIPBatch4Impl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        vector,
        bits1,
        bits2,
        bits3,
        bits4,
        dim,
        results,
        &generic::RaBitQFloatTwoBitCenteredIPBatch4);
#else
    generic::RaBitQFloatTwoBitCenteredIPBatch4(vector, bits1, bits2, bits3, bits4, dim, results);
#endif
}

float
RaBitQFloatThreeBitCenteredIP(const float* vector, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::RaBitQFloatThreeBitCenteredIPImpl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        vector, bits, dim, &generic::RaBitQFloatThreeBitCenteredIP);
#else
    return generic::RaBitQFloatThreeBitCenteredIP(vector, bits, dim);
#endif
}

void
RaBitQFloatThreeBitCenteredIPBatch4(const float* vector,
                                    const uint8_t* bits1,
                                    const uint8_t* bits2,
                                    const uint8_t* bits3,
                                    const uint8_t* bits4,
                                    uint64_t dim,
                                    float* results) {
#if defined(ENABLE_AVX2)
    simd::RaBitQFloatThreeBitCenteredIPBatch4Impl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        vector,
        bits1,
        bits2,
        bits3,
        bits4,
        dim,
        results,
        &generic::RaBitQFloatThreeBitCenteredIPBatch4);
#else
    generic::RaBitQFloatThreeBitCenteredIPBatch4(vector, bits1, bits2, bits3, bits4, dim, results);
#endif
}

float
RaBitQFloatThreeBitIPByLookup(const float* lookup,
                              const uint8_t* bits,
                              uint64_t dim,
                              uint32_t reorder_bits) {
    return RaBitQFloatMultiBitIPByLookup(lookup, bits, dim, reorder_bits, 3);
}

float
RaBitQFloatMultiBitIPByLookup(const float* lookup,
                              const uint8_t* bits,
                              uint64_t dim,
                              uint32_t reorder_bits,
                              uint32_t filter_bits) {
#if defined(ENABLE_AVX2)
    const uint64_t plane_bytes = (dim + 7) / 8;
    __m256 sum = _mm256_setzero_ps();
    uint64_t block = 0;
    for (; block + 8 <= plane_bytes; block += 8) {
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto* plane = bits + static_cast<uint64_t>(bit) * plane_bytes;
            alignas(32) int indices[8];
            for (uint32_t lane = 0; lane < 8; ++lane) {
                indices[lane] = static_cast<int>((block + lane) * 256 + plane[block + lane]);
            }
            const __m256i index = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices));
            const __m256 values = _mm256_i32gather_ps(lookup, index, 4);
            const __m256 weight =
                _mm256_set1_ps(static_cast<float>(1U << (reorder_bits + filter_bits - bit - 1)));
            sum = _mm256_fmadd_ps(values, weight, sum);
        }
    }

    float result = AVX2_REDUCE_ADD_PS(sum);
    for (; block < plane_bytes; ++block) {
        const auto* block_lookup = lookup + block * 256;
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto* plane = bits + static_cast<uint64_t>(bit) * plane_bytes;
            const uint32_t weight = 1U << (reorder_bits + filter_bits - bit - 1);
            result += block_lookup[plane[block]] * static_cast<float>(weight);
        }
    }
    return result;
#else
    return generic::RaBitQFloatMultiBitIPByLookup(lookup, bits, dim, reorder_bits, filter_bits);
#endif
}

void
RaBitQFloatThreeBitIPBatch4ByLookup(const float* lookup,
                                    const uint8_t* bits1,
                                    const uint8_t* bits2,
                                    const uint8_t* bits3,
                                    const uint8_t* bits4,
                                    uint64_t dim,
                                    uint32_t reorder_bits,
                                    float* results) {
    RaBitQFloatMultiBitIPBatch4ByLookup(
        lookup, bits1, bits2, bits3, bits4, dim, reorder_bits, 3, results);
}

void
RaBitQFloatMultiBitIPBatch4ByLookup(const float* lookup,
                                    const uint8_t* bits1,
                                    const uint8_t* bits2,
                                    const uint8_t* bits3,
                                    const uint8_t* bits4,
                                    uint64_t dim,
                                    uint32_t reorder_bits,
                                    uint32_t filter_bits,
                                    float* results) {
#if defined(ENABLE_AVX2)
    results[0] = 0.0F;
    results[1] = 0.0F;
    results[2] = 0.0F;
    results[3] = 0.0F;

    const uint64_t plane_bytes = (dim + 7) / 8;
    const uint8_t* codes[4] = {bits1, bits2, bits3, bits4};
    __m256 sums[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};

    uint64_t block = 0;
    for (; block + 8 <= plane_bytes; block += 8) {
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t bit = 0; bit < filter_bits; ++bit) {
                const auto* plane = codes[i] + static_cast<uint64_t>(bit) * plane_bytes;
                alignas(32) int indices[8];
                for (uint32_t lane = 0; lane < 8; ++lane) {
                    indices[lane] = static_cast<int>((block + lane) * 256 + plane[block + lane]);
                }
                const __m256i index = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices));
                const __m256 values = _mm256_i32gather_ps(lookup, index, 4);
                const __m256 weight = _mm256_set1_ps(
                    static_cast<float>(1U << (reorder_bits + filter_bits - bit - 1)));
                sums[i] = _mm256_fmadd_ps(values, weight, sums[i]);
            }
        }
    }

    for (uint32_t i = 0; i < 4; ++i) {
        results[i] = AVX2_REDUCE_ADD_PS(sums[i]);
    }
    for (; block < plane_bytes; ++block) {
        const auto* block_lookup = lookup + block * 256;
        for (uint32_t i = 0; i < 4; ++i) {
            const auto* bits = codes[i];
            for (uint32_t bit = 0; bit < filter_bits; ++bit) {
                const auto* plane = bits + static_cast<uint64_t>(bit) * plane_bytes;
                const uint32_t weight = 1U << (reorder_bits + filter_bits - bit - 1);
                results[i] += block_lookup[plane[block]] * static_cast<float>(weight);
            }
        }
    }
#else
    generic::RaBitQFloatMultiBitIPBatch4ByLookup(
        lookup, bits1, bits2, bits3, bits4, dim, reorder_bits, filter_bits, results);
#endif
}

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits) {
#if defined(ENABLE_AVX2)
    return simd::RaBitQFloatSplitCodeIPImpl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
#else
    return avx::RaBitQFloatSplitCodeIP(vector, one_bit_code, supplement_code, dim, supplement_bits);
#endif
}

float
RaBitQFloatSupplementCodeIP(const float* vector,
                            const uint8_t* supplement_code,
                            uint64_t dim,
                            uint32_t supplement_bits) {
#if defined(ENABLE_AVX2)
    if (dim == 0 or supplement_bits == 0) {
        return 0.0F;
    }
    if (supplement_bits > 7) {
        return avx::RaBitQFloatSupplementCodeIP(vector, supplement_code, dim, supplement_bits);
    }

    const uint64_t plane_bytes = (dim + 7) / 8;
    __m256 sum = _mm256_setzero_ps();

    uint64_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        const uint64_t byte_idx = d >> 3;
        uint64_t packed_codes = 0;
        for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
            const auto* plane = supplement_code + static_cast<uint64_t>(bit) * plane_bytes;
            packed_codes |= kBitExpandLut[plane[byte_idx]] << bit;
        }
        const auto packed = _mm_cvtsi64_si128(static_cast<int64_t>(packed_codes));
        const auto code = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(packed));
        const __m256 vec = _mm256_loadu_ps(vector + d);
        sum = _mm256_fmadd_ps(code, vec, sum);
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result =
        lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; d < dim; ++d) {
        const uint64_t byte_idx = d >> 3;
        const uint8_t bit_mask = static_cast<uint8_t>(1U << (d & 7));
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
            const auto* plane = supplement_code + static_cast<uint64_t>(bit) * plane_bytes;
            if ((plane[byte_idx] & bit_mask) != 0U) {
                code += 1U << bit;
            }
        }
        result += vector[d] * static_cast<float>(code);
    }
    return result;
#else
    return avx::RaBitQFloatSupplementCodeIP(vector, supplement_code, dim, supplement_bits);
#endif
}

float
RaBitQFloatPackedSupplementCodeIP(const float* vector,
                                  const uint8_t* packed_supplement_code,
                                  uint64_t dim,
                                  uint32_t supplement_bits) {
#if defined(ENABLE_AVX2)
    const simd::RaBitQPackedSupplementLayout layout{dim, supplement_bits};
    if (not layout.UsesCompactChunks()) {
        return generic::RaBitQFloatPackedSupplementCodeIP(
            vector, packed_supplement_code, dim, supplement_bits);
    }

    __m256 sum = _mm256_setzero_ps();
    for (uint64_t chunk = 0; chunk < layout.FullChunkCount(); ++chunk) {
        __m128i values[4];
        DecodeCompactSupplementChunk(
            packed_supplement_code + chunk * layout.ChunkBytes(), supplement_bits, values);
        const uint64_t begin = chunk * simd::RaBitQPackedSupplementLayout::CHUNK_DIM;
        for (uint64_t group = 0; group < 4U; ++group) {
            AccumulateCompactSupplement16(values[group], vector + begin + group * 16U, sum);
        }
    }

    float result = AVX2_REDUCE_ADD_PS(sum);
    if (layout.TailDimension() != 0U) {
        result += generic::RaBitQFloatPackedSupplementCodeIP(
            vector + layout.FullDimension(),
            packed_supplement_code + layout.FullChunksSize(),
            layout.TailDimension(),
            supplement_bits);
    }
    return result;
#else
    return generic::RaBitQFloatPackedSupplementCodeIP(
        vector, packed_supplement_code, dim, supplement_bits);
#endif
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_AVX2)
    simd::DivScalarImpl<simd::SimdTraits<simd::Avx2Tag>>(from, to, dim, scalar, &avx::DivScalar);
#else
    sse::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
    float norm = std::sqrt(FP32ComputeIP(from, from, dim));
    avx2::DivScalar(from, to, dim, norm);
    return norm;
}

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_AVX2)
    if (pq_dim == 0) {
        return;
    }
    __m256i sum[4];
    __m256i result_sum[4];
    for (uint64_t i = 0; i < 4; i++) {
        sum[i] = _mm256_setzero_si256();
        result_sum[i] = _mm256_setzero_si256();
    }
    // Each 16-bit lane accumulates one lookup byte (<= 255) per iteration and
    // would overflow after ~258 of them, so periodically drain the int16
    // partials into the int32 result and reset the accumulators.
    constexpr uint64_t kFlushInterval = 256;
    uint64_t since_flush = 0;
    auto flush = [&]() {
        for (int64_t idx = 0; idx < 4; idx++) {
            const auto low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(sum[idx]));
            const auto high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(sum[idx], 1));
            result_sum[idx] = _mm256_add_epi32(result_sum[idx], _mm256_add_epi32(low, high));
            sum[idx] = _mm256_setzero_si256();
        }
    };
    const auto sign4 = _mm256_set1_epi8(0x0F);
    const auto sign8 = _mm256_set1_epi16(0xFF);
    uint64_t i = 0;
    for (; i + 1 < pq_dim; i += 2) {
        auto dict = _mm256_loadu_si256((__m256i*)(lookup_table));
        lookup_table += 32;
        auto code = _mm256_loadu_si256((__m256i*)(codes));
        codes += 32;
        auto code1 = _mm256_and_si256(code, sign4);
        auto code2 = _mm256_and_si256(_mm256_srli_epi16(code, 4), sign4);
        auto res1 = _mm256_shuffle_epi8(dict, code1);
        auto res2 = _mm256_shuffle_epi8(dict, code2);
        sum[0] = _mm256_add_epi16(sum[0], _mm256_and_si256(res1, sign8));
        sum[1] = _mm256_add_epi16(sum[1], _mm256_srli_epi16(res1, 8));
        sum[2] = _mm256_add_epi16(sum[2], _mm256_and_si256(res2, sign8));
        sum[3] = _mm256_add_epi16(sum[3], _mm256_srli_epi16(res2, 8));
        if (++since_flush == kFlushInterval) {
            flush();
            since_flush = 0;
        }
    }
    flush();
    for (uint64_t idx = 0; idx < 4; ++idx) {
        const auto previous =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(result + idx * 8));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + idx * 8),
                            _mm256_add_epi32(previous, result_sum[idx]));
    }
    if (pq_dim > i) {
        avx::PQFastScanLookUp32(lookup_table, codes, pq_dim - i, result);
    }
#else
    avx::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
PQFastScanLookUp32HighAcc(const uint8_t* RESTRICT low_lookup_table,
                          const uint8_t* RESTRICT high_lookup_table,
                          const uint8_t* RESTRICT codes,
                          uint64_t pq_dim,
                          int32_t* RESTRICT result) {
#if defined(ENABLE_AVX2)
    if (pq_dim == 0) {
        return;
    }
    constexpr uint64_t kSafeGroupCount = 256;
    const auto low_nibble_mask = _mm256_set1_epi8(0x0F);
    while (pq_dim >= 2) {
        const uint64_t group_count = (pq_dim < kSafeGroupCount ? pq_dim : kSafeGroupCount) & ~1ULL;
        __m256i sums[2][4];
        for (uint64_t byte_plane = 0; byte_plane < 2; ++byte_plane) {
            for (uint64_t result_group = 0; result_group < 4; ++result_group) {
                sums[byte_plane][result_group] = _mm256_setzero_si256();
            }
        }

        for (uint64_t group = 0; group < group_count; group += 2) {
            const auto code = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
            const auto low_codes = _mm256_and_si256(code, low_nibble_mask);
            const auto high_codes = _mm256_and_si256(_mm256_srli_epi16(code, 4), low_nibble_mask);
            const auto low_dict =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(low_lookup_table));
            const auto high_dict =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(high_lookup_table));

            auto accumulate = [&](uint64_t byte_plane, const __m256i dict) {
                const auto low_values = _mm256_shuffle_epi8(dict, low_codes);
                const auto high_values = _mm256_shuffle_epi8(dict, high_codes);
                sums[byte_plane][0] = _mm256_add_epi16(sums[byte_plane][0], low_values);
                sums[byte_plane][1] =
                    _mm256_add_epi16(sums[byte_plane][1], _mm256_srli_epi16(low_values, 8));
                sums[byte_plane][2] = _mm256_add_epi16(sums[byte_plane][2], high_values);
                sums[byte_plane][3] =
                    _mm256_add_epi16(sums[byte_plane][3], _mm256_srli_epi16(high_values, 8));
            };
            accumulate(0, low_dict);
            accumulate(1, high_dict);
            low_lookup_table += 32;
            high_lookup_table += 32;
            codes += 32;
        }

        __m256i nibble_sums[2][2];
        auto fold_groups = [](__m256i even_words, const __m256i odd_words) {
            even_words = _mm256_sub_epi16(even_words, _mm256_slli_epi16(odd_words, 8));
            return _mm256_add_epi16(_mm256_permute2f128_si256(even_words, odd_words, 0x21),
                                    _mm256_blend_epi32(even_words, odd_words, 0xF0));
        };
        for (uint64_t byte_plane = 0; byte_plane < 2; ++byte_plane) {
            nibble_sums[byte_plane][0] = fold_groups(sums[byte_plane][0], sums[byte_plane][1]);
            nibble_sums[byte_plane][1] = fold_groups(sums[byte_plane][2], sums[byte_plane][3]);
        }

        auto combine_byte_planes =
            [&](const __m256i low_bytes, const __m256i high_bytes, uint64_t result_index) {
                const auto low0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(low_bytes));
                const auto low1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(low_bytes, 1));
                const auto high0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(high_bytes));
                const auto high1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(high_bytes, 1));
                const auto combined0 = _mm256_add_epi32(low0, _mm256_slli_epi32(high0, 8));
                const auto combined1 = _mm256_add_epi32(low1, _mm256_slli_epi32(high1, 8));
                auto* result0 = result + result_index * 8;
                auto* result1 = result0 + 8;
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(result0),
                    _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(result0)),
                                     combined0));
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(result1),
                    _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(result1)),
                                     combined1));
            };
        combine_byte_planes(nibble_sums[0][0], nibble_sums[1][0], 0);
        combine_byte_planes(nibble_sums[0][1], nibble_sums[1][1], 2);
        pq_dim -= group_count;
    }
    if (pq_dim != 0) {
        avx::PQFastScanLookUp32HighAcc(low_lookup_table, high_lookup_table, codes, pq_dim, result);
    }
#else
    avx::PQFastScanLookUp32HighAcc(low_lookup_table, high_lookup_table, codes, pq_dim, result);
#endif
}
void
PQFastScanLookUp32HighAccOverwrite(const uint8_t* RESTRICT low_lookup_table,
                                   const uint8_t* RESTRICT high_lookup_table,
                                   const uint8_t* RESTRICT codes,
                                   uint64_t pq_dim,
                                   int32_t* RESTRICT result) {
#if defined(ENABLE_AVX2)
    constexpr uint64_t kSafeGroupCount = 256;
    const auto low_nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i result_sums[4]{_mm256_setzero_si256(),
                           _mm256_setzero_si256(),
                           _mm256_setzero_si256(),
                           _mm256_setzero_si256()};
    while (pq_dim >= 2) {
        const uint64_t group_count = (pq_dim < kSafeGroupCount ? pq_dim : kSafeGroupCount) & ~1ULL;
        __m256i sums[2][4];
        for (uint64_t byte_plane = 0; byte_plane < 2; ++byte_plane) {
            for (uint64_t result_group = 0; result_group < 4; ++result_group) {
                sums[byte_plane][result_group] = _mm256_setzero_si256();
            }
        }

        for (uint64_t group = 0; group < group_count; group += 2) {
            const auto code = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
            const auto low_codes = _mm256_and_si256(code, low_nibble_mask);
            const auto high_codes = _mm256_and_si256(_mm256_srli_epi16(code, 4), low_nibble_mask);
            const auto low_dict =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(low_lookup_table));
            const auto high_dict =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(high_lookup_table));

            auto accumulate = [&](uint64_t byte_plane, const __m256i dict) {
                const auto low_values = _mm256_shuffle_epi8(dict, low_codes);
                const auto high_values = _mm256_shuffle_epi8(dict, high_codes);
                sums[byte_plane][0] = _mm256_add_epi16(sums[byte_plane][0], low_values);
                sums[byte_plane][1] =
                    _mm256_add_epi16(sums[byte_plane][1], _mm256_srli_epi16(low_values, 8));
                sums[byte_plane][2] = _mm256_add_epi16(sums[byte_plane][2], high_values);
                sums[byte_plane][3] =
                    _mm256_add_epi16(sums[byte_plane][3], _mm256_srli_epi16(high_values, 8));
            };
            accumulate(0, low_dict);
            accumulate(1, high_dict);
            low_lookup_table += 32;
            high_lookup_table += 32;
            codes += 32;
        }

        __m256i nibble_sums[2][2];
        auto fold_groups = [](__m256i even_words, const __m256i odd_words) {
            even_words = _mm256_sub_epi16(even_words, _mm256_slli_epi16(odd_words, 8));
            return _mm256_add_epi16(_mm256_permute2f128_si256(even_words, odd_words, 0x21),
                                    _mm256_blend_epi32(even_words, odd_words, 0xF0));
        };
        for (uint64_t byte_plane = 0; byte_plane < 2; ++byte_plane) {
            nibble_sums[byte_plane][0] = fold_groups(sums[byte_plane][0], sums[byte_plane][1]);
            nibble_sums[byte_plane][1] = fold_groups(sums[byte_plane][2], sums[byte_plane][3]);
        }

        auto combine_byte_planes =
            [&](const __m256i low_bytes, const __m256i high_bytes, uint64_t result_index) {
                const auto low0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(low_bytes));
                const auto low1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(low_bytes, 1));
                const auto high0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(high_bytes));
                const auto high1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(high_bytes, 1));
                const auto combined0 = _mm256_add_epi32(low0, _mm256_slli_epi32(high0, 8));
                const auto combined1 = _mm256_add_epi32(low1, _mm256_slli_epi32(high1, 8));
                result_sums[result_index] = _mm256_add_epi32(result_sums[result_index], combined0);
                result_sums[result_index + 1] =
                    _mm256_add_epi32(result_sums[result_index + 1], combined1);
            };
        combine_byte_planes(nibble_sums[0][0], nibble_sums[1][0], 0);
        combine_byte_planes(nibble_sums[0][1], nibble_sums[1][1], 2);
        pq_dim -= group_count;
    }
    if (pq_dim != 0) {
        alignas(32) int32_t tail_result[32];
        avx::PQFastScanLookUp32HighAccOverwrite(
            low_lookup_table, high_lookup_table, codes, pq_dim, tail_result);
        for (uint64_t result_group = 0; result_group < 4; ++result_group) {
            result_sums[result_group] = _mm256_add_epi32(
                result_sums[result_group],
                _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(tail_result + result_group * 8)));
        }
    }
    for (uint64_t result_group = 0; result_group < 4; ++result_group) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + result_group * 8),
                            result_sums[result_group]);
    }
#else
    avx::PQFastScanLookUp32HighAccOverwrite(
        low_lookup_table, high_lookup_table, codes, pq_dim, result);
#endif
}

uint32_t
FP32LessThan32Mask(const float* values, float limit) {
#if defined(ENABLE_AVX2)
    const auto limit_vec = _mm256_set1_ps(limit);
    uint32_t mask = 0;
    for (uint64_t i = 0; i < 4; ++i) {
        const auto value = _mm256_loadu_ps(values + i * 8);
        mask |=
            static_cast<uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(value, limit_vec, _CMP_LT_OQ)))
            << (i * 8);
    }
    return mask;
#else
    return avx::FP32LessThan32Mask(values, limit);
#endif
}

uint32_t
RaBitQFastScan32ResidualPostprocess(const int32_t* accumulators,
                                    uint32_t filter_bits,
                                    float delta,
                                    float sum_vl,
                                    float query_sum,
                                    float query_norm,
                                    float query_bucket_norm_sqr,
                                    float inv_sqrt_d,
                                    const float* f_add,
                                    const float* f_scale,
                                    const float* filter_norm_codes,
                                    uint32_t valid_size,
                                    float* dists,
                                    float* filter_inner_products) {
#if defined(ENABLE_AVX2)
    if (valid_size != 32) {
        return avx::RaBitQFastScan32ResidualPostprocess(accumulators,
                                                        filter_bits,
                                                        delta,
                                                        sum_vl,
                                                        query_sum,
                                                        query_norm,
                                                        query_bucket_norm_sqr,
                                                        inv_sqrt_d,
                                                        f_add,
                                                        f_scale,
                                                        filter_norm_codes,
                                                        valid_size,
                                                        dists,
                                                        filter_inner_products);
    }
    const auto delta_v = _mm256_set1_ps(delta);
    const auto sum_vl_v = _mm256_set1_ps(sum_vl);
    const auto query_sum_v = _mm256_set1_ps(query_sum);
    const auto query_norm_v = _mm256_set1_ps(query_norm);
    const auto query_bucket_norm_v = _mm256_set1_ps(query_bucket_norm_sqr);
    const auto inv_sqrt_d_v = _mm256_set1_ps(inv_sqrt_d);
    const auto two_v = _mm256_set1_ps(2.0F);
    const auto max_v = _mm256_set1_ps(std::numeric_limits<float>::max());
    const auto epsilon_v = _mm256_set1_ps(std::numeric_limits<float>::epsilon());
    const auto nan_v = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
    const auto sign_v = _mm256_set1_ps(-0.0F);
    uint32_t computed_mask = 0;
    if (filter_bits == 1) {
        for (uint32_t offset = 0; offset < 32; offset += 8) {
            auto lookup_sum_v = _mm256_cvtepi32_ps(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulators + offset)));
            lookup_sum_v = _mm256_add_ps(_mm256_mul_ps(lookup_sum_v, delta_v), sum_vl_v);
            const auto f_add_v = _mm256_loadu_ps(f_add + offset);
            const auto f_scale_v = _mm256_loadu_ps(f_scale + offset);
            const auto abs_f_add_v = _mm256_andnot_ps(sign_v, f_add_v);
            const auto abs_f_scale_v = _mm256_andnot_ps(sign_v, f_scale_v);
            auto valid_v =
                _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(abs_f_add_v, max_v, _CMP_LE_OQ),
                                            _mm256_cmp_ps(abs_f_scale_v, max_v, _CMP_LE_OQ)),
                              _mm256_cmp_ps(abs_f_scale_v, epsilon_v, _CMP_GT_OQ));
            const auto filter_ip_v = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_mul_ps(two_v, lookup_sum_v), query_sum_v), inv_sqrt_d_v);
            auto result_v =
                _mm256_add_ps(_mm256_add_ps(f_add_v, query_bucket_norm_v),
                              _mm256_mul_ps(_mm256_mul_ps(f_scale_v, query_norm_v), filter_ip_v));
            const auto abs_result_v = _mm256_andnot_ps(sign_v, result_v);
            valid_v = _mm256_and_ps(valid_v, _mm256_cmp_ps(abs_result_v, max_v, _CMP_LE_OQ));
            result_v = _mm256_blendv_ps(max_v, result_v, valid_v);
            _mm256_storeu_ps(dists + offset, result_v);
            if (filter_inner_products != nullptr) {
                _mm256_storeu_ps(filter_inner_products + offset,
                                 _mm256_blendv_ps(nan_v, filter_ip_v, valid_v));
            }
            computed_mask |= static_cast<uint32_t>(_mm256_movemask_ps(valid_v)) << offset;
        }
        return computed_mask;
    }

    const float filter_center = 0.5F * static_cast<float>((1U << filter_bits) - 1U);
    const auto center_query_sum_v = _mm256_set1_ps(filter_center * query_sum);
    for (uint32_t offset = 0; offset < 32; offset += 8) {
        auto lookup_sum_v = _mm256_setzero_ps();
        for (uint32_t plane = 0; plane < filter_bits; ++plane) {
            const float weight = static_cast<float>(1U << (filter_bits - plane - 1));
            auto accumulator_v = _mm256_cvtepi32_ps(_mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(accumulators + plane * 32 + offset)));
            accumulator_v = _mm256_add_ps(_mm256_mul_ps(accumulator_v, delta_v), sum_vl_v);
            lookup_sum_v =
                _mm256_add_ps(lookup_sum_v, _mm256_mul_ps(accumulator_v, _mm256_set1_ps(weight)));
        }
        lookup_sum_v = _mm256_sub_ps(lookup_sum_v, center_query_sum_v);
        const auto f_add_v = _mm256_loadu_ps(f_add + offset);
        const auto f_scale_v = _mm256_loadu_ps(f_scale + offset);
        const auto abs_f_add_v = _mm256_andnot_ps(sign_v, f_add_v);
        const auto abs_f_scale_v = _mm256_andnot_ps(sign_v, f_scale_v);
        __m256 valid_v =
            _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(abs_f_add_v, max_v, _CMP_LE_OQ),
                                        _mm256_cmp_ps(abs_f_scale_v, max_v, _CMP_LE_OQ)),
                          _mm256_cmp_ps(abs_f_scale_v, epsilon_v, _CMP_GT_OQ));
        const auto norm_code_v = _mm256_loadu_ps(filter_norm_codes + offset);
        valid_v =
            _mm256_and_ps(valid_v, _mm256_cmp_ps(norm_code_v, _mm256_setzero_ps(), _CMP_GT_OQ));
        const auto filter_ip_v = _mm256_div_ps(lookup_sum_v, norm_code_v);
        auto result_v =
            _mm256_add_ps(_mm256_add_ps(f_add_v, query_bucket_norm_v),
                          _mm256_mul_ps(_mm256_mul_ps(f_scale_v, query_norm_v), filter_ip_v));
        const auto abs_result_v = _mm256_andnot_ps(sign_v, result_v);
        valid_v = _mm256_and_ps(valid_v, _mm256_cmp_ps(abs_result_v, max_v, _CMP_LE_OQ));
        result_v = _mm256_blendv_ps(max_v, result_v, valid_v);
        _mm256_storeu_ps(dists + offset, result_v);
        if (filter_inner_products != nullptr) {
            _mm256_storeu_ps(filter_inner_products + offset,
                             _mm256_blendv_ps(nan_v, filter_ip_v, valid_v));
        }
        computed_mask |= static_cast<uint32_t>(_mm256_movemask_ps(valid_v)) << offset;
    }
    return computed_mask;
#else
    return avx::RaBitQFastScan32ResidualPostprocess(accumulators,
                                                    filter_bits,
                                                    delta,
                                                    sum_vl,
                                                    query_sum,
                                                    query_norm,
                                                    query_bucket_norm_sqr,
                                                    inv_sqrt_d,
                                                    f_add,
                                                    f_scale,
                                                    filter_norm_codes,
                                                    valid_size,
                                                    dists,
                                                    filter_inner_products);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX2)
    simd::BitAndImpl<simd::BitTraits<simd::Avx2BitTag>>(x, y, num_byte, result, &sse::BitAnd);
#else
    return sse::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX2)
    simd::BitOrImpl<simd::BitTraits<simd::Avx2BitTag>>(x, y, num_byte, result, &sse::BitOr);
#else
    return sse::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX2)
    simd::BitXorImpl<simd::BitTraits<simd::Avx2BitTag>>(x, y, num_byte, result, &sse::BitXor);
#else
    return sse::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX2)
    simd::BitNotImpl<simd::BitTraits<simd::Avx2BitTag>>(x, num_byte, result, &sse::BitNot);
#else
    return sse::BitNot(x, num_byte, result);
#endif
}

void
VecRescale(float* data, uint64_t dim, float val) {
#if defined(ENABLE_AVX2)
    simd::VecRescaleImpl<simd::SimdTraits<simd::Avx2Tag>>(data, dim, val, &sse::VecRescale);
#else
    sse::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_AVX2)
    simd::RotateOpImpl<simd::SimdTraits<simd::Avx2Tag>>(data, idx, dim_, step);
#else
    avx::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, uint64_t dim_) {
#if defined(ENABLE_AVX2)
    uint64_t n = dim_;
    uint64_t step = 1;
    while (step < n) {
        if (step >= 8) {
            avx2::RotateOp(data, 0, dim_, step);
        } else if (step == 4) {
            sse::RotateOp(data, 0, dim_, step);
        } else {
            generic::RotateOp(data, 0, dim_, step);
        }
        step *= 2;
    }
#else
    return avx::FHTRotate(data, dim_);
#endif
}

void
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_AVX2)
    simd::KacsWalkImpl<simd::SimdTraits<simd::Avx2Tag>>(data, len, &avx::KacsWalk);
#else
    avx::KacsWalk(data, len);
#endif
}

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
#if defined(ENABLE_AVX2)
    return simd::NormalizeWithCentroidImpl<simd::SimdTraits<simd::Avx2Tag>>(
        from, centroid, to, dim, &avx::NormalizeWithCentroid);
#else
    return sse::NormalizeWithCentroid(from, centroid, to, dim);
#endif
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
#if defined(ENABLE_AVX2)
    simd::InverseNormalizeWithCentroidImpl<simd::SimdTraits<simd::Avx2Tag>>(
        from, centroid, to, dim, norm, &avx::InverseNormalizeWithCentroid);
#else
    sse::InverseNormalizeWithCentroid(from, centroid, to, dim, norm);
#endif
}

}  // namespace vsag::avx2
