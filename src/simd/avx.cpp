
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

#if defined(ENABLE_AVX)
#include <immintrin.h>

#include "simd/kernels/kernels.h"
#include "simd/traits/simd_traits_avx.h"

inline float
avx_reduce_add_ps(__m256 a) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, a);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}
#endif

#include <cmath>
#include <cstdint>

#include "simd.h"

namespace vsag::avx {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0F - avx::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx::INT8ComputeL2Sqr(pVect1, pVect2, qty);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx::INT8ComputeIP(pVect1, pVect2, qty);
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -avx::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_AVX)
    simd::PQDistanceFloat256Impl<simd::SimdTraits<simd::AVX_Tag>>(
        single_dim_centers, single_dim_val, result, &sse::PQDistanceFloat256);
#else
    return sse::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

void
Prefetch(const void* data) {
    sse::Prefetch(data);
}

#if defined(ENABLE_AVX)
__inline __m256i __attribute__((__always_inline__)) load_8_char_and_convert(const uint8_t* data) {
    __m128i first_8 =
        _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, data[3], data[2], data[1], data[0]);
    __m128i second_8 =
        _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, data[7], data[6], data[5], data[4]);
    __m128i first_32 = _mm_cvtepu8_epi32(first_8);
    __m128i second_32 = _mm_cvtepu8_epi32(second_8);
    return _mm256_set_m128i(second_32, first_32);
}
#endif

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::ComputeIPImpl<simd::SimdTraits<simd::AVX_Tag>>(
        query, codes, dim, &sse::FP32ComputeIP);
#else
    return sse::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::AVX_Tag>>(
        query, codes, dim, &sse::FP32ComputeL2Sqr);
#else
    return sse::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP32SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const float* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
#if defined(ENABLE_AVX)
    __m256 q_vec = _mm256_set1_ps(query_val);
    uint32_t i = 0;
    for (; i + 8 <= num; i += 8) {
        __m256 val_vec = _mm256_loadu_ps(vals + i);
        __m256 delta_vec = _mm256_mul_ps(val_vec, q_vec);

        alignas(32) float res[8];
        _mm256_store_ps(res, delta_vec);

        for (int k = 0; k < 8; ++k) {
            dists[ids[i + k]] += res[k];
        }
    }
    for (; i < num; ++i) {
        dists[ids[i]] += vals[i] * query_val;
    }
#else
    return sse::FP32SparseAccumulate(dists, ids, vals, query_val, num);
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
#if defined(ENABLE_AVX)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::AVX_Tag>, simd::Batch4Kind::IP>(
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
        &sse::FP32ComputeIPBatch4);
#else
    return sse::FP32ComputeIPBatch4(
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
#if defined(ENABLE_AVX)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::AVX_Tag>, simd::Batch4Kind::L2>(
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
        &sse::FP32ComputeL2SqrBatch4);
#else
    return sse::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX)
    simd::BinaryOpImpl<simd::SimdTraits<simd::AVX_Tag>, simd::BinaryOp::Sub>(
        x, y, z, dim, &sse::FP32Sub);
#else
    sse::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX)
    simd::BinaryOpImpl<simd::SimdTraits<simd::AVX_Tag>, simd::BinaryOp::Add>(
        x, y, z, dim, &sse::FP32Add);
#else
    sse::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX)
    simd::BinaryOpImpl<simd::SimdTraits<simd::AVX_Tag>, simd::BinaryOp::Mul>(
        x, y, z, dim, &sse::FP32Mul);
#else
    sse::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX)
    simd::BinaryOpImpl<simd::SimdTraits<simd::AVX_Tag>, simd::BinaryOp::Div>(
        x, y, z, dim, &sse::FP32Div);
#else
    sse::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
    return sse::FP32ReduceAdd(x, dim);
}

#if defined(ENABLE_AVX)
__inline __m256i __attribute__((__always_inline__)) load_8_short(const uint16_t* data) {
    return _mm256_set_epi16(data[7],
                            0,
                            data[6],
                            0,
                            data[5],
                            0,
                            data[4],
                            0,
                            data[3],
                            0,
                            data[2],
                            0,
                            data[1],
                            0,
                            data[0],
                            0);
}
#endif

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::HalfComputeIPImpl<simd::BF16Traits<simd::AVX_BF16_Tag>>(
        query, codes, dim, &sse::BF16ComputeIP);
#else
    return sse::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::HalfComputeL2SqrImpl<simd::BF16Traits<simd::AVX_BF16_Tag>>(
        query, codes, dim, &sse::BF16ComputeL2Sqr);
#else
    return sse::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::HalfComputeIPImpl<simd::FP16Traits<simd::AVX_FP16_Tag>>(
        query, codes, dim, &sse::FP16ComputeIP);
#else
    return sse::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::HalfComputeL2SqrImpl<simd::FP16Traits<simd::AVX_FP16_Tag>>(
        query, codes, dim, &sse::FP16ComputeL2Sqr);
#else
    return sse::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP16SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const uint16_t* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
#if defined(ENABLE_AVX)
    __m256 q_vec = _mm256_set1_ps(query_val);
    uint32_t i = 0;
    for (; i + 8 <= num; i += 8) {
        __m128i val_half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vals + i));
        __m256 val_vec = _mm256_cvtph_ps(val_half);
        __m256 delta_vec = _mm256_mul_ps(val_vec, q_vec);

        alignas(32) float res[8];
        _mm256_store_ps(res, delta_vec);

        for (int k = 0; k < 8; ++k) {
            dists[ids[i + k]] += res[k];
        }
    }
    for (; i < num; ++i) {
        dists[ids[i]] += generic::FP16ToFloat(vals[i]) * query_val;
    }
#else
    return sse::FP16SparseAccumulate(dists, ids, vals, query_val, num);
#endif
}

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX)
    // TODO: impl based on AVX
    return sse::INT8ComputeL2Sqr(query, codes, dim);
#else
    return sse::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
INT8ComputeIP(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX)
    return sse::INT8ComputeIP(query, codes, dim);
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
#if defined(ENABLE_AVX)
    return simd::SQ8ComputeIPImpl<simd::SQ8Traits<simd::AVX_SQ8_Tag>>(
        query, codes, lower_bound, diff, dim, &sse::SQ8ComputeIP);
#else
    return sse::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::SQ8ComputeL2SqrImpl<simd::SQ8Traits<simd::AVX_SQ8_Tag>>(
        query, codes, lower_bound, diff, dim, &sse::SQ8ComputeL2Sqr);
#else
    return sse::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::SQ8ComputeCodesIPImpl<simd::SQ8Traits<simd::AVX_SQ8_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &sse::SQ8ComputeCodesIP);
#else
    return sse::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::SQ8ComputeCodesL2SqrImpl<simd::SQ8Traits<simd::AVX_SQ8_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &sse::SQ8ComputeCodesL2Sqr);
#else
    return sse::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

void
SQ8SparseAccumulate(float* RESTRICT dists,
                    const uint16_t* RESTRICT ids,
                    const uint8_t* RESTRICT vals,
                    float query_val,
                    uint32_t num) {
#if defined(ENABLE_AVX)
    __m256 q_vec = _mm256_set1_ps(query_val);
    uint32_t i = 0;
    for (; i + 8 <= num; i += 8) {
        __m256i val_i32 = load_8_char_and_convert(vals + i);
        __m256 val_vec = _mm256_cvtepi32_ps(val_i32);
        __m256 delta_vec = _mm256_mul_ps(val_vec, q_vec);

        alignas(32) float res[8];
        _mm256_store_ps(res, delta_vec);

        for (int k = 0; k < 8; ++k) {
            dists[ids[i + k]] += res[k];
        }
    }
    for (; i < num; ++i) {
        dists[ids[i]] += static_cast<float>(vals[i]) * query_val;
    }
#else
    return sse::SQ8SparseAccumulate(dists, ids, vals, query_val, num);
#endif
}

#if defined(ENABLE_AVX)
__inline __m128i __attribute__((__always_inline__)) load_4_char(const uint8_t* data) {
    return _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, data[3], data[2], data[1], data[0]);
}
#endif

#if defined(ENABLE_AVX)
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

    values01 = _mm256_add_ps(_mm256_mul_ps(values01, diff_vec0), lb_vec0);
    values23 = _mm256_add_ps(_mm256_mul_ps(values23, diff_vec1), lb_vec1);
}
#endif

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::SQ4ComputeIPImpl<simd::SQ4Traits<simd::AVX_SQ4_Tag>>(
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
#if defined(ENABLE_AVX)
    return simd::SQ4ComputeL2SqrImpl<simd::SQ4Traits<simd::AVX_SQ4_Tag>>(
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
#if defined(ENABLE_AVX)
    return simd::SQ4ComputeCodesIPImpl<simd::SQ4Traits<simd::AVX_SQ4_Tag>>(
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
#if defined(ENABLE_AVX)
    return simd::SQ4ComputeCodesL2SqrImpl<simd::SQ4Traits<simd::AVX_SQ4_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &sse::SQ4ComputeCodesL2Sqr);
#else
    return sse::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX)
    return sse::SQ4UniformComputeCodesIP(codes1, codes2, dim);  // TODO(LHT): implement
#else
    return sse::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX)
    return sse::SQ8UniformComputeCodesIP(codes1, codes2, dim);  // TODO(LHT): implement
#else
    return sse::SQ8UniformComputeCodesIP(codes1, codes2, dim);
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
        out[i] = avx::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_AVX)
    return simd::RaBitQFloatBinaryIPImpl<simd::RaBitQTraits<simd::AVX_RaBitQ_Tag>>(
        vector, bits, dim, inv_sqrt_d, &sse::RaBitQFloatBinaryIP);
#else
    return sse::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
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
#if defined(ENABLE_AVX)
    simd::RaBitQFloatBinaryIPBatch4Impl<simd::RaBitQTraits<simd::AVX_RaBitQ_Tag>>(
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
    sse::RaBitQFloatBinaryIPBatch4(vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
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
    generic::RaBitQFloatThreeBitIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, reorder_bits, results);
}

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits) {
#if defined(ENABLE_AVX)
    return simd::RaBitQFloatSplitCodeIPImpl<simd::RaBitQTraits<simd::AVX_RaBitQ_Tag>>(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
#else
    return sse::RaBitQFloatSplitCodeIP(vector, one_bit_code, supplement_code, dim, supplement_bits);
#endif
}

float
RaBitQFloatSupplementCodeIP(const float* vector,
                            const uint8_t* supplement_code,
                            uint64_t dim,
                            uint32_t supplement_bits) {
    return generic::RaBitQFloatSupplementCodeIP(vector, supplement_code, dim, supplement_bits);
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_AVX)
    simd::DivScalarImpl<simd::SimdTraits<simd::AVX_Tag>>(from, to, dim, scalar, &sse::DivScalar);
#else
    sse::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
    float norm = std::sqrt(FP32ComputeIP(from, from, dim));
    avx::DivScalar(from, to, dim, norm);
    return norm;
}

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_AVX)
    sse::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#else
    sse::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX)
    simd::BitAndImpl<simd::BitTraits<simd::AVX_Bit_Tag>>(x, y, num_byte, result, &sse::BitAnd);
#else
    return sse::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX)
    simd::BitOrImpl<simd::BitTraits<simd::AVX_Bit_Tag>>(x, y, num_byte, result, &sse::BitOr);
#else
    return sse::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX)
    simd::BitXorImpl<simd::BitTraits<simd::AVX_Bit_Tag>>(x, y, num_byte, result, &sse::BitXor);
#else
    return sse::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX)
    simd::BitNotImpl<simd::BitTraits<simd::AVX_Bit_Tag>>(x, num_byte, result, &sse::BitNot);
#else
    return sse::BitNot(x, num_byte, result);
#endif
}
void
VecRescale(float* data, uint64_t dim, float val) {
#if defined(ENABLE_AVX)
    simd::VecRescaleImpl<simd::SimdTraits<simd::AVX_Tag>>(data, dim, val, &sse::VecRescale);
#else
    sse::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_AVX)
    simd::RotateOpImpl<simd::SimdTraits<simd::AVX_Tag>>(data, idx, dim_, step);
#else
    sse::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, uint64_t dim_) {
#if defined(ENABLE_AVX)
    uint64_t n = dim_;
    uint64_t step = 1;
    while (step < n) {
        if (step >= 8) {
            avx::RotateOp(data, 0, dim_, step);
        } else if (step == 4) {
            sse::RotateOp(data, 0, dim_, step);
        } else {
            generic::RotateOp(data, 0, dim_, step);
        }
        step *= 2;
    }
#else
    return sse::FHTRotate(data, dim_);
#endif
}

void
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_AVX)
    simd::KacsWalkImpl<simd::SimdTraits<simd::AVX_Tag>>(data, len, &sse::KacsWalk);
#else
    sse::KacsWalk(data, len);
#endif
}

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
#if defined(ENABLE_AVX)
    return simd::NormalizeWithCentroidImpl<simd::SimdTraits<simd::AVX_Tag>>(
        from, centroid, to, dim, &sse::NormalizeWithCentroid);
#else
    return sse::NormalizeWithCentroid(from, centroid, to, dim);
#endif
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
#if defined(ENABLE_AVX)
    simd::InverseNormalizeWithCentroidImpl<simd::SimdTraits<simd::AVX_Tag>>(
        from, centroid, to, dim, norm, &sse::InverseNormalizeWithCentroid);
#else
    sse::InverseNormalizeWithCentroid(from, centroid, to, dim, norm);
#endif
}

}  // namespace vsag::avx
