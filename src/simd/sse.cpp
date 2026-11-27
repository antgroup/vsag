
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

#include <cstdint>
#include <cstring>

#include "simd/int8_simd.h"
#if defined(ENABLE_SSE)
#include <x86intrin.h>

#include "simd/kernels/kernels.h"
#include "simd/kernels/rabitq_binary.h"
#include "simd/traits/simd_traits_sse.h"
#endif

#include <cmath>

#include "kernels/local_sq_compute.h"
#include "kernels/rabitq_residual4.h"
#include "kernels/rabitq_scalar_compute.h"
#include "kernels/rabitq_small_bitplanes.h"
#include "simd.h"
#include "sq4_local_simd.h"
#include "sq8_local_simd.h"

#if defined(ENABLE_SSE)
#include "simd/kernels/rabitq_pack_128.h"
#endif

namespace vsag::sse {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sse::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sse::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - sse::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sse::INT8ComputeL2Sqr(pVect1, pVect2, qty);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sse::INT8ComputeIP(pVect1, pVect2, qty);
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -sse::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_SSE)
    simd::PQDistanceFloat256Impl<simd::SimdTraits<simd::SseTag>>(
        single_dim_centers, single_dim_val, result, &generic::PQDistanceFloat256);
#else
    return generic::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

#if defined(ENABLE_SSE)
__inline __m128i __attribute__((__always_inline__))
load_4_char(const uint8_t* data) {
    return _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, data[3], data[2], data[1], data[0]);
}
#endif

#if defined(ENABLE_SSE)
__inline __m128i __attribute__((__always_inline__))
load_4_short(const uint16_t* data) {
    return _mm_set_epi16(data[3], 0, data[2], 0, data[1], 0, data[0], 0);
}
#endif

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::ComputeIPImpl<simd::SimdTraits<simd::SseTag>>(
        query, codes, dim, &generic::FP32ComputeIP);
#else
    return vsag::generic::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::SseTag>>(
        query, codes, dim, &generic::FP32ComputeL2Sqr);
#else
    return vsag::generic::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP32SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const float* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
#if defined(ENABLE_SSE)
    __m128 q_vec = _mm_set1_ps(query_val);
    uint32_t i = 0;
    for (; i + 4 <= num; i += 4) {
        __m128 val_vec = _mm_loadu_ps(vals + i);
        __m128 delta_vec = _mm_mul_ps(val_vec, q_vec);

        alignas(16) float res[4];
        _mm_store_ps(res, delta_vec);

        for (int k = 0; k < 4; ++k) {
            dists[ids[i + k]] += res[k];
        }
    }
    for (; i < num; ++i) {
        dists[ids[i]] += vals[i] * query_val;
    }
#else
    return generic::FP32SparseAccumulate(dists, ids, vals, query_val, num);
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
#if defined(ENABLE_SSE)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::SseTag>, simd::Batch4Kind::IP>(
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
        &generic::FP32ComputeIPBatch4);
#else
    return generic::FP32ComputeIPBatch4(
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
#if defined(ENABLE_SSE)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::SseTag>, simd::Batch4Kind::L2>(
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
        &generic::FP32ComputeL2SqrBatch4);
#else
    return generic::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SSE)
    simd::BinaryOpImpl<simd::SimdTraits<simd::SseTag>, simd::BinaryOp::Sub>(
        x, y, z, dim, &generic::FP32Sub);
#else
    return generic::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SSE)
    simd::BinaryOpImpl<simd::SimdTraits<simd::SseTag>, simd::BinaryOp::Add>(
        x, y, z, dim, &generic::FP32Add);
#else
    return generic::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SSE)
    simd::BinaryOpImpl<simd::SimdTraits<simd::SseTag>, simd::BinaryOp::Mul>(
        x, y, z, dim, &generic::FP32Mul);
#else
    return generic::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SSE)
    simd::BinaryOpImpl<simd::SimdTraits<simd::SseTag>, simd::BinaryOp::Div>(
        x, y, z, dim, &generic::FP32Div);
#else
    return generic::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::ReduceAddImpl<simd::SimdTraits<simd::SseTag>>(x, dim, &generic::FP32ReduceAdd);
#else
    return generic::FP32ReduceAdd(x, dim);
#endif
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::HalfComputeIPImpl<simd::BF16Traits<simd::SseBF16Tag>>(
        query, codes, dim, &generic::BF16ComputeIP);
#else
    return generic::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::HalfComputeL2SqrImpl<simd::BF16Traits<simd::SseBF16Tag>>(
        query, codes, dim, &generic::BF16ComputeL2Sqr);
#else
    return generic::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::HalfComputeIPImpl<simd::FP16Traits<simd::SseFP16Tag>>(
        query, codes, dim, &generic::FP16ComputeIP);
#else
    return generic::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::HalfComputeL2SqrImpl<simd::FP16Traits<simd::SseFP16Tag>>(
        query, codes, dim, &generic::FP16ComputeL2Sqr);
#else
    return generic::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP16SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const uint16_t* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
#if defined(ENABLE_SSE)
    constexpr uint32_t kWidth = 4;
    const __m128 query = _mm_set1_ps(query_val);
    const __m128i zero = _mm_setzero_si128();
    const __m128i sign_mask = _mm_set1_epi32(0x8000);
    const __m128i magnitude_mask = _mm_set1_epi32(0x7fff);
    const __m128i bias = _mm_set1_epi32(112U << 23);
    uint32_t i = 0;
    for (; num - i >= kWidth; i += kWidth) {
        const __m128i half =
            _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(vals + i)), zero);
        // Match generic::FP16ToFloat without requiring F16C.
        const __m128i sign = _mm_slli_epi32(_mm_and_si128(half, sign_mask), 16);
        const __m128i magnitude =
            _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(half, magnitude_mask), 13), bias);
        const __m128 values = _mm_castsi128_ps(_mm_or_si128(sign, magnitude));
        alignas(16) float deltas[kWidth];
        _mm_store_ps(deltas, _mm_mul_ps(values, query));
        for (uint32_t lane = 0; lane < kWidth; ++lane) {
            dists[ids[i + lane]] += deltas[lane];
        }
    }
    if (i < num) {
        generic::FP16SparseAccumulate(dists, ids + i, vals + i, query_val, num - i);
    }
#else
    generic::FP16SparseAccumulate(dists, ids, vals, query_val, num);
#endif
}

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::Int8ComputeL2SqrImpl<simd::Int8Traits<simd::SseInt8Tag>>(
        query, codes, dim, &generic::INT8ComputeL2Sqr);
#else
    return generic::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::Int8ComputeIPImpl<simd::Int8Traits<simd::SseInt8Tag>>(
        query, codes, dim, &generic::INT8ComputeIP);
#else
    return generic::INT8ComputeIP(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ8ComputeIPImpl<simd::SQ8Traits<simd::SseSQ8Tag>>(
        query, codes, lower_bound, diff, dim, &generic::SQ8ComputeIP);
#else
    return generic::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ8ComputeL2SqrImpl<simd::SQ8Traits<simd::SseSQ8Tag>>(
        query, codes, lower_bound, diff, dim, &generic::SQ8ComputeL2Sqr);
#else
    return generic::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

void
SQ8ComputeL2SqrBatch4(const float* query,
                      const uint8_t* codes1,
                      const uint8_t* codes2,
                      const uint8_t* codes3,
                      const uint8_t* codes4,
                      const float* lower_bound,
                      const float* diff,
                      uint64_t dim,
                      float& dist1,
                      float& dist2,
                      float& dist3,
                      float& dist4) {
#if defined(ENABLE_SSE)
    simd::SQ8ComputeL2SqrBatch4Impl<simd::SQ8Traits<simd::SseSQ8Tag>>(query,
                                                                      codes1,
                                                                      codes2,
                                                                      codes3,
                                                                      codes4,
                                                                      lower_bound,
                                                                      diff,
                                                                      dim,
                                                                      dist1,
                                                                      dist2,
                                                                      dist3,
                                                                      dist4,
                                                                      &generic::SQ8ComputeL2Sqr);
#else
    vsag::SQ8ComputeL2SqrBatch4Fallback(
        query, codes1, codes2, codes3, codes4, lower_bound, diff, dim, dist1, dist2, dist3, dist4);
#endif
}

void
SQ8ComputeIPBatch4(const float* query,
                   const uint8_t* codes1,
                   const uint8_t* codes2,
                   const uint8_t* codes3,
                   const uint8_t* codes4,
                   const float* lower_bound,
                   const float* diff,
                   uint64_t dim,
                   float& dist1,
                   float& dist2,
                   float& dist3,
                   float& dist4) {
#if defined(ENABLE_SSE)
    simd::SQ8ComputeIPBatch4Impl<simd::SQ8Traits<simd::SseSQ8Tag>>(query,
                                                                   codes1,
                                                                   codes2,
                                                                   codes3,
                                                                   codes4,
                                                                   lower_bound,
                                                                   diff,
                                                                   dim,
                                                                   dist1,
                                                                   dist2,
                                                                   dist3,
                                                                   dist4,
                                                                   &generic::SQ8ComputeIP);
#else
    vsag::SQ8ComputeIPBatch4Fallback(
        query, codes1, codes2, codes3, codes4, lower_bound, diff, dim, dist1, dist2, dist3, dist4);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ8ComputeCodesIPImpl<simd::SQ8Traits<simd::SseSQ8Tag>>(
        codes1, codes2, lower_bound, diff, dim, &generic::SQ8ComputeCodesIP);
#else
    return generic::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ8ComputeCodesL2SqrImpl<simd::SQ8Traits<simd::SseSQ8Tag>>(
        codes1, codes2, lower_bound, diff, dim, &generic::SQ8ComputeCodesL2Sqr);
#else
    return generic::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

void
SQ8SparseAccumulate(float* RESTRICT dists,
                    const uint16_t* RESTRICT ids,
                    const uint8_t* RESTRICT vals,
                    float query_val,
                    uint32_t num) {
#if defined(ENABLE_SSE)
    __m128 q_vec = _mm_set1_ps(query_val);
    uint32_t i = 0;
    for (; i + 4 <= num; i += 4) {
        __m128 val_vec = _mm_set_ps(static_cast<float>(vals[i + 3]),
                                    static_cast<float>(vals[i + 2]),
                                    static_cast<float>(vals[i + 1]),
                                    static_cast<float>(vals[i]));
        __m128 delta_vec = _mm_mul_ps(val_vec, q_vec);

        alignas(16) float res[4];
        _mm_store_ps(res, delta_vec);

        for (int k = 0; k < 4; ++k) {
            dists[ids[i + k]] += res[k];
        }
    }
    for (; i < num; ++i) {
        dists[ids[i]] += static_cast<float>(vals[i]) * query_val;
    }
#else
    return generic::SQ8SparseAccumulate(dists, ids, vals, query_val, num);
#endif
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ4ComputeIPImpl<simd::SQ4Traits<simd::SseSQ4Tag>>(
        query, codes, lower_bound, diff, dim, &generic::SQ4ComputeIP);
#else
    return generic::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ4ComputeL2SqrImpl<simd::SQ4Traits<simd::SseSQ4Tag>>(
        query, codes, lower_bound, diff, dim, &generic::SQ4ComputeL2Sqr);
#else
    return generic::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ4ComputeCodesIPImpl<simd::SQ4Traits<simd::SseSQ4Tag>>(
        codes1, codes2, lower_bound, diff, dim, &generic::SQ4ComputeCodesIP);
#else
    return generic::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ4ComputeCodesL2SqrImpl<simd::SQ4Traits<simd::SseSQ4Tag>>(
        codes1, codes2, lower_bound, diff, dim, &generic::SQ4ComputeCodesL2Sqr);
#else
    return generic::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ4UniformComputeCodesIPImpl<simd::UniformCodeTraits<simd::SseUniformTag>>(
        codes1, codes2, dim, &generic::SQ4UniformComputeCodesIP);
#else
    return generic::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ8UniformComputeCodesIPImpl<simd::UniformCodeTraits<simd::SseUniformTag>>(
        codes1, codes2, dim, &generic::SQ8UniformComputeCodesIP);
#else
    return generic::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

void
SQ8UniformComputeCodesIPBatch(const uint8_t* RESTRICT query,
                              const uint8_t* RESTRICT codes,
                              uint64_t dim,
                              uint64_t n_codes,
                              uint64_t code_stride,
                              float* RESTRICT out) {
#if defined(ENABLE_SSE)
    simd::SQ8UniformComputeCodesIPBatchImpl<simd::UniformCodeTraits<simd::SseUniformTag>>(
        query, codes, dim, n_codes, code_stride, out, &sse::SQ8UniformComputeCodesIP);
#else
    for (uint64_t i = 0; i < n_codes; ++i) {
        out[i] = sse::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
#endif
}

#if defined(ENABLE_SSE)
namespace {
struct SseBinaryTraits : simd::SimdTraits<simd::SseTag> {
    static FloatVec
    select_nibble(uint8_t byte, FloatVec pos, FloatVec neg) {
        const auto bits = _mm_setr_epi32(1, 2, 4, 8);
        const auto mask =
            _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32(byte), bits), bits));
        return _mm_or_ps(_mm_and_ps(mask, pos), _mm_andnot_ps(mask, neg));
    }

    using ByteVec = __m128i;
    using CountVec = __m128i;
    static constexpr uint64_t ByteWidth = 16;

    static CountVec
    zero_counts() {
        return _mm_setzero_si128();
    }
    static ByteVec
    load_bytes(const uint8_t* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    static ByteVec
    bit_and(ByteVec a, ByteVec b) {
        return _mm_and_si128(a, b);
    }
    static CountVec
    accumulate(CountVec sum, ByteVec value) {
        // Per-byte SWAR popcount using only SSE2 (no POPCNT or SSSE3).
        const auto m1 = _mm_set1_epi8(0x55);
        const auto m2 = _mm_set1_epi8(0x33);
        const auto m4 = _mm_set1_epi8(0x0F);
        value = _mm_sub_epi8(value, _mm_and_si128(_mm_srli_epi16(value, 1), m1));
        value = _mm_add_epi8(_mm_and_si128(value, m2), _mm_and_si128(_mm_srli_epi16(value, 2), m2));
        value = _mm_and_si128(_mm_add_epi8(value, _mm_srli_epi16(value, 4)), m4);
        return _mm_add_epi64(sum, _mm_sad_epu8(value, _mm_setzero_si128()));
    }
    static uint32_t
    reduce_counts(CountVec sum) {
        return static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_add_epi64(sum, _mm_srli_si128(sum, 8))));
    }
};
}  // namespace
#endif

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SSE)
    uint64_t result;
    simd::RaBitQSQ4UBinary<SseBinaryTraits, 1, false>(codes, {bits}, dim, &result);
    return static_cast<uint32_t>(result);
#else
    return generic::RaBitQSQ4UBinaryIP(codes, bits, dim);
#endif
}

uint64_t
RaBitQSQ4UBinaryIPWithBaseSum(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SSE)
    uint64_t result;
    simd::RaBitQSQ4UBinary<SseBinaryTraits, 1, true>(codes, {bits}, dim, &result);
    return result;
#else
    return generic::RaBitQSQ4UBinaryIPWithBaseSum(codes, bits, dim);
#endif
}

void
RaBitQSQ4UBinaryIPWithBaseSumBatch4(const uint8_t* codes,
                                    const uint8_t* bits1,
                                    const uint8_t* bits2,
                                    const uint8_t* bits3,
                                    const uint8_t* bits4,
                                    uint64_t dim,
                                    uint64_t* results) {
#if defined(ENABLE_SSE)
    simd::RaBitQSQ4UBinary<SseBinaryTraits, 4, true>(
        codes, {bits1, bits2, bits3, bits4}, dim, results);
#else
    generic::RaBitQSQ4UBinaryIPWithBaseSumBatch4(codes, bits1, bits2, bits3, bits4, dim, results);
#endif
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_SSE)
    float result;
    simd::RaBitQBinaryIP4Lane<SseBinaryTraits>(vector, {bits}, dim, inv_sqrt_d, &result);
    return result;
#else
    return generic::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
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
#if defined(ENABLE_SSE)
    simd::RaBitQBinaryIP4Lane<SseBinaryTraits>(
        vector, {bits1, bits2, bits3, bits4}, dim, inv_sqrt_d, results);
#else
    generic::RaBitQFloatBinaryIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
#endif
}

#if defined(ENABLE_SSE)
namespace {
struct SmallBitplaneTraits : simd::SimdTraits<simd::SseTag> {
    static FloatVec
    SelectBits(uint8_t bits, FloatVec value) {
        const auto masks = _mm_setr_epi32(1, 2, 4, 8);
        return _mm_and_ps(
            _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32(bits), masks), masks)),
            value);
    }
};
}  // namespace
#endif

float
RaBitQFloatTwoBitCenteredIP(const float* vector, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SSE)
    float results[1];
    const uint8_t* codes[] = {bits};
    simd::ComputeSmallBitplaneIP<SmallBitplaneTraits, 2, true>(vector, codes, dim, 0, results);
    return results[0];
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
#if defined(ENABLE_SSE)

    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};
    simd::ComputeSmallBitplaneIP<SmallBitplaneTraits, 2, true>(vector, codes, dim, 0, results);

#else
    generic::RaBitQFloatTwoBitCenteredIPBatch4(vector, bits1, bits2, bits3, bits4, dim, results);
#endif
}

float
RaBitQFloatThreeBitCenteredIP(const float* vector, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SSE)
    float results[1];
    const uint8_t* codes[] = {bits};
    simd::ComputeSmallBitplaneIP<SmallBitplaneTraits, 3, true>(vector, codes, dim, 0, results);
    return results[0];
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
#if defined(ENABLE_SSE)

    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};
    simd::ComputeSmallBitplaneIP<SmallBitplaneTraits, 3, true>(vector, codes, dim, 0, results);

#else
    generic::RaBitQFloatThreeBitCenteredIPBatch4(vector, bits1, bits2, bits3, bits4, dim, results);
#endif
}

float
RaBitQFloatFourBitCenteredIP(const float* vector, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SSE)
    float results[1];
    const uint8_t* codes[] = {bits};
    simd::ComputeSmallBitplaneIP<SmallBitplaneTraits, 4, true>(vector, codes, dim, 0, results);
    return results[0];
#else
    return generic::RaBitQFloatFourBitCenteredIP(vector, bits, dim);
#endif
}

void
RaBitQFloatFourBitCenteredIPBatch4(const float* vector,
                                   const uint8_t* bits1,
                                   const uint8_t* bits2,
                                   const uint8_t* bits3,
                                   const uint8_t* bits4,
                                   uint64_t dim,
                                   float* results) {
#if defined(ENABLE_SSE)

    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};
    simd::ComputeSmallBitplaneIP<SmallBitplaneTraits, 4, true>(vector, codes, dim, 0, results);

#else
    generic::RaBitQFloatFourBitCenteredIPBatch4(vector, bits1, bits2, bits3, bits4, dim, results);
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
#if defined(ENABLE_SSE)

    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};
    simd::ComputeSmallBitplaneIP<SmallBitplaneTraits, 3, false>(
        vector, codes, dim, reorder_bits, results);

#else
    generic::RaBitQFloatThreeBitIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, reorder_bits, results);
#endif
}

float
RaBitQFloatThreeBitIPByLookup(const float* lookup,
                              const uint8_t* bits,
                              uint64_t dim,
                              uint32_t reorder_bits) {
    return RaBitQFloatMultiBitIPByLookup(lookup, bits, dim, reorder_bits, 3);
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

float
RaBitQFloatMultiBitIPByLookup(const float* lookup,
                              const uint8_t* bits,
                              uint64_t dim,
                              uint32_t reorder_bits,
                              uint32_t filter_bits) {
#if defined(ENABLE_SSE)
    const uint8_t* codes[] = {bits};
    float results[1];
    simd::ComputeSmallLookupIP<simd::SimdTraits<simd::SseTag>>(
        lookup, codes, dim, reorder_bits, filter_bits, results);
    return results[0];
#else
    return generic::RaBitQFloatMultiBitIPByLookup(lookup, bits, dim, reorder_bits, filter_bits);
#endif
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
#if defined(ENABLE_SSE)
    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};

    simd::ComputeSmallLookupIP<simd::SimdTraits<simd::SseTag>>(
        lookup, codes, dim, reorder_bits, filter_bits, results);

#else
    generic::RaBitQFloatMultiBitIPBatch4ByLookup(
        lookup, bits1, bits2, bits3, bits4, dim, reorder_bits, filter_bits, results);
#endif
}

float
RaBitQFloatSQIP(const float* vector, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::RaBitQFloatScalarIPImpl<simd::SQ8Traits<simd::SseSQ8Tag>>(
        vector, codes, dim, &generic::RaBitQFloatSQIP);
#else
    return generic::RaBitQFloatSQIP(vector, codes, dim);
#endif
}

uint64_t
RaBitQCodeCodeIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::RaBitQScalarCodesIPImpl<simd::UniformCodeTraits<simd::SseUniformTag>>(
        codes1, codes2, dim, &generic::RaBitQCodeCodeIP);
#else
    return generic::RaBitQCodeCodeIP(codes1, codes2, dim);
#endif
}

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits) {
#if defined(ENABLE_SSE)
    return simd::ComputeSmallPlanarIP<SmallBitplaneTraits>(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
#else
    return generic::RaBitQFloatSplitCodeIP(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
#endif
}

float
RaBitQFloatSupplementCodeIP(const float* vector,
                            const uint8_t* supplement_code,
                            uint64_t dim,
                            uint32_t supplement_bits) {
#if defined(ENABLE_SSE)
    if (supplement_bits == 0) {
        return 0.0F;
    }
    return simd::ComputeSmallPlanarIP<SmallBitplaneTraits>(
        vector, nullptr, supplement_code, dim, supplement_bits);
#else
    return generic::RaBitQFloatSupplementCodeIP(vector, supplement_code, dim, supplement_bits);
#endif
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_SSE)
    simd::DivScalarImpl<simd::SimdTraits<simd::SseTag>>(from, to, dim, scalar, &generic::DivScalar);
#else
    generic::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
    float norm = std::sqrt(FP32ComputeIP(from, from, dim));
    sse::DivScalar(from, to, dim, norm);
    return norm;
}

void
Prefetch(const void* data) {
#if defined(ENABLE_SSE)
    _mm_prefetch(data, _MM_HINT_T0);
#endif
};

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_SSE)
    __m128i sum[4];
    for (uint64_t i = 0; i < 4; i++) {
        sum[i] = _mm_setzero_si128();
    }
    // Each 16-bit lane accumulates one lookup byte (<= 255) per iteration and
    // would overflow after ~258 of them, so periodically drain the int16
    // partials into the int32 result and reset the accumulators.
    constexpr uint64_t kFlushInterval = 32;
    uint64_t since_flush = 0;
    auto flush = [&]() {
        alignas(16) uint16_t temp[8];
        for (int64_t k = 0; k < 4; k++) {
            _mm_store_si128((__m128i*)(temp), sum[k]);
            for (int64_t j = 0; j < 8; j++) {
                result[k * 8 + j] += temp[j];
            }
            sum[k] = _mm_setzero_si128();
        }
    };
    const auto sign4 = _mm_set1_epi8(0x0F);
    const auto sign8 = _mm_set1_epi16(0xFF);
    for (uint64_t i = 0; i < pq_dim; i++) {
        auto dict = _mm_loadu_si128((__m128i*)(lookup_table));
        lookup_table += 16;
        auto code = _mm_loadu_si128((__m128i*)(codes));
        codes += 16;
        auto code1 = _mm_and_si128(code, sign4);
        auto code2 = _mm_and_si128(_mm_srli_epi16(code, 4), sign4);
        auto res1 = _mm_shuffle_epi8(dict, code1);
        auto res2 = _mm_shuffle_epi8(dict, code2);
        sum[0] = _mm_add_epi32(sum[0], _mm_and_si128(res1, sign8));
        sum[1] = _mm_add_epi32(sum[1], _mm_srli_epi16(res1, 8));
        sum[2] = _mm_add_epi32(sum[2], _mm_and_si128(res2, sign8));
        sum[3] = _mm_add_epi32(sum[3], _mm_srli_epi16(res2, 8));
        if (++since_flush == kFlushInterval) {
            flush();
            since_flush = 0;
        }
    }
    flush();
#else
    generic::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
PQFastScanLookUp32HighAcc(const uint8_t* RESTRICT low_lookup_table,
                          const uint8_t* RESTRICT high_lookup_table,
                          const uint8_t* RESTRICT codes,
                          uint64_t pq_dim,
                          int32_t* RESTRICT result) {
#if defined(ENABLE_SSE)
    const auto nibble = _mm_set1_epi8(15);
    const auto even = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    const auto odd = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i sums[8];
    for (int i = 0; i < 8; ++i) {
        sums[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(result + 4 * i));
    }
    for (uint64_t j = 0; j < pq_dim; ++j) {
        const auto lo =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(low_lookup_table + 16 * j));
        const auto hi =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(high_lookup_table + 16 * j));
        const auto code = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + 16 * j));
        const __m128i indices[] = {_mm_and_si128(code, nibble),
                                   _mm_and_si128(_mm_srli_epi16(code, 4), nibble)};
        const __m128i parity[] = {even, odd};
        for (int half = 0; half < 2; ++half) {
            const auto lows = _mm_shuffle_epi8(lo, indices[half]);
            const auto highs = _mm_shuffle_epi8(hi, indices[half]);
            for (int p = 0; p < 2; ++p) {
                const auto values = _mm_unpacklo_epi8(_mm_shuffle_epi8(lows, parity[p]),
                                                      _mm_shuffle_epi8(highs, parity[p]));
                const int slot = 4 * half + 2 * p;
                sums[slot] = _mm_add_epi32(sums[slot], _mm_cvtepu16_epi32(values));
                sums[slot + 1] =
                    _mm_add_epi32(sums[slot + 1], _mm_cvtepu16_epi32(_mm_srli_si128(values, 8)));
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result + 4 * i), sums[i]);
    }
#else
    generic::PQFastScanLookUp32HighAcc(low_lookup_table, high_lookup_table, codes, pq_dim, result);
#endif
}

void
PQFastScanLookUp32HighAccOverwrite(const uint8_t* RESTRICT low_lookup_table,
                                   const uint8_t* RESTRICT high_lookup_table,
                                   const uint8_t* RESTRICT codes,
                                   uint64_t pq_dim,
                                   int32_t* RESTRICT result) {
    for (uint64_t lane = 0; lane < 32; ++lane) {
        result[lane] = 0;
    }
    PQFastScanLookUp32HighAcc(low_lookup_table, high_lookup_table, codes, pq_dim, result);
}

uint32_t
FP32LessThan32Mask(const float* values, float limit) {
#if defined(ENABLE_SSE)
    const auto limit_vec = _mm_set1_ps(limit);
    uint32_t mask = 0;
    for (uint64_t i = 0; i < 8; ++i) {
        const auto value = _mm_loadu_ps(values + i * 4);
        mask |= static_cast<uint32_t>(_mm_movemask_ps(_mm_cmplt_ps(value, limit_vec))) << (i * 4);
    }
    return mask;
#else
    return generic::FP32LessThan32Mask(values, limit);
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
#if defined(ENABLE_SSE)
    return simd::RaBitQResidual4<simd::SimdTraits<simd::SseTag>>(
        accumulators,
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
        filter_inner_products,
        [](const int32_t* p) {
            return _mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
        });
#else
    return generic::RaBitQFastScan32ResidualPostprocess(accumulators,
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
#if defined(ENABLE_SSE)
    simd::BitAndImpl<simd::BitTraits<simd::SseBitTag>>(x, y, num_byte, result, &generic::BitAnd);
#else
    return generic::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SSE)
    simd::BitOrImpl<simd::BitTraits<simd::SseBitTag>>(x, y, num_byte, result, &generic::BitOr);
#else
    return generic::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SSE)
    simd::BitXorImpl<simd::BitTraits<simd::SseBitTag>>(x, y, num_byte, result, &generic::BitXor);
#else
    return generic::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SSE)
    simd::BitNotImpl<simd::BitTraits<simd::SseBitTag>>(x, num_byte, result, &generic::BitNot);
#else
    return generic::BitNot(x, num_byte, result);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim) {
#if defined(ENABLE_SSE)
    const auto masks = _mm_setr_epi32(1, 2, 4, 8);
    const auto sign = _mm_set1_epi32(INT32_MIN);
    uint64_t i = 0;
    for (; dim - i >= 4; i += 4) {
        const auto bits = _mm_set1_epi32(flip[i / 8] >> (i % 8));
        const auto selected = _mm_cmpeq_epi32(_mm_and_si128(bits, masks), masks);
        const auto values = _mm_loadu_ps(data + i);
        _mm_storeu_ps(data + i,
                      _mm_xor_ps(values, _mm_castsi128_ps(_mm_and_si128(selected, sign))));
    }
    for (; i < dim; ++i) {
        if ((flip[i / 8] & (1U << (i % 8))) != 0U) {
            data[i] = -data[i];
        }
    }
#else
    generic::FlipSign(flip, data, dim);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_SSE)
    simd::RotateOpImpl<simd::SimdTraits<simd::SseTag>>(data, idx, dim_, step);
#else
    generic::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, uint64_t dim_) {
#if defined(ENABLE_SSE)
    uint64_t n = dim_;
    uint64_t step = 1;
    while (step < n) {
        if (step >= 4) {
            sse::RotateOp(data, 0, dim_, step);
        } else {
            generic::RotateOp(data, 0, dim_, step);
        }
        step *= 2;
    }
#else
    return generic::FHTRotate(data, dim_);
#endif
}

void
VecRescale(float* data, uint64_t dim, float val) {
#if defined(ENABLE_SSE)
    simd::VecRescaleImpl<simd::SimdTraits<simd::SseTag>>(data, dim, val, &generic::VecRescale);
#else
    generic::VecRescale(data, dim, val);
#endif
}

void
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_SSE)
    simd::KacsWalkImpl<simd::SimdTraits<simd::SseTag>>(data, len, &generic::KacsWalk);
#else
    generic::KacsWalk(data, len);
#endif
}

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::NormalizeWithCentroidImpl<simd::SimdTraits<simd::SseTag>>(
        from, centroid, to, dim, &generic::NormalizeWithCentroid);
#else
    return generic::NormalizeWithCentroid(from, centroid, to, dim);
#endif
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
#if defined(ENABLE_SSE)
    simd::InverseNormalizeWithCentroidImpl<simd::SimdTraits<simd::SseTag>>(
        from, centroid, to, dim, norm, &generic::InverseNormalizeWithCentroid);
#else
    generic::InverseNormalizeWithCentroid(from, centroid, to, dim, norm);
#endif
}

}  // namespace vsag::sse

namespace vsag::sse {
float
SQ8LocalComputeL2Sqr(const float* query, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::LocalSQCompute<simd::SQ8Traits<simd::SseSQ8Tag>, true, 8, false>(
        query, nullptr, codes, dim);
#else
    return generic::SQ8LocalComputeL2Sqr(query, codes, dim);
#endif
}
float
SQ8LocalComputeCodesL2Sqr(const uint8_t* a, const uint8_t* b, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::LocalSQCompute<simd::SQ8Traits<simd::SseSQ8Tag>, false, 8, false>(
        nullptr, a, b, dim);
#else
    return generic::SQ8LocalComputeCodesL2Sqr(a, b, dim);
#endif
}

float
SQ8LocalComputeIP(const float* query, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::LocalSQCompute<simd::SQ8Traits<simd::SseSQ8Tag>, true, 8, true>(
        query, nullptr, codes, dim);
#else
    return generic::SQ8LocalComputeIP(query, codes, dim);
#endif
}
float
SQ8LocalComputeCodesIP(const uint8_t* a, const uint8_t* b, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::LocalSQCompute<simd::SQ8Traits<simd::SseSQ8Tag>, false, 8, true>(
        nullptr, a, b, dim);
#else
    return generic::SQ8LocalComputeCodesIP(a, b, dim);
#endif
}

float
SQ4LocalComputeL2Sqr(const float* query, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::LocalSQCompute<simd::SQ8Traits<simd::SseSQ8Tag>, true, 4, false>(
        query, nullptr, codes, dim);
#else
    return generic::SQ4LocalComputeL2Sqr(query, codes, dim);
#endif
}
float
SQ4LocalComputeCodesL2Sqr(const uint8_t* a, const uint8_t* b, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::LocalSQCompute<simd::SQ8Traits<simd::SseSQ8Tag>, false, 4, false>(
        nullptr, a, b, dim);
#else
    return generic::SQ4LocalComputeCodesL2Sqr(a, b, dim);
#endif
}

float
SQ4LocalComputeIP(const float* query, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::LocalSQCompute<simd::SQ8Traits<simd::SseSQ8Tag>, true, 4, true>(
        query, nullptr, codes, dim);
#else
    return generic::SQ4LocalComputeIP(query, codes, dim);
#endif
}
float
SQ4LocalComputeCodesIP(const uint8_t* a, const uint8_t* b, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::LocalSQCompute<simd::SQ8Traits<simd::SseSQ8Tag>, false, 4, true>(
        nullptr, a, b, dim);
#else
    return generic::SQ4LocalComputeCodesIP(a, b, dim);
#endif
}
}  // namespace vsag::sse
