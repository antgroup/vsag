
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

#include "simd/int8_simd.h"
#if defined(ENABLE_SSE)
#include <x86intrin.h>

#include "simd/kernels/kernels.h"
#include "simd/traits/simd_traits_sse.h"
#endif

#include <cmath>

#include "simd.h"

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
    simd::PQDistanceFloat256Impl<simd::SimdTraits<simd::SSE_Tag>>(
        single_dim_centers, single_dim_val, result, &generic::PQDistanceFloat256);
#else
    return generic::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

#if defined(ENABLE_SSE)
__inline __m128i __attribute__((__always_inline__)) load_4_char(const uint8_t* data) {
    return _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, data[3], data[2], data[1], data[0]);
}
#endif

#if defined(ENABLE_SSE)
__inline __m128i __attribute__((__always_inline__)) load_4_short(const uint16_t* data) {
    return _mm_set_epi16(data[3], 0, data[2], 0, data[1], 0, data[0], 0);
}
#endif

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::ComputeIPImpl<simd::SimdTraits<simd::SSE_Tag>>(
        query, codes, dim, &generic::FP32ComputeIP);
#else
    return vsag::generic::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::SSE_Tag>>(
        query, codes, dim, &generic::FP32ComputeL2Sqr);
#else
    return vsag::generic::FP32ComputeL2Sqr(query, codes, dim);
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
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::SSE_Tag>, simd::Batch4Kind::IP>(
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
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::SSE_Tag>, simd::Batch4Kind::L2>(
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
    simd::BinaryOpImpl<simd::SimdTraits<simd::SSE_Tag>, simd::BinaryOp::Sub>(
        x, y, z, dim, &generic::FP32Sub);
#else
    return generic::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SSE)
    simd::BinaryOpImpl<simd::SimdTraits<simd::SSE_Tag>, simd::BinaryOp::Add>(
        x, y, z, dim, &generic::FP32Add);
#else
    return generic::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SSE)
    simd::BinaryOpImpl<simd::SimdTraits<simd::SSE_Tag>, simd::BinaryOp::Mul>(
        x, y, z, dim, &generic::FP32Mul);
#else
    return generic::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SSE)
    simd::BinaryOpImpl<simd::SimdTraits<simd::SSE_Tag>, simd::BinaryOp::Div>(
        x, y, z, dim, &generic::FP32Div);
#else
    return generic::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::ReduceAddImpl<simd::SimdTraits<simd::SSE_Tag>>(x, dim, &generic::FP32ReduceAdd);
#else
    return generic::FP32ReduceAdd(x, dim);
#endif
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::HalfComputeIPImpl<simd::BF16Traits<simd::SSE_BF16_Tag>>(
        query, codes, dim, &generic::BF16ComputeIP);
#else
    return generic::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::HalfComputeL2SqrImpl<simd::BF16Traits<simd::SSE_BF16_Tag>>(
        query, codes, dim, &generic::BF16ComputeL2Sqr);
#else
    return generic::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    return generic::FP16ComputeIP(query, codes, dim);
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    return generic::FP16ComputeL2Sqr(query, codes, dim);
}

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::Int8ComputeL2SqrImpl<simd::Int8Traits<simd::SSE_Int8_Tag>>(
        query, codes, dim, &generic::INT8ComputeL2Sqr);
#else
    return generic::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::Int8ComputeIPImpl<simd::Int8Traits<simd::SSE_Int8_Tag>>(
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
    return simd::SQ8ComputeIPImpl<simd::SQ8Traits<simd::SSE_SQ8_Tag>>(
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
    return simd::SQ8ComputeL2SqrImpl<simd::SQ8Traits<simd::SSE_SQ8_Tag>>(
        query, codes, lower_bound, diff, dim, &generic::SQ8ComputeL2Sqr);
#else
    return generic::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ8ComputeCodesIPImpl<simd::SQ8Traits<simd::SSE_SQ8_Tag>>(
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
    return simd::SQ8ComputeCodesL2SqrImpl<simd::SQ8Traits<simd::SSE_SQ8_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &generic::SQ8ComputeCodesL2Sqr);
#else
    return generic::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::SQ4ComputeIPImpl<simd::SQ4Traits<simd::SSE_SQ4_Tag>>(
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
    return simd::SQ4ComputeL2SqrImpl<simd::SQ4Traits<simd::SSE_SQ4_Tag>>(
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
    return simd::SQ4ComputeCodesIPImpl<simd::SQ4Traits<simd::SSE_SQ4_Tag>>(
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
    return simd::SQ4ComputeCodesL2SqrImpl<simd::SQ4Traits<simd::SSE_SQ4_Tag>>(
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
    return simd::SQ4UniformComputeCodesIPImpl<simd::UniformCodeTraits<simd::SSE_Uniform_Tag>>(
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
    return simd::SQ8UniformComputeCodesIPImpl<simd::UniformCodeTraits<simd::SSE_Uniform_Tag>>(
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
    for (uint64_t i = 0; i < n_codes; ++i) {
        out[i] = sse::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_SSE)
    return generic::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);  // TODO(zxy): implement
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
    generic::RaBitQFloatBinaryIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
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
    return generic::RaBitQFloatSplitCodeIP(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
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
#if defined(ENABLE_SSE)
    simd::DivScalarImpl<simd::SimdTraits<simd::SSE_Tag>>(
        from, to, dim, scalar, &generic::DivScalar);
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
    }
    alignas(128) uint16_t temp[8];
    for (int64_t i = 0; i < 4; i++) {
        _mm_store_si128((__m128i*)(temp), sum[i]);
        for (int64_t j = 0; j < 8; j++) {
            result[i * 8 + j] += temp[j];
        }
    }
#else
    generic::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SSE)
    simd::BitAndImpl<simd::BitTraits<simd::SSE_Bit_Tag>>(x, y, num_byte, result, &generic::BitAnd);
#else
    return generic::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SSE)
    simd::BitOrImpl<simd::BitTraits<simd::SSE_Bit_Tag>>(x, y, num_byte, result, &generic::BitOr);
#else
    return generic::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SSE)
    simd::BitXorImpl<simd::BitTraits<simd::SSE_Bit_Tag>>(x, y, num_byte, result, &generic::BitXor);
#else
    return generic::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SSE)
    simd::BitNotImpl<simd::BitTraits<simd::SSE_Bit_Tag>>(x, num_byte, result, &generic::BitNot);
#else
    return generic::BitNot(x, num_byte, result);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_SSE)
    simd::RotateOpImpl<simd::SimdTraits<simd::SSE_Tag>>(data, idx, dim_, step);
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
    simd::VecRescaleImpl<simd::SimdTraits<simd::SSE_Tag>>(data, dim, val, &generic::VecRescale);
#else
    generic::VecRescale(data, dim, val);
#endif
}

void
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_SSE)
    simd::KacsWalkImpl<simd::SimdTraits<simd::SSE_Tag>>(data, len, &generic::KacsWalk);
#else
    generic::KacsWalk(data, len);
#endif
}

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
#if defined(ENABLE_SSE)
    return simd::NormalizeWithCentroidImpl<simd::SimdTraits<simd::SSE_Tag>>(
        from, centroid, to, dim, &generic::NormalizeWithCentroid);
#else
    return generic::NormalizeWithCentroid(from, centroid, to, dim);
#endif
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
#if defined(ENABLE_SSE)
    simd::InverseNormalizeWithCentroidImpl<simd::SimdTraits<simd::SSE_Tag>>(
        from, centroid, to, dim, norm, &generic::InverseNormalizeWithCentroid);
#else
    generic::InverseNormalizeWithCentroid(from, centroid, to, dim, norm);
#endif
}

}  // namespace vsag::sse
