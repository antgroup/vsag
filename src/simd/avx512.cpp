
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
#include "simd/int8_simd.h"
#if defined(ENABLE_AVX512)
#include <immintrin.h>

#include "simd/kernels/kernels.h"
#include "simd/traits/simd_traits_avx512.h"
#endif

#include <cmath>

#include "simd.h"

namespace vsag::avx512 {
float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* float_vec1 = (float*)pVect1v;
    auto* float_vec2 = (float*)pVect2v;
    uint64_t dim = *((uint64_t*)qty_ptr);
    return avx512::FP32ComputeL2Sqr(float_vec1, float_vec2, dim);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* float_vec1 = (float*)pVect1v;
    auto* float_vec2 = (float*)pVect2v;
    uint64_t dim = *((uint64_t*)qty_ptr);
    return avx512::FP32ComputeIP(float_vec1, float_vec2, dim);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - avx512::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx512::INT8ComputeL2Sqr(pVect1, pVect2, qty);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return avx512::INT8ComputeIP(pVect1, pVect2, qty);
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -avx512::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_AVX512)
    simd::PQDistanceFloat256Impl<simd::SimdTraits<simd::AVX512_Tag>>(
        single_dim_centers, single_dim_val, result, &avx2::PQDistanceFloat256);
#else
    return avx2::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

void
Prefetch(const void* data) {
    avx2::Prefetch(data);
}

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::ComputeIPImpl<simd::SimdTraits<simd::AVX512_Tag>, /*Unroll=*/4>(
        query, codes, dim, &avx2::FP32ComputeIP);
#else
    return avx2::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::AVX512_Tag>, /*Unroll=*/4>(
        query, codes, dim, &avx2::FP32ComputeL2Sqr);
#else
    return avx2::FP32ComputeL2Sqr(query, codes, dim);
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
#if defined(ENABLE_AVX512)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::AVX512_Tag>, simd::Batch4Kind::IP>(
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
        &avx2::FP32ComputeIPBatch4);
#else
    return avx2::FP32ComputeIPBatch4(
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
#if defined(ENABLE_AVX512)
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::AVX512_Tag>, simd::Batch4Kind::L2>(
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
        &avx2::FP32ComputeL2SqrBatch4);
#else
    return avx::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX512)
    simd::BinaryOpImpl<simd::SimdTraits<simd::AVX512_Tag>, simd::BinaryOp::Sub>(
        x, y, z, dim, &avx2::FP32Sub);
#else
    return avx2::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX512)
    simd::BinaryOpImpl<simd::SimdTraits<simd::AVX512_Tag>, simd::BinaryOp::Add>(
        x, y, z, dim, &avx2::FP32Add);
#else
    return avx2::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX512)
    simd::BinaryOpImpl<simd::SimdTraits<simd::AVX512_Tag>, simd::BinaryOp::Mul>(
        x, y, z, dim, &avx2::FP32Mul);
#else
    return avx2::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX512)
    simd::BinaryOpImpl<simd::SimdTraits<simd::AVX512_Tag>, simd::BinaryOp::Div>(
        x, y, z, dim, &avx2::FP32Div);
#else
    return avx2::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::ReduceAddImpl<simd::SimdTraits<simd::AVX512_Tag>>(x, dim, &avx2::FP32ReduceAdd);
#else
    return sse::FP32ReduceAdd(x, dim);
#endif
}

#if defined(ENABLE_AVX512)
__inline __m512i __attribute__((__always_inline__)) load_16_short(const uint16_t* data) {
    __m256i bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
    __m512i bf32 = _mm512_cvtepu16_epi32(bf16);
    return _mm512_slli_epi32(bf32, 16);
}
#endif

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::Int8ComputeIPImpl<simd::Int8Traits<simd::AVX512_Int8_Tag>>(
        query, codes, dim, &avx2::INT8ComputeIP);
#else
    return avx2::INT8ComputeIP(query, codes, dim);
#endif
}

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::Int8ComputeL2SqrImpl<simd::Int8Traits<simd::AVX512_Int8_Tag>>(
        query, codes, dim, &avx2::INT8ComputeL2Sqr);
#else
    return avx2::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::HalfComputeIPImpl<simd::BF16Traits<simd::AVX512_BF16_Tag>>(
        query, codes, dim, &avx2::BF16ComputeIP);
#else
    return avx2::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::HalfComputeL2SqrImpl<simd::BF16Traits<simd::AVX512_BF16_Tag>>(
        query, codes, dim, &avx2::BF16ComputeL2Sqr);
#else
    return avx2::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::HalfComputeIPImpl<simd::FP16Traits<simd::AVX512_FP16_Tag>>(
        query, codes, dim, &avx2::FP16ComputeIP);
#else
    return avx2::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::HalfComputeL2SqrImpl<simd::FP16Traits<simd::AVX512_FP16_Tag>>(
        query, codes, dim, &avx2::FP16ComputeL2Sqr);
#else
    return avx2::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ8ComputeIPImpl<simd::SQ8Traits<simd::AVX512_SQ8_Tag>>(
        query, codes, lower_bound, diff, dim, &avx2::SQ8ComputeIP);
#else
    return avx2::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ8ComputeL2SqrImpl<simd::SQ8Traits<simd::AVX512_SQ8_Tag>>(
        query, codes, lower_bound, diff, dim, &avx2::SQ8ComputeL2Sqr);
#else
    return avx2::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ8ComputeCodesIPImpl<simd::SQ8Traits<simd::AVX512_SQ8_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &avx2::SQ8ComputeCodesIP);
#else
    return avx2::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ8ComputeCodesL2SqrImpl<simd::SQ8Traits<simd::AVX512_SQ8_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &avx2::SQ8ComputeCodesL2Sqr);
#else
    return avx2::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

#if defined(ENABLE_AVX512)
// Helper: unpack 16 bytes (32 x 4-bit) into two __m512 scaled floats [0..1]
static inline void
unpack_4bit_to_m512(const uint8_t* codes,
                    __m512* out0,  // first 16 values (v0..v15)
                    __m512* out1   // next 16 values (v16..v31)
) {
    __m128i code_vec = _mm_loadu_si128((const __m128i*)codes);

    __m128i low_nibbles = _mm_and_si128(code_vec, _mm_set1_epi8(0x0F));
    __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(code_vec, 4), _mm_set1_epi8(0x0F));

    __m128i unpacked_lo = _mm_unpacklo_epi8(low_nibbles, high_nibbles);  // 16 bytes: v0..v15
    __m128i unpacked_hi = _mm_unpackhi_epi8(low_nibbles, high_nibbles);  // 16 bytes: v16..v31

    __m512i indices0 = _mm512_cvtepu8_epi32(unpacked_lo);
    __m512i indices1 = _mm512_cvtepu8_epi32(unpacked_hi);

    __m512 floats0 = _mm512_cvtepi32_ps(indices0);
    __m512 floats1 = _mm512_cvtepi32_ps(indices1);

    const __m512 scale = _mm512_set1_ps(1.0f / 15.0f);
    *out0 = _mm512_mul_ps(floats0, scale);
    *out1 = _mm512_mul_ps(floats1, scale);
}
#endif

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ4ComputeIPImpl<simd::SQ4Traits<simd::AVX512_SQ4_Tag>>(
        query, codes, lower_bound, diff, dim, &avx2::SQ4ComputeIP);
#else
    return avx2::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ4ComputeL2SqrImpl<simd::SQ4Traits<simd::AVX512_SQ4_Tag>>(
        query, codes, lower_bound, diff, dim, &avx2::SQ4ComputeL2Sqr);
#else
    return avx2::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ4ComputeCodesIPImpl<simd::SQ4Traits<simd::AVX512_SQ4_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &avx2::SQ4ComputeCodesIP);
#else
    return avx2::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ4ComputeCodesL2SqrImpl<simd::SQ4Traits<simd::AVX512_SQ4_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &avx2::SQ4ComputeCodesL2Sqr);
#else
    return avx2::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ4UniformComputeCodesIPImpl<simd::UniformCodeTraits<simd::AVX512_Uniform_Tag>>(
        codes1, codes2, dim, &avx2::SQ4UniformComputeCodesIP);
#else
    return avx2::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::SQ8UniformComputeCodesIPImpl<simd::UniformCodeTraits<simd::AVX512_Uniform_Tag>>(
        codes1, codes2, dim, &avx2::SQ8UniformComputeCodesIP);
#else
    return avx2::SQ8UniformComputeCodesIP(codes1, codes2, dim);
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
        out[i] = avx512::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_AVX512)
    return simd::RaBitQFloatBinaryIPImpl<simd::RaBitQTraits<simd::AVX512_RaBitQ_Tag>>(
        vector, bits, dim, inv_sqrt_d, &avx2::RaBitQFloatBinaryIP);
#else
    return avx2::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
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
#if defined(ENABLE_AVX512)
    simd::RaBitQFloatBinaryIPBatch4Impl<simd::RaBitQTraits<simd::AVX512_RaBitQ_Tag>>(
        vector,
        bits1,
        bits2,
        bits3,
        bits4,
        dim,
        inv_sqrt_d,
        results,
        &avx2::RaBitQFloatBinaryIPBatch4);
#else
    avx2::RaBitQFloatBinaryIPBatch4(vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
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
    avx2::RaBitQFloatThreeBitIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, reorder_bits, results);
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
#if defined(ENABLE_AVX512)
    const uint64_t plane_bytes = (dim + 7) / 8;
    __m512 sum = _mm512_setzero_ps();
    uint64_t block = 0;
    for (; block + 16 <= plane_bytes; block += 16) {
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto* plane = bits + static_cast<uint64_t>(bit) * plane_bytes;
            alignas(64) int indices[16];
            for (uint32_t lane = 0; lane < 16; ++lane) {
                indices[lane] = static_cast<int>((block + lane) * 256 + plane[block + lane]);
            }
            const __m512i index = _mm512_load_si512(reinterpret_cast<const void*>(indices));
            const __m512 values = _mm512_i32gather_ps(index, lookup, 4);
            const __m512 weight =
                _mm512_set1_ps(static_cast<float>(1U << (reorder_bits + filter_bits - bit - 1)));
            sum = _mm512_fmadd_ps(values, weight, sum);
        }
    }

    float result = _mm512_reduce_add_ps(sum);
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
    return avx2::RaBitQFloatMultiBitIPByLookup(lookup, bits, dim, reorder_bits, filter_bits);
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
#if defined(ENABLE_AVX512)
    results[0] = 0.0F;
    results[1] = 0.0F;
    results[2] = 0.0F;
    results[3] = 0.0F;

    const uint64_t plane_bytes = (dim + 7) / 8;
    const uint8_t* codes[4] = {bits1, bits2, bits3, bits4};
    __m512 sums[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};

    uint64_t block = 0;
    for (; block + 16 <= plane_bytes; block += 16) {
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t bit = 0; bit < filter_bits; ++bit) {
                const auto* plane = codes[i] + static_cast<uint64_t>(bit) * plane_bytes;
                alignas(64) int indices[16];
                for (uint32_t lane = 0; lane < 16; ++lane) {
                    indices[lane] = static_cast<int>((block + lane) * 256 + plane[block + lane]);
                }
                const __m512i index = _mm512_load_si512(reinterpret_cast<const void*>(indices));
                const __m512 values = _mm512_i32gather_ps(index, lookup, 4);
                const __m512 weight = _mm512_set1_ps(
                    static_cast<float>(1U << (reorder_bits + filter_bits - bit - 1)));
                sums[i] = _mm512_fmadd_ps(values, weight, sums[i]);
            }
        }
    }

    for (uint32_t i = 0; i < 4; ++i) {
        results[i] = _mm512_reduce_add_ps(sums[i]);
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
    avx2::RaBitQFloatMultiBitIPBatch4ByLookup(
        lookup, bits1, bits2, bits3, bits4, dim, reorder_bits, filter_bits, results);
#endif
}

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits) {
#if defined(ENABLE_AVX512)
    return simd::RaBitQFloatSplitCodeIPImpl<simd::RaBitQTraits<simd::AVX512_RaBitQ_Tag>>(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
#else
    return avx2::RaBitQFloatSplitCodeIP(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
#endif
}

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
    // require dim align with 512
#if defined(ENABLE_AVX512)
    if (dim == 0) {
        return 0;
    }

    // LUT has size of 2^8, lookup[i] = pop_count(i), where len(i) == 8
    const __m512i lookup = _mm512_setr_epi64(0x0302020102010100llu,
                                             0x0403030203020201llu,
                                             0x0302020102010100llu,
                                             0x0403030203020201llu,
                                             0x0302020102010100llu,
                                             0x0403030203020201llu,
                                             0x0302020102010100llu,
                                             0x0403030203020201llu);

    uint32_t result = 0;
    uint64_t num_bytes = (dim + 7) / 8;

    const __m512i low_mask = _mm512_set1_epi8(0x0F);

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        uint64_t i = 0;

        __m512i acc = _mm512_setzero_si512();

        for (; i + 64 <= num_bytes; i += 64) {
            __m512i vec_codes = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(codes + bit_pos * num_bytes + i));
            __m512i vec_bits = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(bits + i));

            __m512i and_result = _mm512_and_si512(vec_codes, vec_bits);

            // 64 * 8
            __m512i lo = _mm512_and_si512(and_result, low_mask);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi32(and_result, 4), low_mask);

            __m512i popcnt1 = _mm512_shuffle_epi8(lookup, lo);
            __m512i popcnt2 = _mm512_shuffle_epi8(lookup, hi);

            __m512i local = _mm512_add_epi8(popcnt1, popcnt2);

            // 8 * 64
            acc = _mm512_add_epi64(acc, _mm512_sad_epu8(local, _mm512_setzero_si512()));
        }

        __m256i t0 = _mm512_extracti64x4_epi64(acc, 0);
        __m256i t1 = _mm512_extracti64x4_epi64(acc, 1);

        uint64_t p0 = _mm256_extract_epi64(t0, 0) + _mm256_extract_epi64(t0, 1) +
                      _mm256_extract_epi64(t0, 2) + _mm256_extract_epi64(t0, 3);

        uint64_t p1 = _mm256_extract_epi64(t1, 0) + _mm256_extract_epi64(t1, 1) +
                      _mm256_extract_epi64(t1, 2) + _mm256_extract_epi64(t1, 3);

        uint64_t sum = p0 + p1;

        for (; i < num_bytes; ++i) {
            uint8_t bitwise_and = codes[bit_pos * num_bytes + i] & bits[i];
            sum += __builtin_popcount(bitwise_and);
        }

        result += sum << bit_pos;
    }

    return result;

#else
    return generic::RaBitQSQ4UBinaryIP(codes, bits, dim);
#endif
}

float
RaBitQFloatSupplementCodeIP(const float* vector,
                            const uint8_t* supplement_code,
                            uint64_t dim,
                            uint32_t supplement_bits) {
    return avx2::RaBitQFloatSupplementCodeIP(vector, supplement_code, dim, supplement_bits);
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_AVX512)
    simd::DivScalarImpl<simd::SimdTraits<simd::AVX512_Tag>>(
        from, to, dim, scalar, &avx2::DivScalar);
#else
    avx2::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
    float norm = std::sqrt(FP32ComputeIP(from, from, dim));
    avx512::DivScalar(from, to, dim, norm);
    return norm;
}

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_AVX512)
    if (pq_dim == 0) {
        return;
    }
    __m512i sum[4];
    for (uint64_t i = 0; i < 4; i++) {
        sum[i] = _mm512_setzero_si512();
    }
    const auto sign4 = _mm512_set1_epi8(0x0F);
    const auto sign8 = _mm512_set1_epi16(0xFF);
    uint64_t i = 0;
    for (; i + 3 < pq_dim; i += 4) {
        auto dict = _mm512_loadu_si512((__m512i*)(lookup_table));
        lookup_table += 64;
        auto code = _mm512_loadu_si512((__m512i*)(codes));
        codes += 64;
        auto code1 = _mm512_and_si512(code, sign4);
        auto code2 = _mm512_and_si512(_mm512_srli_epi16(code, 4), sign4);
        auto res1 = _mm512_shuffle_epi8(dict, code1);
        auto res2 = _mm512_shuffle_epi8(dict, code2);
        sum[0] = _mm512_add_epi16(sum[0], _mm512_and_si512(res1, sign8));
        sum[1] = _mm512_add_epi16(sum[1], _mm512_srli_epi16(res1, 8));
        sum[2] = _mm512_add_epi16(sum[2], _mm512_and_si512(res2, sign8));
        sum[3] = _mm512_add_epi16(sum[3], _mm512_srli_epi16(res2, 8));
    }
    alignas(512) uint16_t temp[32];
    for (int64_t idx = 0; idx < 4; idx++) {
        _mm512_store_si512((__m512i*)(temp), sum[idx]);
        for (int64_t j = 0; j < 8; j++) {
            result[idx * 8 + j] += temp[j] + temp[j + 8] + temp[j + 16] + temp[j + 24];
        }
    }
    if (pq_dim > i) {
        avx2::PQFastScanLookUp32(lookup_table, codes, pq_dim - i, result);
    }
#else
    avx2::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX512)
    simd::BitAndImpl<simd::BitTraits<simd::AVX512_Bit_Tag>>(x, y, num_byte, result, &avx2::BitAnd);
#else
    return avx2::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX512)
    simd::BitOrImpl<simd::BitTraits<simd::AVX512_Bit_Tag>>(x, y, num_byte, result, &avx2::BitOr);
#else
    return avx2::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX512)
    simd::BitXorImpl<simd::BitTraits<simd::AVX512_Bit_Tag>>(x, y, num_byte, result, &avx2::BitXor);
#else
    return avx2::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX512)
    simd::BitNotImpl<simd::BitTraits<simd::AVX512_Bit_Tag>>(x, num_byte, result, &avx2::BitNot);
#else
    return avx2::BitNot(x, num_byte, result);
#endif
}

void
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_AVX512)
    simd::KacsWalkImpl<simd::SimdTraits<simd::AVX512_Tag>>(data, len, &avx2::KacsWalk);
#else
    avx2::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim) {
#if defined(ENABLE_AVX512)
    constexpr uint64_t kFloatsPerChunk = 64;
    uint64_t i = 0;
    for (; i + 64 < dim; i += kFloatsPerChunk) {
        // Load 64 bits (8 bytes) from the bit sequence
        uint64_t mask_bits;
        std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

        // Split into four 16-bit mask segments
        const __mmask16 mask0 = _cvtu32_mask16(static_cast<uint32_t>(mask_bits & 0xFFFF));
        const __mmask16 mask1 = _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 16) & 0xFFFF));
        const __mmask16 mask2 = _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 32) & 0xFFFF));
        const __mmask16 mask3 = _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 48) & 0xFFFF));

        // Prepare sign-flip constant
        const __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));

        // Process 16 floats at a time with each mask segment
        __m512 vec0 = _mm512_loadu_ps(&data[i]);
        vec0 = _mm512_mask_xor_ps(vec0, mask0, vec0, sign_flip);
        _mm512_storeu_ps(&data[i], vec0);

        __m512 vec1 = _mm512_loadu_ps(&data[i + 16]);
        vec1 = _mm512_mask_xor_ps(vec1, mask1, vec1, sign_flip);
        _mm512_storeu_ps(&data[i + 16], vec1);

        __m512 vec2 = _mm512_loadu_ps(&data[i + 32]);
        vec2 = _mm512_mask_xor_ps(vec2, mask2, vec2, sign_flip);
        _mm512_storeu_ps(&data[i + 32], vec2);

        __m512 vec3 = _mm512_loadu_ps(&data[i + 48]);
        vec3 = _mm512_mask_xor_ps(vec3, mask3, vec3, sign_flip);
        _mm512_storeu_ps(&data[i + 48], vec3);
    }
    for (; i < dim; i++) {
        bool mask = (flip[i / 8] & (1 << (i % 8))) != 0;
        if (mask) {
            data[i] = -data[i];
        }
    }
#else
    return generic::FlipSign(flip, data, dim);
#endif
}

void
VecRescale(float* data, uint64_t dim, float val) {
#if defined(ENABLE_AVX512)
    simd::VecRescaleImpl<simd::SimdTraits<simd::AVX512_Tag>>(data, dim, val, &avx2::VecRescale);
#else
    avx2::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_AVX512)
    simd::RotateOpImpl<simd::SimdTraits<simd::AVX512_Tag>>(data, idx, dim_, step);
#else
    avx2::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, uint64_t dim_) {
#if defined(ENABLE_AVX512)
    uint64_t n = dim_;
    uint64_t step = 1;
    while (step < n) {
        if (step >= 16) {  // step is the power of 2
            avx512::RotateOp(data, 0, dim_, step);
        } else if (step == 8) {
            avx2::RotateOp(data, 0, dim_, step);
        } else if (step == 4) {
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

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return simd::NormalizeWithCentroidImpl<simd::SimdTraits<simd::AVX512_Tag>>(
        from, centroid, to, dim, &avx2::NormalizeWithCentroid);
#else
    return avx2::NormalizeWithCentroid(from, centroid, to, dim);
#endif
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
#if defined(ENABLE_AVX512)
    simd::InverseNormalizeWithCentroidImpl<simd::SimdTraits<simd::AVX512_Tag>>(
        from, centroid, to, dim, norm, &avx2::InverseNormalizeWithCentroid);
#else
    avx2::InverseNormalizeWithCentroid(from, centroid, to, dim, norm);
#endif
}

}  // namespace vsag::avx512
