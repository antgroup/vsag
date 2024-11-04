
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

#include <immintrin.h>

#include <cmath>

#include "fp32_simd.h"
#include "normalize.h"
#include "simd.h"
#include "sq4_simd.h"
#include "sq4_uniform_simd.h"
#include "sq8_simd.h"

namespace vsag {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

float
L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}

float
InnerProductSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;

    const float* pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return sum;
}

float
INT8InnerProduct512AVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    __mmask32 mask = 0xFFFFFFFF;
    __mmask64 mask64 = 0xFFFFFFFFFFFFFFFF;

    size_t qty = *((size_t*)qty_ptr);
    int32_t cTmp[16];

    int8_t* pVect1 = (int8_t*)pVect1v;
    int8_t* pVect2 = (int8_t*)pVect2v;
    const int8_t* pEnd1 = pVect1 + qty;

    __m512i sum512 = _mm512_set1_epi32(0);

    while (pVect1 < pEnd1) {
        __m256i v1 = _mm256_maskz_loadu_epi8(mask, pVect1);
        __m512i v1_512 = _mm512_cvtepi8_epi16(v1);
        pVect1 += 32;
        __m256i v2 = _mm256_maskz_loadu_epi8(mask, pVect2);
        __m512i v2_512 = _mm512_cvtepi8_epi16(v2);
        pVect2 += 32;

        sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v1_512, v2_512));
    }

    _mm512_mask_storeu_epi32(cTmp, mask64, sum512);
    double res = 0;
    for (int i = 0; i < 16; i++) {
        res += cTmp[i];
    }
    return res;
}

float
INT8InnerProduct256AVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    __mmask16 mask = 0xFFFF;
    __mmask64 mask64 = 0xFFFFFFFFFFFFFFFF;
    size_t qty = *((size_t*)qty_ptr);

    int32_t cTmp[16];

    int8_t* pVect1 = (int8_t*)pVect1v;
    int8_t* pVect2 = (int8_t*)pVect2v;
    const int8_t* pEnd1 = pVect1 + qty;

    __m512i sum512 = _mm512_set1_epi32(0);

    while (pVect1 < pEnd1) {
        __m128i v1 = _mm_maskz_loadu_epi8(mask, pVect1);
        __m512i v1_512 = _mm512_cvtepi8_epi32(v1);
        pVect1 += 16;
        __m128i v2 = _mm_maskz_loadu_epi8(mask, pVect2);
        __m512i v2_512 = _mm512_cvtepi8_epi32(v2);
        pVect2 += 16;

        sum512 = _mm512_add_epi32(sum512, _mm512_mullo_epi32(v1_512, v2_512));
    }

    _mm512_mask_storeu_epi32(cTmp, mask64, sum512);
    double res = 0;
    for (int i = 0; i < 16; i++) {
        res += cTmp[i];
    }
    return res;
}

float
INT8InnerProduct256ResidualsAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty2 = qty >> 4 << 4;
    double res = INT8InnerProduct256AVX512(pVect1v, pVect2v, &qty2);
    int8_t* pVect1 = (int8_t*)pVect1v + qty2;
    int8_t* pVect2 = (int8_t*)pVect2v + qty2;

    size_t qty_left = qty - qty2;
    if (qty_left != 0) {
        res += INT8InnerProduct(pVect1, pVect2, &qty_left);
    }
    return res;
}

float
INT8InnerProduct256ResidualsAVX512Distance(const void* pVect1v,
                                           const void* pVect2v,
                                           const void* qty_ptr) {
    return -INT8InnerProduct256ResidualsAVX512(pVect1v, pVect2v, qty_ptr);
}

float
INT8InnerProduct512ResidualsAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty2 = qty >> 5 << 5;
    double res = INT8InnerProduct512AVX512(pVect1v, pVect2v, &qty2);
    int8_t* pVect1 = (int8_t*)pVect1v + qty2;
    int8_t* pVect2 = (int8_t*)pVect2v + qty2;

    size_t qty_left = qty - qty2;
    if (qty_left != 0) {
        res += INT8InnerProduct256ResidualsAVX512(pVect1, pVect2, &qty_left);
    }
    return res;
}

float
INT8InnerProduct512ResidualsAVX512Distance(const void* pVect1v,
                                           const void* pVect2v,
                                           const void* qty_ptr) {
    return -INT8InnerProduct512ResidualsAVX512(pVect1v, pVect2v, qty_ptr);
}

namespace avx512 {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    const int n = dim / 16;
    if (n == 0) {
        return avx2::FP32ComputeIP(query, codes, dim);
    }
    // process 16 floats at a time
    __m512 sum = _mm512_setzero_ps();  // initialize to 0
    for (int i = 0; i < n; ++i) {
        __m512 a = _mm512_loadu_ps(query + i * 16);     // load 16 floats from memory
        __m512 b = _mm512_loadu_ps(codes + i * 16);     // load 16 floats from memory
        sum = _mm512_add_ps(sum, _mm512_mul_ps(a, b));  // accumulate the product
    }
    float ip = _mm512_reduce_add_ps(sum);
    ip += avx2::FP32ComputeIP(query + n * 16, codes + n * 16, dim - n * 16);
    return ip;
#else
    return vsag::Generic::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    const int n = dim / 16;
    if (n == 0) {
        return avx2::FP32ComputeL2Sqr(query, codes, dim);
    }
    // process 16 floats at a time
    __m512 sum = _mm512_setzero_ps();  // initialize to 0
    for (int i = 0; i < n; ++i) {
        __m512 a = _mm512_loadu_ps(query + i * 16);  // load 16 floats from memory
        __m512 b = _mm512_loadu_ps(codes + i * 16);  // load 16 floats from memory
        __m512 diff = _mm512_sub_ps(a, b);           // calculate the difference
        sum = _mm512_fmadd_ps(diff, diff, sum);      // accumulate the squared difference
    }
    float l2 = _mm512_reduce_add_ps(sum);
    l2 += avx2::FP32ComputeL2Sqr(query + n * 16, codes + n * 16, dim - n * 16);
    return l2;
#else
    return vsag::Generic::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim) {
#if defined(ENABLE_AVX512)
    // Initialize the sum to 0
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    // Process the data in 512-bit chunks
    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m128i code_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i));
        __m512i codes_512 = _mm512_cvtepu8_epi32(code_values);
        __m512 code_floats = _mm512_cvtepi32_ps(codes_512);
        __m512 query_values = _mm512_loadu_ps(query + i);
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lowerBound_values = _mm512_loadu_ps(lowerBound + i);

        // Perform calculations
        __m512 scaled_codes =
            _mm512_mul_ps(_mm512_div_ps(code_floats, _mm512_set1_ps(255.0f)), diff_values);
        __m512 adjusted_codes = _mm512_add_ps(scaled_codes, lowerBound_values);
        __m512 val = _mm512_mul_ps(query_values, adjusted_codes);
        sum = _mm512_add_ps(sum, val);
    }
    // Horizontal addition
    float finalResult = _mm512_reduce_add_ps(sum);
    // Process the remaining elements recursively
    finalResult += avx2::SQ8ComputeIP(query + i, codes + i, lowerBound + i, diff + i, dim - i);
    return finalResult;
#else
    return Generic::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lowerBound,
                const float* diff,
                uint64_t dim) {
#if defined(ENABLE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m128i code_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i));
        __m512i codes_512 = _mm512_cvtepu8_epi32(code_values);
        __m512 code_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes_512), _mm512_set1_ps(255.0f));
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lowerBound_values = _mm512_loadu_ps(lowerBound + i);
        __m512 query_values = _mm512_loadu_ps(query + i);

        // Perform calculations
        __m512 scaled_codes = _mm512_mul_ps(code_floats, diff_values);
        scaled_codes = _mm512_add_ps(scaled_codes, lowerBound_values);
        __m512 val = _mm512_sub_ps(query_values, scaled_codes);
        val = _mm512_mul_ps(val, val);
        sum = _mm512_add_ps(sum, val);
    }

    // Horizontal addition
    float result = _mm512_reduce_add_ps(sum);
    // Process the remaining elements
    result += avx2::SQ8ComputeL2Sqr(query + i, codes + i, lowerBound + i, diff + i, dim - i);
    return result;
#else
    return Generic::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m128i code1_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes1 + i));
        __m128i code2_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes2 + i));
        __m512i codes1_512 = _mm512_cvtepu8_epi32(code1_values);
        __m512i codes2_512 = _mm512_cvtepu8_epi32(code2_values);
        __m512 code1_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes1_512), _mm512_set1_ps(255.0f));
        __m512 code2_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes2_512), _mm512_set1_ps(255.0f));
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lowerBound_values = _mm512_loadu_ps(lowerBound + i);

        // Perform calculations
        __m512 scaled_codes1 = _mm512_fmadd_ps(code1_floats, diff_values, lowerBound_values);
        __m512 scaled_codes2 = _mm512_fmadd_ps(code2_floats, diff_values, lowerBound_values);
        __m512 val = _mm512_mul_ps(scaled_codes1, scaled_codes2);
        sum = _mm512_add_ps(sum, val);
    }
    // Horizontal addition
    float result = _mm512_reduce_add_ps(sum);
    // Process the remaining elements
    result += avx2::SQ8ComputeCodesIP(codes1 + i, codes2 + i, lowerBound + i, diff + i, dim - i);
    return result;
#else
    return Generic::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lowerBound,
                     const float* diff,
                     uint64_t dim) {
#if defined(ENABLE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        __m128i code1_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes1 + i));
        __m128i code2_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes2 + i));
        __m512i codes1_512 = _mm512_cvtepu8_epi32(code1_values);
        __m512i codes2_512 = _mm512_cvtepu8_epi32(code2_values);
        __m512 code1_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes1_512), _mm512_set1_ps(255.0f));
        __m512 code2_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes2_512), _mm512_set1_ps(255.0f));
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lowerBound_values = _mm512_loadu_ps(lowerBound + i);

        // Perform calculations
        __m512 scaled_codes1 = _mm512_fmadd_ps(code1_floats, diff_values, lowerBound_values);
        __m512 scaled_codes2 = _mm512_fmadd_ps(code2_floats, diff_values, lowerBound_values);
        __m512 val = _mm512_sub_ps(scaled_codes1, scaled_codes2);
        val = _mm512_mul_ps(val, val);
        sum = _mm512_add_ps(sum, val);
    }

    // Horizontal addition
    float result = _mm512_reduce_add_ps(sum);
    // Process the remaining elements
    result += avx2::SQ8ComputeCodesL2Sqr(codes1 + i, codes2 + i, lowerBound + i, diff + i, dim - i);
    return result;
#else
    return Generic::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
#endif
}

float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             uint64_t dim) {
    return generic::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                uint64_t dim) {
    return generic::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  uint64_t dim) {
    return generic::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     uint64_t dim) {
    return generic::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim == 0) {
        return 0;
    }
    alignas(512) int16_t temp[32];
    int32_t result = 0;
    uint64_t d = 0;
    __m512i sum = _mm512_setzero_si512();
    __m512i mask = _mm512_set1_epi8(0xf);
    for (; d + 127 < dim; d += 128) {
        auto xx = _mm512_loadu_si512((__m512i*)(codes1 + (d >> 1)));
        auto yy = _mm512_loadu_si512((__m512i*)(codes2 + (d >> 1)));
        auto xx1 = _mm512_and_si512(xx, mask);                        // 64 * 8bits
        auto xx2 = _mm512_and_si512(_mm512_srli_epi16(xx, 4), mask);  // 64 * 8bits
        auto yy1 = _mm512_and_si512(yy, mask);
        auto yy2 = _mm512_and_si512(_mm512_srli_epi16(yy, 4), mask);

        sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(xx1, yy1));
        sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(xx2, yy2));
    }
    _mm512_store_si512((__m512i*)temp, sum);
    for (int i = 0; i < 32; ++i) {
        result += temp[i];
    }
    result += avx2::SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);
    return result;
#else
    return avx2::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_AVX2)
    if (dim == 0) {
        return;
    }
    if (scalar == 0) {
        scalar = 1.0f;  // TODO(LHT): logger?
    }
    int i = 0;
    __m512 scalarVec = _mm512_set1_ps(scalar);
    for (; i + 15 < dim; i += 16) {
        __m512 vec = _mm512_loadu_ps(from + i);
        vec = _mm512_div_ps(vec, scalarVec);
        _mm512_storeu_ps(to + i, vec);
    }
    avx2::DivScalar(from + i, to + i, dim - i, scalar);
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

}  // namespace avx512

}  // namespace vsag
