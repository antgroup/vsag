
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

#include <x86intrin.h>

#include <cmath>

#include "fp32_simd.h"
#include "normalize.h"
#include "sq4_simd.h"
#include "sq4_uniform_simd.h"
#include "sq8_simd.h"

namespace vsag {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

extern float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

extern float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);

/* L2 Distance */
float
L2SqrSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty4 = qty >> 2;

    const float* pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

float
L2SqrSIMD4ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4ExtSSE(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float* pVect1 = (float*)pVect1v + qty4;
    float* pVect2 = (float*)pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}

float
L2SqrSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

extern float (*L2SqrSIMD16Ext)(const void*, const void*, const void*);

float
L2SqrSIMD16ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float* pVect1 = (float*)pVect1v + qty16;
    float* pVect2 = (float*)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}

/* IP Distance */
float
InnerProductSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float* pEnd1 = pVect1 + 16 * qty16;
    const float* pEnd2 = pVect1 + 4 * qty4;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

float
InnerProductSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;

    const float* pEnd1 = pVect1 + 16 * qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

extern float (*InnerProductSIMD16Ext)(const void*, const void*, const void*);
extern float (*InnerProductSIMD4Ext)(const void*, const void*, const void*);

float
InnerProductDistanceSIMD16ExtResidualsSSE(const void* pVect1v,
                                          const void* pVect2v,
                                          const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float* pVect1 = (float*)pVect1v + qty16;
    float* pVect2 = (float*)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

float
InnerProductDistanceSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return 1.0f - InnerProductSIMD16Ext(pVect1v, pVect2v, qty_ptr);
}

float
InnerProductDistanceSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return 1.0f - InnerProductSIMD4Ext(pVect1v, pVect2v, qty_ptr);
}

float
InnerProductDistanceSIMD4ExtResidualsSSE(const void* pVect1v,
                                         const void* pVect2v,
                                         const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float* pVect1 = (float*)pVect1v + qty4;
    float* pVect2 = (float*)pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}

void
PQDistanceSSEFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
    const float* float_centers = (const float*)single_dim_centers;
    float* float_result = (float*)result;
    for (size_t idx = 0; idx < 256; idx += 4) {
        __m128 v_centers_dim = _mm_loadu_ps(float_centers + idx);
        __m128 v_query_vec = _mm_set1_ps(single_dim_val);
        __m128 v_diff = _mm_sub_ps(v_centers_dim, v_query_vec);
        __m128 v_diff_sq = _mm_mul_ps(v_diff, v_diff);
        __m128 v_chunk_dists = _mm_loadu_ps(&float_result[idx]);
        v_chunk_dists = _mm_add_ps(v_chunk_dists, v_diff_sq);
        _mm_storeu_ps(&float_result[idx], v_chunk_dists);
    }
}

namespace sse {

#if defined(ENABLE_SSE)

__inline __m128i __attribute__((__always_inline__)) load_4_char(const uint8_t* data) {
    return _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, data[3], data[2], data[1], data[0]);
}

#endif

float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    const int n = dim / 4;
    if (n == 0) {
        return generic::FP32ComputeIP(query, codes, dim);
    }
    // process 4 floats at a time
    __m128 sum = _mm_setzero_ps();  // initialize to 0
    for (int i = 0; i < n; ++i) {
        __m128 a = _mm_loadu_ps(query + i * 4);   // load 4 floats from memory
        __m128 b = _mm_loadu_ps(codes + i * 4);   // load 4 floats from memory
        sum = _mm_add_ps(sum, _mm_mul_ps(a, b));  // accumulate the product
    }
    alignas(16) float result[4];
    _mm_store_ps(result, sum);  // store the accumulated result into an array
    float ip = result[0] + result[1] + result[2] +
               result[3];  // calculate the sum of the accumulated results
    ip += generic::FP32ComputeIP(query + n * 4, codes + n * 4, dim - n * 4);
    return ip;
#else
    return vsag::Generic::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    const uint64_t n = dim / 4;
    if (n == 0) {
        return generic::FP32ComputeL2Sqr(query, codes, dim);
    }
    __m128 sum = _mm_setzero_ps();
    for (int i = 0; i < n; ++i) {
        __m128 a = _mm_loadu_ps(query + i * 4);
        __m128 b = _mm_loadu_ps(codes + i * 4);
        __m128 diff = _mm_sub_ps(a, b);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    alignas(16) float result[4];
    _mm_store_ps(result, sum);
    float l2 = result[0] + result[1] + result[2] + result[3];
    l2 += generic::FP32ComputeL2Sqr(query + n * 4, codes + n * 4, dim - n * 4);
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
#if defined(ENABLE_SSE)
    // Initialize the sum to 0
    __m128 sum = _mm_setzero_ps();

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        // Load data into registers
        __m128i code_values = load_4_char(codes + i);
        __m128 code_floats = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(code_values));
        __m128 query_values = _mm_loadu_ps(query + i);
        __m128 diff_values = _mm_loadu_ps(diff + i);
        __m128 lowerBound_values = _mm_loadu_ps(lowerBound + i);

        // Perform calculations
        __m128 scaled_codes = _mm_mul_ps(_mm_div_ps(code_floats, _mm_set1_ps(255.0f)), diff_values);
        __m128 adjusted_codes = _mm_add_ps(scaled_codes, lowerBound_values);
        __m128 val = _mm_mul_ps(query_values, adjusted_codes);
        sum = _mm_add_ps(sum, val);
    }

    // Horizontal addition
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    // Extract the result from the register
    alignas(16) float result[4];
    _mm_store_ps(result, sum);

    return result[0] +
           generic::SQ8ComputeIP(query + i, codes + i, lowerBound + i, diff + i, dim - i);
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
#if defined(ENABLE_SSE)
    __m128 sum = _mm_setzero_ps();

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        // Load data into registers
        __m128i code_values = _mm_cvtepu8_epi32(load_4_char(codes + i));
        __m128 code_floats = _mm_div_ps(_mm_cvtepi32_ps(code_values), _mm_set1_ps(255.0f));
        __m128 diff_values = _mm_loadu_ps(diff + i);
        __m128 lowerBound_values = _mm_loadu_ps(lowerBound + i);
        __m128 query_values = _mm_loadu_ps(query + i);

        // Perform calculations
        __m128 scaled_codes = _mm_mul_ps(code_floats, diff_values);
        scaled_codes = _mm_add_ps(scaled_codes, lowerBound_values);
        __m128 val = _mm_sub_ps(query_values, scaled_codes);
        val = _mm_mul_ps(val, val);
        sum = _mm_add_ps(sum, val);
    }
    // Perform horizontal addition
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    // Extract the result from the register
    float result;
    _mm_store_ss(&result, sum);

    result += generic::SQ8ComputeL2Sqr(query + i, codes + i, lowerBound + i, diff + i, dim - i);

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
#if defined(ENABLE_SSE)
    __m128 sum = _mm_setzero_ps();
    uint64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        // Load data into registers
        __m128i code1_values = load_4_char(codes1 + i);
        __m128i code2_values = load_4_char(codes2 + i);
        __m128i codes1_128 = _mm_cvtepu8_epi32(code1_values);
        __m128i codes2_128 = _mm_cvtepu8_epi32(code2_values);
        __m128 codes1_floats = _mm_div_ps(_mm_cvtepi32_ps(codes1_128), _mm_set1_ps(255.0f));
        __m128 codes2_floats = _mm_div_ps(_mm_cvtepi32_ps(codes2_128), _mm_set1_ps(255.0f));
        __m128 diff_values = _mm_loadu_ps(diff + i);
        __m128 lowerBound_values = _mm_loadu_ps(lowerBound + i);
        // Perform calculations
        __m128 scaled_codes1 =
            _mm_add_ps(_mm_mul_ps(codes1_floats, diff_values), lowerBound_values);
        __m128 scaled_codes2 =
            _mm_add_ps(_mm_mul_ps(codes2_floats, diff_values), lowerBound_values);
        __m128 val = _mm_mul_ps(scaled_codes1, scaled_codes2);
        sum = _mm_add_ps(sum, val);
    }
    // Horizontal addition
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    // Extract the result from the register
    float result;
    _mm_store_ss(&result, sum);
    result += generic::SQ8ComputeCodesIP(codes1 + i, codes2 + i, lowerBound + i, diff + i, dim - i);
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
#if defined(ENABLE_SSE)
    __m128 sum = _mm_setzero_ps();
    uint64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        // Load data into registers
        __m128i code1_values = load_4_char(codes1 + i);
        __m128i code2_values = load_4_char(codes2 + i);
        __m128i codes1_128 = _mm_cvtepu8_epi32(code1_values);
        __m128i codes2_128 = _mm_cvtepu8_epi32(code2_values);
        __m128 codes1_floats = _mm_div_ps(_mm_cvtepi32_ps(codes1_128), _mm_set1_ps(255.0f));
        __m128 codes2_floats = _mm_div_ps(_mm_cvtepi32_ps(codes2_128), _mm_set1_ps(255.0f));
        __m128 diff_values = _mm_loadu_ps(diff + i);
        __m128 lowerBound_values = _mm_loadu_ps(lowerBound + i);
        // Perform calculations
        __m128 scaled_codes1 =
            _mm_add_ps(_mm_mul_ps(codes1_floats, diff_values), lowerBound_values);
        __m128 scaled_codes2 =
            _mm_add_ps(_mm_mul_ps(codes2_floats, diff_values), lowerBound_values);
        __m128 val = _mm_sub_ps(scaled_codes1, scaled_codes2);
        val = _mm_mul_ps(val, val);
        sum = _mm_add_ps(sum, val);
    }
    // Horizontal addition
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    // Extract the result from the register
    float result;
    _mm_store_ss(&result, sum);
    result +=
        generic::SQ8ComputeCodesL2Sqr(codes1 + i, codes2 + i, lowerBound + i, diff + i, dim - i);
    return result;
#else
    return Generic::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
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
#if defined(ENABLE_SSE)
    if (dim == 0) {
        return 0;
    }
    alignas(128) int16_t temp[8];
    int32_t result = 0;
    uint64_t d = 0;
    __m128i sum = _mm_setzero_si128();
    __m128i mask = _mm_set1_epi8(0xf);
    for (; d + 31 < dim; d += 32) {
        auto xx = _mm_loadu_si128((__m128i*)(codes1 + (d >> 1)));
        auto yy = _mm_loadu_si128((__m128i*)(codes2 + (d >> 1)));
        auto xx1 = _mm_and_si128(xx, mask);                     // 16 * 8bits
        auto xx2 = _mm_and_si128(_mm_srli_epi16(xx, 4), mask);  // 16 * 8bits
        auto yy1 = _mm_and_si128(yy, mask);
        auto yy2 = _mm_and_si128(_mm_srli_epi16(yy, 4), mask);

        sum = _mm_add_epi16(sum, _mm_maddubs_epi16(xx1, yy1));
        sum = _mm_add_epi16(sum, _mm_maddubs_epi16(xx2, yy2));
    }
    _mm_store_si128((__m128i*)temp, sum);
    for (int i = 0; i < 8; ++i) {
        result += temp[i];
    }
    result += generic::SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);
    return result;
#else
    return generic::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_SSE)
    if (dim == 0) {
        return;
    }
    if (scalar == 0) {
        scalar = 1.0f;  // TODO(LHT): logger?
    }
    int i = 0;
    __m128 scalarVec = _mm_set1_ps(scalar);
    for (; i + 3 < dim; i += 4) {
        __m128 vec = _mm_loadu_ps(from + i);
        vec = _mm_div_ps(vec, scalarVec);
        _mm_storeu_ps(to + i, vec);
    }
    generic::DivScalar(from + i, to + i, dim - i, scalar);
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
}  // namespace sse

}  // namespace vsag
