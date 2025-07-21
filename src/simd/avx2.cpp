
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

#if defined(ENABLE_AVX2)
#include <immintrin.h>
#endif

#include <cmath>
#include <cstdint>

#include "simd.h"

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace vsag::avx2 {

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
    return 1.0f - avx2::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return avx::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);  // TODO(LHT): implement
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -avx2::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_AVX2)
    auto* float_centers = (const float*)single_dim_centers;
    auto* float_result = (float*)result;
    for (uint64_t idx = 0; idx < 256; idx += 8) {
        __m256 v_centers_dim = _mm256_loadu_ps(float_centers + idx);
        __m256 v_query_vec = _mm256_set1_ps(single_dim_val);
        __m256 v_diff = _mm256_sub_ps(v_centers_dim, v_query_vec);
        __m256 v_diff_sq = _mm256_mul_ps(v_diff, v_diff);
        __m256 v_chunk_dists = _mm256_loadu_ps(&float_result[idx]);
        v_chunk_dists = _mm256_add_ps(v_chunk_dists, v_diff_sq);
        _mm256_storeu_ps(&float_result[idx], v_chunk_dists);
    }
#else
    return avx::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

#if defined(ENABLE_AVX2)
__inline __m128i __attribute__((__always_inline__)) load_8_char(const uint8_t* data) {
    return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(data));
}
#endif

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    const int n = dim / 8;
    if (n == 0) {
        return avx::FP32ComputeIP(query, codes, dim);
    }
    // process 8 floats at a time
    __m256 sum = _mm256_setzero_ps();  // initialize to 0
    for (int i = 0; i < n; ++i) {
        __m256 a = _mm256_loadu_ps(query + i * 8);      // load 8 floats from memory
        __m256 b = _mm256_loadu_ps(codes + i * 8);      // load 8 floats from memory
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));  // accumulate the product
    }
    alignas(32) float result[8];
    _mm256_store_ps(result, sum);  // store the accumulated result into an array
    float ip = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];  // calculate the sum of the accumulated results
    ip += avx::FP32ComputeIP(query + n * 8, codes + n * 8, dim - n * 8);
    return ip;
#else
    return avx::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    const int n = dim / 8;
    if (n == 0) {
        return avx::FP32ComputeL2Sqr(query, codes, dim);
    }
    // process 8 floats at a time
    __m256 sum = _mm256_setzero_ps();  // initialize to 0
    for (int i = 0; i < n; ++i) {
        __m256 a = _mm256_loadu_ps(query + i * 8);  // load 8 floats from memory
        __m256 b = _mm256_loadu_ps(codes + i * 8);  // load 8 floats from memory
        __m256 diff = _mm256_sub_ps(a, b);          // calculate the difference
        sum = _mm256_fmadd_ps(diff, diff, sum);     // accumulate the squared difference
    }
    alignas(32) float result[8];
    _mm256_store_ps(result, sum);  // store the accumulated result into an array
    float l2 = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];  // calculate the sum of the accumulated results
    l2 += avx::FP32ComputeL2Sqr(query + n * 8, codes + n * 8, dim - n * 8);
    return l2;
#else
    return avx::FP32ComputeL2Sqr(query, codes, dim);
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
    if (dim < 8) {
        return avx::FP32ComputeIPBatch4(
            query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
    }

    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query + i);
        __m256 c1 = _mm256_loadu_ps(codes1 + i);
        __m256 c2 = _mm256_loadu_ps(codes2 + i);
        __m256 c3 = _mm256_loadu_ps(codes3 + i);
        __m256 c4 = _mm256_loadu_ps(codes4 + i);
        sum1 = _mm256_fmadd_ps(q, c1, sum1);
        sum2 = _mm256_fmadd_ps(q, c2, sum2);
        sum3 = _mm256_fmadd_ps(q, c3, sum3);
        sum4 = _mm256_fmadd_ps(q, c4, sum4);
    }
    alignas(32) float result[8];
    _mm256_store_ps(result, sum1);
    result1 += result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];
    _mm256_store_ps(result, sum2);
    result2 += result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];
    _mm256_store_ps(result, sum3);
    result3 += result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];
    _mm256_store_ps(result, sum4);
    result4 += result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];
    if (i < dim) {
        avx::FP32ComputeIPBatch4(query + i,
                                 dim - i,
                                 codes1 + i,
                                 codes2 + i,
                                 codes3 + i,
                                 codes4 + i,
                                 result1,
                                 result2,
                                 result3,
                                 result4);
    }
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
    if (dim < 8) {
        return sse::FP32ComputeL2SqrBatch4(
            query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
    }
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query + i);
        __m256 c1 = _mm256_loadu_ps(codes1 + i);
        __m256 c2 = _mm256_loadu_ps(codes2 + i);
        __m256 c3 = _mm256_loadu_ps(codes3 + i);
        __m256 c4 = _mm256_loadu_ps(codes4 + i);
        __m256 diff1 = _mm256_sub_ps(q, c1);
        __m256 diff2 = _mm256_sub_ps(q, c2);
        __m256 diff3 = _mm256_sub_ps(q, c3);
        __m256 diff4 = _mm256_sub_ps(q, c4);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        sum4 = _mm256_fmadd_ps(diff4, diff4, sum4);
    }
    alignas(32) float result[8];
    _mm256_store_ps(result, sum1);
    result1 += result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];
    _mm256_store_ps(result, sum2);
    result2 += result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];
    _mm256_store_ps(result, sum3);
    result3 += result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];
    _mm256_store_ps(result, sum4);
    result4 += result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];
    if (i < dim) {
        avx::FP32ComputeL2SqrBatch4(query + i,
                                    dim - i,
                                    codes1 + i,
                                    codes2 + i,
                                    codes3 + i,
                                    codes4 + i,
                                    result1,
                                    result2,
                                    result3,
                                    result4);
    }
#else
    return avx::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX2)
    if (dim < 8) {
        return sse::FP32Sub(x, y, z, dim);
    }
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 a = _mm256_loadu_ps(x + i);
        __m256 b = _mm256_loadu_ps(y + i);
        __m256 c = _mm256_sub_ps(a, b);
        _mm256_storeu_ps(z + i, c);
    }
    if (i < dim) {
        sse::FP32Sub(x + i, y + i, z + i, dim - i);
    }
#else
    return sse::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX2)
    if (dim < 8) {
        return sse::FP32Add(x, y, z, dim);
    }
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 a = _mm256_loadu_ps(x + i);
        __m256 b = _mm256_loadu_ps(y + i);
        __m256 c = _mm256_add_ps(a, b);
        _mm256_storeu_ps(z + i, c);
    }
    if (i < dim) {
        sse::FP32Add(x + i, y + i, z + i, dim - i);
    }
#else
    return sse::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX2)
    if (dim < 8) {
        return sse::FP32Mul(x, y, z, dim);
    }
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 a = _mm256_loadu_ps(x + i);
        __m256 b = _mm256_loadu_ps(y + i);
        __m256 c = _mm256_mul_ps(a, b);
        _mm256_storeu_ps(z + i, c);
    }
    if (i < dim) {
        sse::FP32Mul(x + i, y + i, z + i, dim - i);
    }
#else
    return sse::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX2)
    if (dim < 8) {
        return sse::FP32Div(x, y, z, dim);
    }
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 a = _mm256_loadu_ps(x + i);
        __m256 b = _mm256_loadu_ps(y + i);
        __m256 c = _mm256_div_ps(a, b);
        _mm256_storeu_ps(z + i, c);
    }
    if (i < dim) {
        sse::FP32Div(x + i, y + i, z + i, dim - i);
    }
#else
    return sse::FP32Div(x, y, z, dim);
#endif
}
float
FP32ReduceAdd(const float* x, uint64_t dim) {
    return sse::FP32ReduceAdd(x, dim);
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
    // Initialize the sum to 0
    __m256 sum = _mm256_setzero_ps();
    const auto* query_bf16 = (const uint16_t*)(query);
    const auto* codes_bf16 = (const uint16_t*)(codes);

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 7 < dim; i += 8) {
        // Load data into registers
        __m256i query_shift = load_8_short(query_bf16 + i);
        __m256 query_float = _mm256_castsi256_ps(query_shift);

        // Load data into registers
        __m256i code_shift = load_8_short(codes_bf16 + i);
        __m256 code_float = _mm256_castsi256_ps(code_shift);

        sum = _mm256_fmadd_ps(code_float, query_float, sum);
    }

    alignas(32) float result[8];
    _mm256_store_ps(result, sum);  // store the accumulated result into an array
    float ip = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];  // calculate the sum of the accumulated results

    return ip + avx::BF16ComputeIP(query + i * 2, codes + i * 2, dim - i);
#else
    return avx::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    // Initialize the sum to 0
    __m256 sum = _mm256_setzero_ps();
    const auto* query_bf16 = (const uint16_t*)(query);
    const auto* codes_bf16 = (const uint16_t*)(codes);

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 7 < dim; i += 8) {
        // Load data into registers
        __m256i query_shift = load_8_short(query_bf16 + i);
        __m256 query_float = _mm256_castsi256_ps(query_shift);

        // Load data into registers
        __m256i code_shift = load_8_short(codes_bf16 + i);
        __m256 code_float = _mm256_castsi256_ps(code_shift);

        __m256 diff = _mm256_sub_ps(code_float, query_float);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    alignas(32) float result[8];
    _mm256_store_ps(result, sum);  // store the accumulated result into an array
    float l2 = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];  // calculate the sum of the accumulated results

    return l2 + avx::BF16ComputeL2Sqr(query + i * 2, codes + i * 2, dim - i);
#else
    return avx::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    // Initialize the sum to 0
    __m256 sum = _mm256_setzero_ps();
    const auto* query_fp16 = (const uint16_t*)(query);
    const auto* codes_fp16 = (const uint16_t*)(codes);

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 7 < dim; i += 8) {
        // Load data into registers
        __m128i query_load = _mm_loadu_si128(reinterpret_cast<const __m128i*>(query_fp16 + i));
        __m256 query_float = _mm256_cvtph_ps(query_load);

        // Load data into registers
        __m128i code_load = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes_fp16 + i));
        __m256 code_float = _mm256_cvtph_ps(code_load);

        sum = _mm256_fmadd_ps(code_float, query_float, sum);
    }

    alignas(32) float result[8];
    _mm256_store_ps(result, sum);  // store the accumulated result into an array
    float ip = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];  // calculate the sum of the accumulated results

    return ip + avx::FP16ComputeIP(query + i * 2, codes + i * 2, dim - i);
#else
    return avx::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    // Initialize the sum to 0
    __m256 sum = _mm256_setzero_ps();
    const auto* query_fp16 = (const uint16_t*)(query);
    const auto* codes_fp16 = (const uint16_t*)(codes);

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 7 < dim; i += 8) {
        // Load data into registers
        __m128i query_load = _mm_loadu_si128(reinterpret_cast<const __m128i*>(query_fp16 + i));
        __m256 query_float = _mm256_cvtph_ps(query_load);

        // Load data into registers
        __m128i code_load = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes_fp16 + i));
        __m256 code_float = _mm256_cvtph_ps(code_load);

        __m256 diff = _mm256_sub_ps(code_float, query_float);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    alignas(32) float result[8];
    _mm256_store_ps(result, sum);  // store the accumulated result into an array
    float l2 = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] +
               result[7];  // calculate the sum of the accumulated results

    return l2 + avx::FP16ComputeL2Sqr(query + i * 2, codes + i * 2, dim - i);
#else
    return avx::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_AVX2)
    __m256 sum = _mm256_setzero_ps();
    uint64_t i = 0;

    for (; i + 7 < dim; i += 8) {
        __m128i code_values = load_8_char(codes + i);
        __m256 code_floats = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(code_values));
        __m256 query_values = _mm256_loadu_ps(query + i);
        __m256 diff_values = _mm256_loadu_ps(diff + i);
        __m256 lower_bound_values = _mm256_loadu_ps(lower_bound + i);

        __m256 scaled_codes =
            _mm256_mul_ps(_mm256_div_ps(code_floats, _mm256_set1_ps(255.0f)), diff_values);
        __m256 adjusted_codes = _mm256_add_ps(scaled_codes, lower_bound_values);
        __m256 val = _mm256_mul_ps(query_values, adjusted_codes);
        sum = _mm256_add_ps(sum, val);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_final = _mm_add_ps(sum_low, sum_high);

    alignas(16) float result[4];
    _mm_store_ps(result, sum_final);
    float finalResult = result[0] + result[1] + result[2] + result[3];

    // Process the remaining elements recursively
    finalResult += avx::SQ8ComputeIP(query + i, codes + i, lower_bound + i, diff + i, dim - i);
    return finalResult;
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
    __m256 sum = _mm256_setzero_ps();
    uint64_t i = 0;

    for (; i + 7 < dim; i += 8) {
        // Load data into registers
        __m256i code_values = _mm256_cvtepu8_epi32(load_8_char(codes + i));
        __m256 code_floats = _mm256_div_ps(_mm256_cvtepi32_ps(code_values), _mm256_set1_ps(255.0f));
        __m256 diff_values = _mm256_loadu_ps(diff + i);
        __m256 lower_bound_values = _mm256_loadu_ps(lower_bound + i);
        __m256 query_values = _mm256_loadu_ps(query + i);

        // Perform calculations
        __m256 scaled_codes = _mm256_mul_ps(code_floats, diff_values);
        scaled_codes = _mm256_add_ps(scaled_codes, lower_bound_values);
        __m256 val = _mm256_sub_ps(query_values, scaled_codes);
        val = _mm256_mul_ps(val, val);
        sum = _mm256_add_ps(sum, val);
    }

    // Horizontal addition
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_final = _mm_add_ps(sum_low, sum_high);
    sum_final = _mm_hadd_ps(sum_final, sum_final);
    sum_final = _mm_hadd_ps(sum_final, sum_final);

    // Extract the result from the register
    float result;
    _mm_store_ss(&result, sum_final);

    // Process the remaining elements
    result += avx::SQ8ComputeL2Sqr(query + i, codes + i, lower_bound + i, diff + i, dim - i);
    return result;
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
    __m256 sum = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 7 < dim; i += 8) {
        // Load data into registers
        __m128i code1_values = load_8_char(codes1 + i);
        __m128i code2_values = load_8_char(codes2 + i);
        __m256i codes1_256 = _mm256_cvtepu8_epi32(code1_values);
        __m256i codes2_256 = _mm256_cvtepu8_epi32(code2_values);
        __m256 code1_floats = _mm256_div_ps(_mm256_cvtepi32_ps(codes1_256), _mm256_set1_ps(255.0f));
        __m256 code2_floats = _mm256_div_ps(_mm256_cvtepi32_ps(codes2_256), _mm256_set1_ps(255.0f));
        __m256 diff_values = _mm256_loadu_ps(diff + i);
        __m256 lower_bound_values = _mm256_loadu_ps(lower_bound + i);
        // Perform calculations
        __m256 scaled_codes1 = _mm256_fmadd_ps(code1_floats, diff_values, lower_bound_values);
        __m256 scaled_codes2 = _mm256_fmadd_ps(code2_floats, diff_values, lower_bound_values);
        __m256 val = _mm256_mul_ps(scaled_codes1, scaled_codes2);
        sum = _mm256_add_ps(sum, val);
    }

    // Horizontal addition
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_final = _mm_add_ps(sum_low, sum_high);
    sum_final = _mm_hadd_ps(sum_final, sum_final);
    sum_final = _mm_hadd_ps(sum_final, sum_final);

    // Extract the result from the register
    float result;
    _mm_store_ss(&result, sum_final);

    result += avx::SQ8ComputeCodesIP(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
    return result;
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
    __m256 sum = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 7 < dim; i += 8) {
        // Load data into registers
        __m256i code1_values = _mm256_cvtepu8_epi32(load_8_char(codes1 + i));
        __m256i code2_values = _mm256_cvtepu8_epi32(load_8_char(codes2 + i));
        __m256 codes1_floats =
            _mm256_div_ps(_mm256_cvtepi32_ps(code1_values), _mm256_set1_ps(255.0f));
        __m256 codes2_floats =
            _mm256_div_ps(_mm256_cvtepi32_ps(code2_values), _mm256_set1_ps(255.0f));
        __m256 diff_values = _mm256_loadu_ps(diff + i);
        __m256 lower_bound_values = _mm256_loadu_ps(lower_bound + i);
        // Perform calculations
        __m256 scaled_codes1 = _mm256_fmadd_ps(codes1_floats, diff_values, lower_bound_values);
        __m256 scaled_codes2 = _mm256_fmadd_ps(codes2_floats, diff_values, lower_bound_values);
        __m256 val = _mm256_sub_ps(scaled_codes1, scaled_codes2);
        val = _mm256_mul_ps(val, val);
        sum = _mm256_add_ps(sum, val);
    }
    // Horizontal addition
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_final = _mm_add_ps(sum_low, sum_high);
    sum_final = _mm_hadd_ps(sum_final, sum_final);
    sum_final = _mm_hadd_ps(sum_final, sum_final);
    // Extract the result from the register
    float result;
    _mm_store_ss(&result, sum_final);

    result += avx::SQ8ComputeCodesL2Sqr(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
    return result;
#else
    return avx::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
    return avx::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
    return avx::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
    return avx::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
    return avx::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX2)
    if (dim == 0) {
        return 0;
    }
    alignas(256) int16_t temp[16];
    int32_t result = 0;
    uint64_t d = 0;
    __m256i sum = _mm256_setzero_si256();
    __m256i mask = _mm256_set1_epi8(0xf);
    for (; d + 63 < dim; d += 64) {
        auto xx = _mm256_loadu_si256((__m256i*)(codes1 + (d >> 1)));
        auto yy = _mm256_loadu_si256((__m256i*)(codes2 + (d >> 1)));
        auto xx1 = _mm256_and_si256(xx, mask);                        // 32 * 8bits
        auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);  // 32 * 8bits
        auto yy1 = _mm256_and_si256(yy, mask);
        auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);

        sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(xx1, yy1));
        sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(xx2, yy2));
    }
    _mm256_store_si256((__m256i*)temp, sum);
    for (int i = 0; i < 16; ++i) {
        result += temp[i];
    }
    result += avx::SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);
    return result;
#else
    return avx::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX2)
    if (dim == 0) {
        return 0.0f;
    }

    alignas(32) int32_t temp[8];
    int32_t result = 0;
    uint64_t d = 0;
    __m256i sum = _mm256_setzero_si256();
    __m256i mask = _mm256_set1_epi16(0xff);
    for (; d + 31 < dim; d += 32) {
        auto xx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes1 + d));
        auto yy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes2 + d));

        auto xx1 = _mm256_and_si256(xx, mask);
        auto xx2 = _mm256_srli_epi16(xx, 8);
        auto yy1 = _mm256_and_si256(yy, mask);
        auto yy2 = _mm256_srli_epi16(yy, 8);

        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(xx1, yy1));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(xx2, yy2));
    }
    _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum);
    for (int i : temp) {
        result += i;
    }
    result += static_cast<int32_t>(avx::SQ8UniformComputeCodesIP(codes1 + d, codes2 + d, dim - d));
    return static_cast<float>(result);
#else
    return avx::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_AVX2)
    if (dim == 0) {
        return 0.0f;
    }

    if (dim < 8) {
        return sse::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
    }

    uint64_t d = 0;
    float result = 0.0f;
    alignas(32) float temp[8];
    __m256 sum = _mm256_setzero_ps();
    const __m256 inv_sqrt_d_vec = _mm256_set1_ps(inv_sqrt_d);
    const __m256 neg_inv_sqrt_d_vec = _mm256_set1_ps(-inv_sqrt_d);

    for (; d + 8 <= dim; d += 8) {
        __m256 vec = _mm256_loadu_ps(vector + d);

        __m256i mask = _mm256_set1_epi32(static_cast<int>(bits[d / 8]));
        mask = _mm256_and_si256(mask,
                                _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80));
        mask = _mm256_cmpeq_epi32(mask, _mm256_setzero_si256());
        mask = _mm256_andnot_si256(mask, _mm256_set1_epi32(0xFFFFFFFF));

        __m256 b_vec =
            _mm256_blendv_ps(neg_inv_sqrt_d_vec, inv_sqrt_d_vec, _mm256_castsi256_ps(mask));

        sum = _mm256_fmadd_ps(b_vec, vec, sum);
    }

    _mm256_storeu_ps(temp, sum);
    for (float val : temp) {
        result += val;
    }

    result += avx::RaBitQFloatBinaryIP(vector + d, bits + d / 8, dim - d, inv_sqrt_d);

    return result;
#else
    return avx::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
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
    __m256 scalarVec = _mm256_set1_ps(scalar);
    for (; i + 7 < dim; i += 8) {
        __m256 vec = _mm256_loadu_ps(from + i);
        vec = _mm256_div_ps(vec, scalarVec);
        _mm256_storeu_ps(to + i, vec);
    }
    avx::DivScalar(from + i, to + i, dim - i, scalar);
#else
    avx::DivScalar(from, to, dim, scalar);
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
    for (size_t i = 0; i < 4; i++) {
        sum[i] = _mm256_setzero_si256();
    }
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
    }
    alignas(256) uint16_t temp[16];
    for (int64_t idx = 0; idx < 4; idx++) {
        _mm256_store_si256((__m256i*)(temp), sum[idx]);
        for (int64_t j = 0; j < 8; j++) {
            result[idx * 8 + j] += temp[j] + temp[j + 8];
        }
    }
    if (pq_dim > i) {
        avx::PQFastScanLookUp32(lookup_table, codes, pq_dim - i, result);
    }
#else
    avx::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX2)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 32) {
        return sse::BitAnd(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 31 < num_byte; i += 32) {
        __m256i x_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        __m256i y_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i));
        __m256i z_vec = _mm256_and_si256(x_vec, y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i), z_vec);
    }
    if (i < num_byte) {
        sse::BitAnd(x + i, y + i, num_byte - i, result + i);
    }
#else
    return sse::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX2)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 32) {
        return sse::BitOr(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 31 < num_byte; i += 32) {
        __m256i x_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        __m256i y_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i));
        __m256i z_vec = _mm256_or_si256(x_vec, y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i), z_vec);
    }
    if (i < num_byte) {
        sse::BitOr(x + i, y + i, num_byte - i, result + i);
    }
#else
    return sse::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX2)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 32) {
        return sse::BitXor(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 31 < num_byte; i += 32) {
        __m256i x_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        __m256i y_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i));
        __m256i z_vec = _mm256_xor_si256(x_vec, y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i), z_vec);
    }
    if (i < num_byte) {
        sse::BitXor(x + i, y + i, num_byte - i, result + i);
    }
#else
    return sse::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX2)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 32) {
        return sse::BitNot(x, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 31 < num_byte; i += 32) {
        __m256i x_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        __m256i z_vec = _mm256_xor_si256(x_vec, _mm256_set1_epi8(0xFF));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i), z_vec);
    }
    if (i < num_byte) {
        sse::BitNot(x + i, num_byte - i, result + i);
    }
#else
    return sse::BitNot(x, num_byte, result);
#endif
}

void
VecRescale(float* data, size_t dim, float val) {
#if defined(ENABLE_AVX2)
    int i = 0;
    __m256 val_vec = _mm256_set1_ps(val);
    for (; i + 8 < dim; i += 8) {
        __m256 data_vec = _mm256_loadu_ps(&data[i]);
        __m256 result_vec = _mm256_mul_ps(data_vec, val_vec);
        _mm256_storeu_ps(&data[i], result_vec);
    }

    sse::VecRescale(data + i, dim - i, val);
#else
    return avx::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_AVX2)
    for (int i = idx; i < dim_; i += step * 2) {
        for (int j = 0; j < step; j += 8) {
            __m256 g1 = _mm256_loadu_ps(&data[i + j]);
            __m256 g2 = _mm256_loadu_ps(&data[i + j + step]);
            _mm256_storeu_ps(&data[i + j], _mm256_add_ps(g1, g2));
            _mm256_storeu_ps(&data[i + j + step], _mm256_sub_ps(g1, g2));
        }
    }
#else
    return avx::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, size_t dim_) {
#if defined(ENABLE_AVX2)
    size_t n = dim_;
    size_t step = 1;
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
KacsWalk(float* data, size_t len) {
#if defined(ENABLE_AVX2)
    size_t base = len % 2;
    size_t offset = base + (len / 2);  // for odd dim
    size_t i = 0;
    for (; i + 8 < len / 2; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 y = _mm256_loadu_ps(&data[i + offset]);
        _mm256_storeu_ps(&data[i], _mm256_add_ps(x, y));
        _mm256_storeu_ps(&data[i + offset], _mm256_sub_ps(x, y));
    }
    for (; i < len / 2; i++) {
        float add = data[i] + data[i + offset];
        float sub = data[i] - data[i + offset];
        data[i] = add;
        data[i + offset] = sub;
    }
    if (base != 0) {
        data[len / 2] *= std::sqrt(2.0F);
        //In odd condition, we operate the prev len/2 items and the post len/2 items, the No.len/2 item stay still,
        //As we need to resize the while sequence in the next step, so we increase the val of No.len/2 item to eliminate the impact of the following resize.
    }
#else
    return avx::KacsWalk(data, len);
#endif
}
}  // namespace vsag::avx2
