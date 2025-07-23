
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
#if defined(ENABLE_AVX512)
#include <immintrin.h>
#endif

#include <cmath>

#include "simd.h"

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace vsag::avx512 {
float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* float_vec1 = (float*)pVect1v;
    auto* float_vec2 = (float*)pVect2v;
    size_t dim = *((size_t*)qty_ptr);
    return avx512::FP32ComputeL2Sqr(float_vec1, float_vec2, dim);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* float_vec1 = (float*)pVect1v;
    auto* float_vec2 = (float*)pVect2v;
    size_t dim = *((size_t*)qty_ptr);
    return avx512::FP32ComputeIP(float_vec1, float_vec2, dim);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - avx512::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(ENABLE_AVX512)
    __mmask32 mask = 0xFFFFFFFF;

    auto qty = *((size_t*)qty_ptr);
    const uint32_t n = (qty >> 5);
    if (n == 0) {
        return avx2::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
    }

    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;

    __m512i sum512 = _mm512_set1_epi32(0);
    for (uint32_t i = 0; i < n; ++i) {
        __m256i v1 = _mm256_maskz_loadu_epi8(mask, pVect1 + (i << 5));
        __m512i v1_512 = _mm512_cvtepi8_epi16(v1);
        __m256i v2 = _mm256_maskz_loadu_epi8(mask, pVect2 + (i << 5));
        __m512i v2_512 = _mm512_cvtepi8_epi16(v2);
        sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v1_512, v2_512));
    }
    auto res = static_cast<float>(_mm512_reduce_add_epi32(sum512));
    uint64_t new_dim = qty & (0x1F);
    res += avx2::INT8InnerProduct(pVect1 + (n << 5), pVect2 + (n << 5), &new_dim);
    return res;
#else
    return avx2::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
#endif
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -avx512::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
    return avx2::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
}

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim < 16) {
        return avx2::FP32ComputeIP(query, codes, dim);
    }
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    uint64_t i = 0;
    for (; i + 63 < dim; i += 64) {
        __m512 a0 = _mm512_loadu_ps(query + i);
        __m512 b0 = _mm512_loadu_ps(codes + i);

        __m512 a1 = _mm512_loadu_ps(query + i + 16);
        __m512 b1 = _mm512_loadu_ps(codes + i + 16);

        __m512 a2 = _mm512_loadu_ps(query + i + 32);
        __m512 b2 = _mm512_loadu_ps(codes + i + 32);

        __m512 a3 = _mm512_loadu_ps(query + i + 48);
        __m512 b3 = _mm512_loadu_ps(codes + i + 48);

        sum0 = _mm512_fmadd_ps(a0, b0, sum0);
        sum1 = _mm512_fmadd_ps(a1, b1, sum1);
        sum2 = _mm512_fmadd_ps(a2, b2, sum2);
        sum3 = _mm512_fmadd_ps(a3, b3, sum3);
    }
    __m512 sum = _mm512_add_ps(sum0, sum1);
    sum = _mm512_add_ps(sum, sum2);
    sum = _mm512_add_ps(sum, sum3);

    for (; i + 15 < dim; i += 16) {
        __m512 a = _mm512_loadu_ps(query + i);  // load 16 floats from memory
        __m512 b = _mm512_loadu_ps(codes + i);  // load 16 floats from memory
        sum = _mm512_fmadd_ps(a, b, sum);
    }
    float ip = _mm512_reduce_add_ps(sum);
    if (dim - i > 0) {
        ip += avx2::FP32ComputeIP(query + i, codes + i, dim - i);
    }
    return ip;
#else
    return avx2::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim < 16) {
        return avx2::FP32ComputeL2Sqr(query, codes, dim);
    }

    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    uint64_t i = 0;
    for (; i + 63 < dim; i += 64) {
        __m512 a0 = _mm512_loadu_ps(query + i);
        __m512 b0 = _mm512_loadu_ps(codes + i);

        __m512 a1 = _mm512_loadu_ps(query + i + 16);
        __m512 b1 = _mm512_loadu_ps(codes + i + 16);

        __m512 a2 = _mm512_loadu_ps(query + i + 32);
        __m512 b2 = _mm512_loadu_ps(codes + i + 32);

        __m512 a3 = _mm512_loadu_ps(query + i + 48);
        __m512 b3 = _mm512_loadu_ps(codes + i + 48);

        __m512 diff0 = _mm512_sub_ps(a0, b0);
        __m512 diff1 = _mm512_sub_ps(a1, b1);
        __m512 diff2 = _mm512_sub_ps(a2, b2);
        __m512 diff3 = _mm512_sub_ps(a3, b3);

        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
    }

    __m512 sum = _mm512_add_ps(sum0, sum1);
    sum = _mm512_add_ps(sum, sum2);
    sum = _mm512_add_ps(sum, sum3);

    for (; i + 15 < dim; i += 16) {
        __m512 a = _mm512_loadu_ps(query + i);
        __m512 b = _mm512_loadu_ps(codes + i);
        __m512 diff = _mm512_sub_ps(a, b);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    float l2 = _mm512_reduce_add_ps(sum);
    if (dim > i) {
        l2 += avx2::FP32ComputeL2Sqr(query + i, codes + i, dim - i);
    }
    return l2;
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
    if (dim < 16) {
        return avx2::FP32ComputeIPBatch4(
            query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
    }

    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 q = _mm512_loadu_ps(query + i);
        __m512 c1 = _mm512_loadu_ps(codes1 + i);
        __m512 c2 = _mm512_loadu_ps(codes2 + i);
        __m512 c3 = _mm512_loadu_ps(codes3 + i);
        __m512 c4 = _mm512_loadu_ps(codes4 + i);

        sum1 = _mm512_fmadd_ps(q, c1, sum1);
        sum2 = _mm512_fmadd_ps(q, c2, sum2);
        sum3 = _mm512_fmadd_ps(q, c3, sum3);
        sum4 = _mm512_fmadd_ps(q, c4, sum4);
    }
    result1 += _mm512_reduce_add_ps(sum1);
    result2 += _mm512_reduce_add_ps(sum2);
    result3 += _mm512_reduce_add_ps(sum3);
    result4 += _mm512_reduce_add_ps(sum4);
    if (dim - i > 0) {
        avx2::FP32ComputeIPBatch4(query + i,
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
    if (dim < 16) {
        return avx2::FP32ComputeL2SqrBatch4(
            query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
    }
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 q = _mm512_loadu_ps(query + i);
        __m512 c1 = _mm512_loadu_ps(codes1 + i);
        __m512 c2 = _mm512_loadu_ps(codes2 + i);
        __m512 c3 = _mm512_loadu_ps(codes3 + i);
        __m512 c4 = _mm512_loadu_ps(codes4 + i);
        __m512 diff1 = _mm512_sub_ps(q, c1);
        __m512 diff2 = _mm512_sub_ps(q, c2);
        __m512 diff3 = _mm512_sub_ps(q, c3);
        __m512 diff4 = _mm512_sub_ps(q, c4);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
        sum4 = _mm512_fmadd_ps(diff4, diff4, sum4);
    }
    result1 += _mm512_reduce_add_ps(sum1);
    result2 += _mm512_reduce_add_ps(sum2);
    result3 += _mm512_reduce_add_ps(sum3);
    result4 += _mm512_reduce_add_ps(sum4);
    if (dim - i > 0) {
        avx2::FP32ComputeL2SqrBatch4(query + i,
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
#if defined(ENABLE_AVX512)
    if (dim < 16) {
        return avx2::FP32Sub(x, y, z, dim);
    }
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_loadu_ps(y + i);
        __m512 diff_vec = _mm512_sub_ps(x_vec, y_vec);
        _mm512_storeu_ps(z + i, diff_vec);
    }
    if (dim > i) {
        avx2::FP32Sub(x + i, y + i, z + i, dim - i);
    }
#else
    return avx2::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim < 16) {
        return avx2::FP32Add(x, y, z, dim);
    }
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_loadu_ps(y + i);
        __m512 sum_vec = _mm512_add_ps(x_vec, y_vec);
        _mm512_storeu_ps(z + i, sum_vec);
    }
    if (dim > i) {
        avx2::FP32Add(x + i, y + i, z + i, dim - i);
    }
#else
    return avx2::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim < 16) {
        return avx2::FP32Mul(x, y, z, dim);
    }
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_loadu_ps(y + i);
        __m512 mul_vec = _mm512_mul_ps(x_vec, y_vec);
        _mm512_storeu_ps(z + i, mul_vec);
    }
    if (dim > i) {
        avx2::FP32Mul(x + i, y + i, z + i, dim - i);
    }
#else
    return avx2::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim < 16) {
        return avx2::FP32Div(x, y, z, dim);
    }
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 y_vec = _mm512_loadu_ps(y + i);
        __m512 div_vec = _mm512_div_ps(x_vec, y_vec);
        _mm512_storeu_ps(z + i, div_vec);
    }
    if (dim > i) {
        avx2::FP32Div(x + i, y + i, z + i, dim - i);
    }
#else
    return avx2::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
    return sse::FP32ReduceAdd(x, dim);
}

#if defined(ENABLE_AVX512)
__inline __m512i __attribute__((__always_inline__)) load_16_short(const uint16_t* data) {
    __m256i bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
    __m512i bf32 = _mm512_cvtepu16_epi32(bf16);
    return _mm512_slli_epi32(bf32, 16);
}
#endif

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    // Initialize the sum to 0
    __m512 sum = _mm512_setzero_ps();
    const auto* query_bf16 = (const uint16_t*)(query);
    const auto* codes_bf16 = (const uint16_t*)(codes);

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m512i query_shift = load_16_short(query_bf16 + i);
        __m512 query_float = _mm512_castsi512_ps(query_shift);

        // Load data into registers
        __m512i code_shift = load_16_short(codes_bf16 + i);
        __m512 code_float = _mm512_castsi512_ps(code_shift);

        sum = _mm512_fmadd_ps(code_float, query_float, sum);
    }
    float ip = _mm512_reduce_add_ps(sum);
    if (dim > i) {
        ip += avx2::BF16ComputeIP(query + i * 2, codes + i * 2, dim - i);
    }
    return ip;
#else
    return avx2::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    // Initialize the sum to 0
    __m512 sum = _mm512_setzero_ps();
    const auto* query_bf16 = (const uint16_t*)(query);
    const auto* codes_bf16 = (const uint16_t*)(codes);

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m512i query_shift = load_16_short(query_bf16 + i);
        __m512 query_float = _mm512_castsi512_ps(query_shift);

        // Load data into registers
        __m512i code_shift = load_16_short(codes_bf16 + i);
        __m512 code_float = _mm512_castsi512_ps(code_shift);

        __m512 diff = _mm512_sub_ps(code_float, query_float);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    float l2 = _mm512_reduce_add_ps(sum);
    if (dim > i) {
        l2 += avx2::BF16ComputeL2Sqr(query + i * 2, codes + i * 2, dim - i);
    }
    return l2;
#else
    return avx2::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    // Initialize the sum to 0
    __m512 sum = _mm512_setzero_ps();
    const auto* query_fp16 = (const uint16_t*)(query);
    const auto* codes_fp16 = (const uint16_t*)(codes);

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m256i query_load = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_fp16 + i));
        __m512 query_float = _mm512_cvtph_ps(query_load);

        // Load data into registers
        __m256i code_load = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_fp16 + i));
        __m512 code_float = _mm512_cvtph_ps(code_load);

        sum = _mm512_fmadd_ps(code_float, query_float, sum);
    }
    float ip = _mm512_reduce_add_ps(sum);
    if (dim > i) {
        ip += avx2::FP16ComputeIP(query + i * 2, codes + i * 2, dim - i);
    }
    return ip;
#else
    return avx2::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    // Initialize the sum to 0
    __m512 sum = _mm512_setzero_ps();
    const auto* query_fp16 = (const uint16_t*)(query);
    const auto* codes_fp16 = (const uint16_t*)(codes);

    // Process the data in 128-bit chunks
    uint64_t i = 0;
    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m256i query_load = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_fp16 + i));
        __m512 query_float = _mm512_cvtph_ps(query_load);

        // Load data into registers
        __m256i code_load = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_fp16 + i));
        __m512 code_float = _mm512_cvtph_ps(code_load);

        __m512 diff = _mm512_sub_ps(code_float, query_float);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    float l2 = _mm512_reduce_add_ps(sum);
    if (dim > i) {
        l2 += avx2::FP16ComputeL2Sqr(query + i * 2, codes + i * 2, dim - i);
    }
    return l2;
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
    // Initialize the sum to 0
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        __m128i code_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i));
        __m512i codes_512 = _mm512_cvtepu8_epi32(code_values);
        __m512 query_values = _mm512_loadu_ps(query + i);
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lower_bound_values = _mm512_loadu_ps(lower_bound + i);

        __m512 normalized =
            _mm512_mul_ps(_mm512_cvtepi32_ps(codes_512), _mm512_set1_ps(1.0F / 255.0F));
        __m512 adjusted = _mm512_fmadd_ps(normalized, diff_values, lower_bound_values);
        sum = _mm512_fmadd_ps(query_values, adjusted, sum);
    }

    float ip = _mm512_reduce_add_ps(sum);

    if (dim > i) {
        ip += avx2::SQ8ComputeIP(query + i, codes + i, lower_bound + i, diff + i, dim - i);
    }
    return ip;
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
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        __m128i code_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i));
        __m512i codes_512 = _mm512_cvtepu8_epi32(code_values);
        __m512 query_values = _mm512_loadu_ps(query + i);
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lower_bound_values = _mm512_loadu_ps(lower_bound + i);

        __m512 normalized =
            _mm512_mul_ps(_mm512_cvtepi32_ps(codes_512), _mm512_set1_ps(1.0f / 255.0f));
        __m512 adjusted = _mm512_fmadd_ps(normalized, diff_values, lower_bound_values);
        __m512 dist = _mm512_sub_ps(query_values, adjusted);
        sum = _mm512_fmadd_ps(dist, dist, sum);
    }

    float l2 = _mm512_reduce_add_ps(sum);
    if (dim > i) {
        l2 += avx2::SQ8ComputeL2Sqr(query + i, codes + i, lower_bound + i, diff + i, dim - i);
    }
    return l2;
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
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        __m128i code1_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes1 + i));
        __m128i code2_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes2 + i));
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lower_bound_values = _mm512_loadu_ps(lower_bound + i);

        __m512 code1_floats = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(code1_values)),
                                            _mm512_set1_ps(1.0f / 255.0f));
        __m512 code2_floats = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(code2_values)),
                                            _mm512_set1_ps(1.0f / 255.0f));

        __m512 scaled_codes1 = _mm512_fmadd_ps(code1_floats, diff_values, lower_bound_values);
        __m512 scaled_codes2 = _mm512_fmadd_ps(code2_floats, diff_values, lower_bound_values);
        sum = _mm512_fmadd_ps(scaled_codes1, scaled_codes2, sum);
    }
    float result = _mm512_reduce_add_ps(sum);

    if (dim > i) {
        result +=
            avx2::SQ8ComputeCodesIP(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
    }
    return result;
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
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        __m128i code1_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes1 + i));
        __m128i code2_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes2 + i));
        __m512 diff_values = _mm512_loadu_ps(diff + i);

        __m512i codes1_512 = _mm512_cvtepu8_epi32(code1_values);
        __m512i codes2_512 = _mm512_cvtepu8_epi32(code2_values);
        __m512 sub = _mm512_cvtepi32_ps(_mm512_sub_epi32(codes1_512, codes2_512));
        __m512 scaled = _mm512_mul_ps(sub, _mm512_set1_ps(1.0 / 255.0f));
        __m512 val = _mm512_mul_ps(scaled, diff_values);
        sum = _mm512_fmadd_ps(val, val, sum);
    }

    float result = _mm512_reduce_add_ps(sum);

    if (dim > i) {
        result +=
            avx2::SQ8ComputeCodesL2Sqr(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
    }
    return result;
#else
    return avx2::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
    return avx2::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
    return avx2::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
    return avx2::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
    return avx2::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim == 0) {
        return 0;
    }
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
    auto sum1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum));
    auto sum2 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum, 1));
    result += _mm512_reduce_add_epi32(sum1);
    result += _mm512_reduce_add_epi32(sum2);
    if (d < dim) {
        result += avx2::SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);
    }
    return result;
#else
    return avx2::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim == 0) {
        return 0.0f;
    }

    uint64_t d = 0;
    __m512i sum = _mm512_setzero_si512();
    const __m512i mask = _mm512_set1_epi16(0xff);
    for (; d + 63 < dim; d += 64) {
        auto xx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes1 + d));
        auto yy = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes2 + d));

        auto xx1 = _mm512_and_si512(xx, mask);
        auto yy1 = _mm512_and_si512(yy, mask);
        auto xx2 = _mm512_srli_epi16(xx, 8);
        auto yy2 = _mm512_srli_epi16(yy, 8);

        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(xx1, yy1));
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(xx2, yy2));
    }
    int32_t result = _mm512_reduce_add_epi32(sum);
    if (d < dim) {
        result += avx2::SQ8UniformComputeCodesIP(codes1 + d, codes2 + d, dim - d);
    }
    return static_cast<float>(result);
#else
    return avx2::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_AVX512)
    if (dim == 0) {
        return 0.0f;
    }

    if (dim < 16) {
        return avx2::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
    }

    uint64_t d = 0;
    __m512 sum = _mm512_setzero_ps();
    const __m512 inv_sqrt_d_vec = _mm512_set1_ps(inv_sqrt_d);
    const __m512 neg_inv_sqrt_d_vec = _mm512_set1_ps(-inv_sqrt_d);

    for (; d + 16 <= dim; d += 16) {
        __m512 vec = _mm512_loadu_ps(vector + d);

        __mmask16 mask = static_cast<__mmask16>(bits[d / 8 + 1] << 8 | bits[d / 8]);

        __m512 b_vec = _mm512_mask_blend_ps(mask, neg_inv_sqrt_d_vec, inv_sqrt_d_vec);

        sum = _mm512_fmadd_ps(b_vec, vec, sum);
    }

    float result = _mm512_reduce_add_ps(sum);

    if (d < dim) {
        result += avx2::RaBitQFloatBinaryIP(vector + d, bits + (d / 8), dim - d, inv_sqrt_d);
    }

    return result;
#else
    return avx2::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
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
    size_t num_bytes = (dim + 7) / 8;

    const __m512i low_mask = _mm512_set1_epi8(0x0F);

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        size_t i = 0;

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

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_AVX512)
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
    if (dim > i) {
        avx2::DivScalar(from + i, to + i, dim - i, scalar);
    }
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
    for (size_t i = 0; i < 4; i++) {
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
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 64) {
        return avx2::BitAnd(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 63 < num_byte; i += 64) {
        __m512i x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + i));
        __m512i y_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(y + i));
        __m512i z_vec = _mm512_and_si512(x_vec, y_vec);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), z_vec);
    }
    if (i < num_byte) {
        avx2::BitAnd(x + i, y + i, num_byte - i, result + i);
    }
#else
    return avx2::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX512)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 64) {
        return avx2::BitOr(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 63 < num_byte; i += 64) {
        __m512i x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + i));
        __m512i y_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(y + i));
        __m512i z_vec = _mm512_or_si512(x_vec, y_vec);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), z_vec);
    }
    if (i < num_byte) {
        avx2::BitOr(x + i, y + i, num_byte - i, result + i);
    }
#else
    return avx2::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX512)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 64) {
        return avx2::BitXor(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 63 < num_byte; i += 64) {
        __m512i x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + i));
        __m512i y_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(y + i));
        __m512i z_vec = _mm512_xor_si512(x_vec, y_vec);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), z_vec);
    }
    if (i < num_byte) {
        avx2::BitXor(x + i, y + i, num_byte - i, result + i);
    }
#else
    return avx2::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_AVX512)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 64) {
        return avx2::BitNot(x, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 63 < num_byte; i += 64) {
        __m512i x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x + i));
        __m512i z_vec = _mm512_xor_si512(x_vec, _mm512_set1_epi8(0xFF));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), z_vec);
    }
    if (i < num_byte) {
        avx2::BitNot(x + i, num_byte - i, result + i);
    }
#else
    return avx2::BitNot(x, num_byte, result);
#endif
}

void
KacsWalk(float* data, size_t len) {
#if defined(ENABLE_AVX512)
    int base = len % 2;
    int offset = base + (len / 2);
    int i = 0;
    for (; i + 16 < len / 2; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        __m512 y = _mm512_loadu_ps(&data[i + offset]);

        __m512 new_x = _mm512_add_ps(x, y);
        __m512 new_y = _mm512_sub_ps(x, y);

        _mm512_storeu_ps(&data[i], new_x);
        _mm512_storeu_ps(&data[i + offset], new_y);
    }
    for (; i < len / 2; i++) {
        float x = data[i];
        float y = data[i + offset];
        data[i] = x + y;
        data[i + offset] = x - y;
    }
    if (base != 0) {
        data[len / 2] *= std::sqrt(2);
    }
#else
    return avx2::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, size_t dim) {
#if defined(ENABLE_AVX512)
    constexpr size_t kFloatsPerChunk = 64;
    size_t i = 0;
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
VecRescale(float* data, size_t dim, float val) {
#if defined(ENABLE_AVX512)
    __m512 scalar = _mm512_set1_ps(val);
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vec = _mm512_loadu_ps(&data[i]);
        vec = _mm512_mul_ps(vec, scalar);
        _mm512_storeu_ps(&data[i], vec);
    }
    avx2::VecRescale(data + i, dim - i, val);
#else
    return avx2::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_AVX512)
    for (int i = 0; i < dim_; i += step * 2) {
        for (int j = 0; j < step; j += 16) {
            __m512 g1 = _mm512_loadu_ps(&data[i + j]);
            __m512 g2 = _mm512_loadu_ps(&data[i + j + step]);
            _mm512_storeu_ps(&data[i + j], _mm512_add_ps(g1, g2));
            _mm512_storeu_ps(&data[i + j + step], _mm512_sub_ps(g1, g2));
        }
    }
#else
    return avx2::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, size_t dim_) {
#if defined(ENABLE_AVX512)
    size_t n = dim_;
    size_t step = 1;
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
}  // namespace vsag::avx512
