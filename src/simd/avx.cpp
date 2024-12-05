
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
#include <cstdint>

#include "fp32_simd.h"
#include "normalize.h"
#include "sq4_simd.h"
#include "sq4_uniform_simd.h"
#include "sq8_simd.h"
#include "sq8_uniform_simd.h"

namespace vsag {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

float
L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
           TmpRes[7];
}

float
InnerProductSIMD4ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float* pEnd1 = pVect1 + 16 * qty16;
    const float* pEnd2 = pVect1 + 4 * qty4;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    __m128 v1, v2;
    __m128 sum_prod =
        _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

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
InnerProductSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;

    const float* pEnd1 = pVect1 + 16 * qty16;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7];

    return sum;
}

void
PQDistanceAVXFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
    const float* float_centers = (const float*)single_dim_centers;
    float* float_result = (float*)result;
    for (size_t idx = 0; idx < 256; idx += 8) {
        __m256 v_centers_dim = _mm256_loadu_ps(float_centers + idx);
        __m256 v_query_vec = _mm256_set1_ps(single_dim_val);
        __m256 v_diff = _mm256_sub_ps(v_centers_dim, v_query_vec);
        __m256 v_diff_sq = _mm256_mul_ps(v_diff, v_diff);
        __m256 v_chunk_dists = _mm256_loadu_ps(&float_result[idx]);
        v_chunk_dists = _mm256_add_ps(v_chunk_dists, v_diff_sq);
        _mm256_storeu_ps(&float_result[idx], v_chunk_dists);
    }
}

namespace avx2 {

#if defined(ENABLE_AVX2)

__inline __m128i __attribute__((__always_inline__)) load_8_char(const uint8_t* data) {
    return _mm_set_epi8(0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        data[7],
                        data[6],
                        data[5],
                        data[4],
                        data[3],
                        data[2],
                        data[1],
                        data[0]);
}

#endif

float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    const int n = dim / 8;
    if (n == 0) {
        return sse::FP32ComputeIP(query, codes, dim);
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
    ip += sse::FP32ComputeIP(query + n * 8, codes + n * 8, dim - n * 8);
    return ip;
#else
    return vsag::generic::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_AVX2)
    const int n = dim / 8;
    if (n == 0) {
        return sse::FP32ComputeL2Sqr(query, codes, dim);
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
    l2 += sse::FP32ComputeL2Sqr(query + n * 8, codes + n * 8, dim - n * 8);
    return l2;
#else
    return vsag::generic::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim) {
#if defined(ENABLE_AVX2)
    __m256 sum = _mm256_setzero_ps();
    uint64_t i = 0;

    for (; i + 7 < dim; i += 8) {
        __m128i code_values = load_8_char(codes + i);
        __m256 code_floats = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(code_values));
        __m256 query_values = _mm256_loadu_ps(query + i);
        __m256 diff_values = _mm256_loadu_ps(diff + i);
        __m256 lowerBound_values = _mm256_loadu_ps(lowerBound + i);

        __m256 scaled_codes =
            _mm256_mul_ps(_mm256_div_ps(code_floats, _mm256_set1_ps(255.0f)), diff_values);
        __m256 adjusted_codes = _mm256_add_ps(scaled_codes, lowerBound_values);
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
    finalResult += sse::SQ8ComputeIP(query + i, codes + i, lowerBound + i, diff + i, dim - i);
    return finalResult;
#else
    return generic::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lowerBound,
                const float* diff,
                uint64_t dim) {
#if defined(ENABLE_AVX2)
    __m256 sum = _mm256_setzero_ps();
    uint64_t i = 0;

    for (; i + 7 < dim; i += 8) {
        // Load data into registers
        __m256i code_values = _mm256_cvtepu8_epi32(load_8_char(codes + i));
        __m256 code_floats = _mm256_div_ps(_mm256_cvtepi32_ps(code_values), _mm256_set1_ps(255.0f));
        __m256 diff_values = _mm256_loadu_ps(diff + i);
        __m256 lowerBound_values = _mm256_loadu_ps(lowerBound + i);
        __m256 query_values = _mm256_loadu_ps(query + i);

        // Perform calculations
        __m256 scaled_codes = _mm256_mul_ps(code_floats, diff_values);
        scaled_codes = _mm256_add_ps(scaled_codes, lowerBound_values);
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
    result += sse::SQ8ComputeL2Sqr(query + i, codes + i, lowerBound + i, diff + i, dim - i);
    return result;
#else
    return vsag::generic::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);  // TODO
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
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
        __m256 lowerBound_values = _mm256_loadu_ps(lowerBound + i);
        // Perform calculations
        __m256 scaled_codes1 = _mm256_fmadd_ps(code1_floats, diff_values, lowerBound_values);
        __m256 scaled_codes2 = _mm256_fmadd_ps(code2_floats, diff_values, lowerBound_values);
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

    result += sse::SQ8ComputeCodesIP(codes1 + i, codes2 + i, lowerBound + i, diff + i, dim - i);
    return result;
#else
    return generic::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lowerBound,
                     const float* diff,
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
        __m256 lowerBound_values = _mm256_loadu_ps(lowerBound + i);
        // Perform calculations
        __m256 scaled_codes1 = _mm256_fmadd_ps(codes1_floats, diff_values, lowerBound_values);
        __m256 scaled_codes2 = _mm256_fmadd_ps(codes2_floats, diff_values, lowerBound_values);
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

    result += sse::SQ8ComputeCodesL2Sqr(codes1 + i, codes2 + i, lowerBound + i, diff + i, dim - i);
    return result;
#else
    return generic::SQ8ComputeCodesL2Sqr(codes1, codes2, lowerBound, diff, dim);
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
    result += sse::SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);
    return result;
#else
    return sse::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim) {
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
    result += static_cast<int32_t>(sse::SQ8UniformComputeCodesIP(codes1 + d, codes2 + d, dim - d));
    return static_cast<float>(result);
#else
    return sse::SQ8UniformComputeCodesIP(codes1, codes2, dim);
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
    sse::DivScalar(from + i, to + i, dim - i, scalar);
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

}  // namespace avx2
}  // namespace vsag
