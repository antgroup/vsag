
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
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim) {
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

float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
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
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
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
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
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
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
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
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             uint64_t dim) {
    return avx::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                uint64_t dim) {
    return avx::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  uint64_t dim) {
    return avx::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     uint64_t dim) {
    return avx::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
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
    result += avx::SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);
    return result;
#else
    return avx::SQ4UniformComputeCodesIP(codes1, codes2, dim);
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
    result += static_cast<int32_t>(avx::SQ8UniformComputeCodesIP(codes1 + d, codes2 + d, dim - d));
    return static_cast<float>(result);
#else
    return avx::SQ8UniformComputeCodesIP(codes1, codes2, dim);
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

}  // namespace vsag::avx2