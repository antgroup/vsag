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

#if defined(ENABLE_SVE)
#include <arm_sve.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "simd.h"
constexpr auto
generate_bit_lookup_table() {
    std::array<std::array<uint8_t, 8>, 256> table{};
    for (int byte_value = 0; byte_value < 256; ++byte_value) {
        for (int bit_pos = 0; bit_pos < 8; ++bit_pos) {
            table[byte_value][bit_pos] = ((byte_value >> bit_pos) & 1) ? 1 : 0;
        }
    }
    return table;
}

static constexpr auto g_bit_lookup_table = generate_bit_lookup_table();

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace vsag::sve {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((size_t*)qty_ptr);
    return sve::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((size_t*)qty_ptr);
    return sve::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - sve::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(ENABLE_SVE)
    auto* pVect1 = (const int8_t*)pVect1v;
    auto* pVect2 = (const int8_t*)pVect2v;
    auto qty = *((size_t*)qty_ptr);

    svint32_t sum = svdup_s32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t predicate = svwhilelt_b8(i, qty);
    do {
        svint8_t vec1 = svld1_s8(predicate, pVect1 + i);
        svint8_t vec2 = svld1_s8(predicate, pVect2 + i);
        sum = svdot_s32(sum, vec1, vec2);
        i += step;
        predicate = svwhilelt_b8(i, qty);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_s32(svptrue_b32(), sum));
#else
    return neon::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
#endif
}

float
INT8InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -sve::INT8InnerProduct(pVect1, pVect2, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_SVE)
    const auto* float_centers = (const float*)single_dim_centers;
    auto* float_result = (float*)result;
    uint64_t num_floats_per_vector = svcntw();
    svfloat32_t value = svdup_f32(single_dim_val);
    int i = 0;
    do {
        svbool_t predicate = svwhilelt_b32(i, 256);
        svfloat32_t centers = svld1_f32(predicate, float_centers + i);
        svfloat32_t results = svld1_f32(predicate, float_result + i);
        svfloat32_t diff = svsub_f32_m(predicate, centers, value);
        results = svmad_f32_m(predicate, diff, diff, results);
        svst1_f32(predicate, float_result + i, results);
        i += num_floats_per_vector;
    } while (i < 256);
#else
    neon::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;

    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t codes_vec = svld1_f32(predicate, codes + i);

        sum = svmla_f32_m(predicate, sum, query_vec, codes_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t codes_vec = svld1_f32(predicate, codes + i);

        svfloat32_t diff = svsub_f32_z(predicate, query_vec, codes_vec);

        sum = svmla_f32_m(predicate, sum, diff, diff);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP32ComputeL2Sqr(query, codes, dim);
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
#if defined(ENABLE_SVE)

    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);
    svfloat32_t sum4 = svdup_f32(0.0f);

    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);

        svfloat32_t codes1_vec = svld1_f32(predicate, codes1 + i);
        sum1 = svmla_f32_m(predicate, sum1, query_vec, codes1_vec);

        svfloat32_t codes2_vec = svld1_f32(predicate, codes2 + i);
        sum2 = svmla_f32_m(predicate, sum2, query_vec, codes2_vec);

        svfloat32_t codes3_vec = svld1_f32(predicate, codes3 + i);
        sum3 = svmla_f32_m(predicate, sum3, query_vec, codes3_vec);

        svfloat32_t codes4_vec = svld1_f32(predicate, codes4 + i);
        sum4 = svmla_f32_m(predicate, sum4, query_vec, codes4_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    result1 = svaddv_f32(svptrue_b32(), sum1);
    result2 = svaddv_f32(svptrue_b32(), sum2);
    result3 = svaddv_f32(svptrue_b32(), sum3);
    result4 = svaddv_f32(svptrue_b32(), sum4);
#else
    neon::FP32ComputeIPBatch4(
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
#if defined(ENABLE_SVE)
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);
    svfloat32_t sum4 = svdup_f32(0.0f);

    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);

        svfloat32_t codes1_vec = svld1_f32(predicate, codes1 + i);
        svfloat32_t diff1 = svsub_f32_z(predicate, query_vec, codes1_vec);
        sum1 = svmla_f32_m(predicate, sum1, diff1, diff1);

        svfloat32_t codes2_vec = svld1_f32(predicate, codes2 + i);
        svfloat32_t diff2 = svsub_f32_z(predicate, query_vec, codes2_vec);
        sum2 = svmla_f32_m(predicate, sum2, diff2, diff2);

        svfloat32_t codes3_vec = svld1_f32(predicate, codes3 + i);
        svfloat32_t diff3 = svsub_f32_z(predicate, query_vec, codes3_vec);
        sum3 = svmla_f32_m(predicate, sum3, diff3, diff3);

        svfloat32_t codes4_vec = svld1_f32(predicate, codes4 + i);
        svfloat32_t diff4 = svsub_f32_z(predicate, query_vec, codes4_vec);
        sum4 = svmla_f32_m(predicate, sum4, diff4, diff4);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    result1 = svaddv_f32(svptrue_b32(), sum1);
    result2 = svaddv_f32(svptrue_b32(), sum2);
    result3 = svaddv_f32(svptrue_b32(), sum3);
    result4 = svaddv_f32(svptrue_b32(), sum4);
#else
    neon::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svsub_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svadd_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svmul_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svdiv_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);

        sum = svadd_f32_m(predicate, sum, x_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP32ReduceAdd(x, dim);
#endif
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t query_u32 = svld1uh_u32(predicate, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(predicate, &codes_bf16[i]);

        query_u32 = svlsl_n_u32_x(predicate, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(predicate, codes_u32, 16);

        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);

        sum = svmla_f32_x(predicate, sum, query_f32, codes_f32);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t query_u32 = svld1uh_u32(predicate, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(predicate, &codes_bf16[i]);

        query_u32 = svlsl_n_u32_x(predicate, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(predicate, codes_u32, 16);

        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);

        svfloat32_t diff = svsub_f32_x(predicate, query_f32, codes_f32);
        sum = svmla_f32_x(predicate, sum, diff, diff);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto* query_fp16 = reinterpret_cast<const __fp16*>(query);
    auto* codes_fp16 = reinterpret_cast<const __fp16*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcnth();
    svbool_t predicate = svwhilelt_b16(i, dim);
    do {
        svfloat16_t query_f16 = svld1_f16(predicate, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(predicate, &codes_fp16[i]);

        svbool_t half_predicate = svptrue_pat_b16(SV_POW2);
        svfloat32_t query_f32_low = svcvt_f32_f16_x(half_predicate, query_f16);
        svfloat32_t codes_f32_low = svcvt_f32_f16_x(half_predicate, codes_f16);

        svfloat16_t query_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_high = svcvt_f32_f16_x(half_predicate, query_f16_high);
        svfloat32_t codes_f32_high = svcvt_f32_f16_x(half_predicate, codes_f16_high);

        sum = svmla_f32_x(svptrue_b32(), sum, query_f32_low, codes_f32_low);
        sum = svmla_f32_x(svptrue_b32(), sum, query_f32_high, codes_f32_high);
        i += step;
        predicate = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)

    auto* query_fp16 = reinterpret_cast<const __fp16*>(query);
    auto* codes_fp16 = reinterpret_cast<const __fp16*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcnth();
    svbool_t predicate = svwhilelt_b16(i, dim);
    do {
        svfloat16_t query_f16 = svld1_f16(predicate, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(predicate, &codes_fp16[i]);

        svbool_t half_predicate = svptrue_pat_b16(SV_POW2);
        svfloat32_t query_f32_low = svcvt_f32_f16_x(half_predicate, query_f16);
        svfloat32_t codes_f32_low = svcvt_f32_f16_x(half_predicate, codes_f16);

        svfloat16_t query_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_high = svcvt_f32_f16_x(half_predicate, query_f16_high);
        svfloat32_t codes_f32_high = svcvt_f32_f16_x(half_predicate, codes_f16_high);

        svfloat32_t diff_low = svsub_f32_x(svptrue_b32(), query_f32_low, codes_f32_low);
        svfloat32_t diff_high = svsub_f32_x(svptrue_b32(), query_f32_high, codes_f32_high);

        sum = svmla_f32_x(svptrue_b32(), sum, diff_low, diff_low);
        sum = svmla_f32_x(svptrue_b32(), sum, diff_high, diff_high);
        i += step;
        predicate = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes_u32 = svld1ub_u32(predicate, codes + i);
        svfloat32_t codes_f32 = svcvt_f32_u32_z(predicate, codes_u32);

        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes_f32, scale_factor), diff_vec);

        sum = svmla_f32_m(predicate, sum, query_vec, dequantized);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes_u32 = svld1ub_u32(predicate, codes + i);
        svfloat32_t codes_f32 = svcvt_f32_u32_z(predicate, codes_u32);

        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes_f32, scale_factor), diff_vec);
        svfloat32_t delta = svsub_f32_z(predicate, query_vec, dequantized);
        sum = svmla_f32_m(predicate, sum, delta, delta);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes1_u32 = svld1ub_u32(predicate, codes1 + i);
        svfloat32_t codes1_f32 = svcvt_f32_u32_z(predicate, codes1_u32);
        svuint32_t codes2_u32 = svld1ub_u32(predicate, codes2 + i);
        svfloat32_t codes2_f32 = svcvt_f32_u32_z(predicate, codes2_u32);

        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized1 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes1_f32, scale_factor), diff_vec);
        svfloat32_t dequantized2 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes2_f32, scale_factor), diff_vec);

        sum = svmla_f32_m(predicate, sum, dequantized1, dequantized2);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes1_u32 = svld1ub_u32(predicate, codes1 + i);
        svfloat32_t codes1_f32 = svcvt_f32_u32_z(predicate, codes1_u32);
        svuint32_t codes2_u32 = svld1ub_u32(predicate, codes2 + i);
        svfloat32_t codes2_f32 = svcvt_f32_u32_z(predicate, codes2_u32);

        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized1 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes1_f32, scale_factor), diff_vec);
        svfloat32_t dequantized2 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes2_f32, scale_factor), diff_vec);

        svfloat32_t delta = svsub_f32_z(predicate, dequantized1, dequantized2);
        sum = svmla_f32_m(predicate, sum, delta, delta);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    uint64_t i = 0;
    const svbool_t predicate = svwhilelt_b32(i, dim);

    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t query_pair = svld2_f32(predicate, &query[i]);
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t query_even = svget2_f32(query_pair, 0);
        svfloat32_t query_odd = svget2_f32(query_pair, 1);
        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes = svld1ub_u32(predicate, &codes[i / 2]);
        svuint32_t codes_even_u32 = svand_n_u32_x(predicate, packed_codes, 0x0F);
        svuint32_t codes_odd_u32 = svlsr_n_u32_x(predicate, packed_codes, 4);
        svfloat32_t codes_even_f32 = svcvt_f32_u32_x(predicate, codes_even_u32);
        svfloat32_t codes_odd_f32 = svcvt_f32_u32_x(predicate, codes_odd_u32);

        svfloat32_t dequantized_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes_odd_f32, scale_factor),
                        diff_odd);

        sum = svmla_f32_x(predicate, sum, query_even, dequantized_even);
        sum = svmla_f32_x(predicate, sum, query_odd, dequantized_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeIP(&query[i], &codes[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    const svbool_t predicate = svptrue_b32();

    uint64_t i = 0;
    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t query_pair = svld2_f32(predicate, &query[i]);
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t query_even = svget2_f32(query_pair, 0);
        svfloat32_t query_odd = svget2_f32(query_pair, 1);
        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes = svld1ub_u32(predicate, &codes[i / 2]);
        svuint32_t codes_even_u32 = svand_n_u32_x(predicate, packed_codes, 0x0F);
        svuint32_t codes_odd_u32 = svlsr_n_u32_x(predicate, packed_codes, 4);
        svfloat32_t codes_even_f32 = svcvt_f32_u32_x(predicate, codes_even_u32);
        svfloat32_t codes_odd_f32 = svcvt_f32_u32_x(predicate, codes_odd_u32);

        svfloat32_t dequantized_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes_odd_f32, scale_factor),
                        diff_odd);

        svfloat32_t delta_even = svsub_f32_x(predicate, query_even, dequantized_even);
        svfloat32_t delta_odd = svsub_f32_x(predicate, query_odd, dequantized_odd);

        sum = svmla_f32_x(predicate, sum, delta_even, delta_even);
        sum = svmla_f32_x(predicate, sum, delta_odd, delta_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeL2Sqr(&query[i], &codes[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    const svbool_t predicate = svptrue_b32();

    uint64_t i = 0;
    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes1 = svld1ub_u32(predicate, &codes1[i / 2]);
        svuint32_t packed_codes2 = svld1ub_u32(predicate, &codes2[i / 2]);

        svuint32_t codes1_even_u32 = svand_n_u32_x(predicate, packed_codes1, 0x0F);
        svuint32_t codes1_odd_u32 = svlsr_n_u32_x(predicate, packed_codes1, 4);
        svuint32_t codes2_even_u32 = svand_n_u32_x(predicate, packed_codes2, 0x0F);
        svuint32_t codes2_odd_u32 = svlsr_n_u32_x(predicate, packed_codes2, 4);

        svfloat32_t codes1_even_f32 = svcvt_f32_u32_x(predicate, codes1_even_u32);
        svfloat32_t codes1_odd_f32 = svcvt_f32_u32_x(predicate, codes1_odd_u32);
        svfloat32_t codes2_even_f32 = svcvt_f32_u32_x(predicate, codes2_even_u32);
        svfloat32_t codes2_odd_f32 = svcvt_f32_u32_x(predicate, codes2_odd_u32);

        svfloat32_t dequantized1_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes1_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized1_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes1_odd_f32, scale_factor),
                        diff_odd);
        svfloat32_t dequantized2_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes2_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized2_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes2_odd_f32, scale_factor),
                        diff_odd);

        sum = svmla_f32_x(predicate, sum, dequantized1_even, dequantized2_even);
        sum = svmla_f32_x(predicate, sum, dequantized1_odd, dequantized2_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeCodesIP(
                   &codes1[i / 2], &codes2[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    const svbool_t predicate = svptrue_b32();

    uint64_t i = 0;
    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes1 = svld1ub_u32(predicate, &codes1[i / 2]);
        svuint32_t packed_codes2 = svld1ub_u32(predicate, &codes2[i / 2]);

        svuint32_t codes1_even_u32 = svand_n_u32_x(predicate, packed_codes1, 0x0F);
        svuint32_t codes1_odd_u32 = svlsr_n_u32_x(predicate, packed_codes1, 4);
        svuint32_t codes2_even_u32 = svand_n_u32_x(predicate, packed_codes2, 0x0F);
        svuint32_t codes2_odd_u32 = svlsr_n_u32_x(predicate, packed_codes2, 4);

        svfloat32_t codes1_even_f32 = svcvt_f32_u32_x(predicate, codes1_even_u32);
        svfloat32_t codes1_odd_f32 = svcvt_f32_u32_x(predicate, codes1_odd_u32);
        svfloat32_t codes2_even_f32 = svcvt_f32_u32_x(predicate, codes2_even_u32);
        svfloat32_t codes2_odd_f32 = svcvt_f32_u32_x(predicate, codes2_odd_u32);

        svfloat32_t dequantized1_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes1_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized1_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes1_odd_f32, scale_factor),
                        diff_odd);
        svfloat32_t dequantized2_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes2_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized2_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes2_odd_f32, scale_factor),
                        diff_odd);

        svfloat32_t delta_even = svsub_f32_x(predicate, dequantized1_even, dequantized2_even);
        svfloat32_t delta_odd = svsub_f32_x(predicate, dequantized1_odd, dequantized2_odd);

        sum = svmla_f32_x(predicate, sum, delta_even, delta_even);
        sum = svmla_f32_x(predicate, sum, delta_odd, delta_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeCodesL2Sqr(
                   &codes1[i / 2], &codes2[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}
float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    svuint32_t sum = svdup_u32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb() * 2;
    svbool_t predicate = svwhilelt_b8(i / 2, (dim + 1) / 2);
    do {
        svuint8_t packed_codes1 = svld1_u8(predicate, codes1 + i / 2);
        svuint8_t packed_codes2 = svld1_u8(predicate, codes2 + i / 2);

        svuint8_t codes1_low = svand_u8_z(predicate, packed_codes1, svdup_u8(0x0F));
        svuint8_t codes1_high = svlsr_n_u8_z(predicate, packed_codes1, 4);
        svuint8_t codes2_low = svand_u8_z(predicate, packed_codes2, svdup_u8(0x0F));
        svuint8_t codes2_high = svlsr_n_u8_z(predicate, packed_codes2, 4);

        sum = svdot_u32(sum, codes1_low, codes2_low);
        sum = svdot_u32(sum, codes1_high, codes2_high);

        i += step;
        predicate = svwhilelt_b8(i / 2, (dim + 1) / 2);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_u32(svptrue_b32(), sum));
#else
    return neon::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    svuint32_t sum = svdup_u32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t predicate = svwhilelt_b8(i, dim);
    do {
        svuint8_t codes1_vec = svld1_u8(predicate, codes1 + i);
        svuint8_t codes2_vec = svld1_u8(predicate, codes2 + i);

        sum = svdot_u32(sum, codes1_vec, codes2_vec);

        i += step;
        predicate = svwhilelt_b8(i, dim);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_u32(svptrue_b32(), sum));
#else
    return neon::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 0.0f;
    }

    auto predicate_array = std::make_unique<uint8_t[]>(dim);

    const uint64_t num_bytes = dim / 8;
    for (uint64_t i = 0; i < num_bytes; ++i) {
        memcpy(&predicate_array[i * 8], g_bit_lookup_table[bits[i]].data(), 8);
    }

    if (dim % 8 != 0) {
        const uint64_t remaining_bits = dim % 8;
        memcpy(&predicate_array[num_bytes * 8],
               g_bit_lookup_table[bits[num_bytes]].data(),
               remaining_bits);
    }

    uint64_t i = 0;
    const uint64_t step = svcntw();
    svfloat32_t sum = svdup_f32(0.0f);

    const svfloat32_t positive_val = svdup_f32(inv_sqrt_d);
    const svfloat32_t negative_val = svdup_f32(-inv_sqrt_d);
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t predicate_values = svld1ub_u32(predicate, &predicate_array[i]);

        svbool_t bit_mask = svcmpne_n_u32(predicate, predicate_values, 0);

        svfloat32_t binary_vec = svsel_f32(bit_mask, positive_val, negative_val);
        svfloat32_t vector_values = svld1_f32(predicate, &vector[i]);
        sum = svmla_f32_m(predicate, sum, vector_values, binary_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
#endif
}

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 0;
    }

    uint32_t result = 0;
    size_t num_bytes = (dim + 7) / 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        size_t i = 0;
        uint64_t sum = 0;

        const uint8_t* current_codes = codes + bit_pos * num_bytes;

        svbool_t predicate = svwhilelt_b8(i, num_bytes);
        do {
            svuint8_t codes_vec = svld1_u8(predicate, current_codes + i);
            svuint8_t bits_vec = svld1_u8(predicate, bits + i);

            svuint8_t and_result = svand_u8_x(predicate, codes_vec, bits_vec);

            svuint8_t popcount = svcnt_u8_x(predicate, and_result);

            sum += svaddv_u8(predicate, popcount);

            i += svcntb();
            predicate = svwhilelt_b8(i, num_bytes);
        } while (svptest_first(svptrue_b8(), predicate));

        result += sum << bit_pos;
    }

    return result;
#else
    return neon::RaBitQSQ4UBinaryIP(codes, bits, dim);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t y_vec = svld1_u8(predicate, y + i);
        svuint8_t result_vec = svand_u8_z(predicate, x_vec, y_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t y_vec = svld1_u8(predicate, y + i);
        svuint8_t result_vec = svorr_u8_z(predicate, x_vec, y_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t y_vec = svld1_u8(predicate, y + i);
        svuint8_t result_vec = sveor_u8_z(predicate, x_vec, y_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t result_vec = svnot_u8_z(predicate, x_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitNot(x, num_byte, result);
#endif
}

void
Prefetch(const void* data) {
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return;
    }
    if (scalar == 0) {
        scalar = 1.0f;
    }
    svfloat32_t divisor = svdup_f32(scalar);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t values = svld1_f32(predicate, from + i);
        svfloat32_t result = svdiv_f32_z(predicate, values, divisor);
        svst1_f32(predicate, to + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
#if defined(ENABLE_SVE)
    float norm = std::sqrt(sve::FP32ComputeIP(from, from, dim));
    if (norm == 0) {
        norm = 1.0f;
    }
    sve::DivScalar(from, to, dim, norm);
    return norm;
#else
    return neon::Normalize(from, to, dim);
#endif
}

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t total_bytes = pq_dim * 16;
    auto step = svcntb();

    const svuint8_t mask_low = svdup_u8(0x0F);
    const svuint16_t mask_low16 = svdup_u16(0x00FF);

    svuint16_t accum0 = svdup_u16(0);
    svuint16_t accum1 = svdup_u16(0);
    svuint16_t accum2 = svdup_u16(0);
    svuint16_t accum3 = svdup_u16(0);

    uint8_t offsets_data[svcntb()];
    for (uint64_t c = 0; c < svcntb() / 16; ++c) std::memset(offsets_data + c * 16, c * 16, 16);

    const svuint8_t index_offsets = svld1_u8(svptrue_b8(), offsets_data);

    svbool_t predicate = svwhilelt_b8(i, total_bytes);
    do {
        svuint8_t table_data = svld1_u8(predicate, lookup_table + i);
        svuint8_t code_data = svld1_u8(predicate, codes + i);

        svuint8_t low_nibbles = svand_u8_z(predicate, code_data, mask_low);
        svuint8_t high_nibbles = svlsr_n_u8_z(predicate, code_data, 4);

        svuint8_t adjusted_low_indices = svadd_u8_z(predicate, low_nibbles, index_offsets);
        svuint8_t adjusted_high_indices = svadd_u8_z(predicate, high_nibbles, index_offsets);

        svuint8_t low_values = svtbl_u8(table_data, adjusted_low_indices);
        svuint8_t high_values = svtbl_u8(table_data, adjusted_high_indices);

        svbool_t predicate_u16 = svwhilelt_b16(i / 2, total_bytes / 2);

        accum0 =
            svadd_u16_m(predicate_u16,
                        accum0,
                        svand_u16_z(predicate_u16, svreinterpret_u16_u8(low_values), mask_low16));
        accum1 = svadd_u16_m(predicate_u16,
                             accum1,
                             svlsr_n_u16_z(predicate_u16, svreinterpret_u16_u8(low_values), 8));
        accum2 =
            svadd_u16_m(predicate_u16,
                        accum2,
                        svand_u16_z(predicate_u16, svreinterpret_u16_u8(high_values), mask_low16));
        accum3 = svadd_u16_m(predicate_u16,
                             accum3,
                             svlsr_n_u16_z(predicate_u16, svreinterpret_u16_u8(high_values), 8));

        i += step;
        predicate = svwhilelt_b8(i, total_bytes);
    } while (svptest_first(svptrue_b8(), predicate));

    uint16_t temp[svcntb() / 2];

    // Segment 0
    svst1_u16(svptrue_b16(), temp, accum0);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[0 * 8 + j] += temp[j + 8 * (k)];

    // Segment 1
    svst1_u16(svptrue_b16(), temp, accum1);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[1 * 8 + j] += temp[j + 8 * k];

    // Segment 2
    svst1_u16(svptrue_b16(), temp, accum2);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[2 * 8 + j] += temp[j + 8 * k];

    // Segment 3
    svst1_u16(svptrue_b16(), temp, accum3);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[3 * 8 + j] += temp[j + 8 * k];

#else
    neon::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
KacsWalk(float* data, size_t len) {
#if defined(ENABLE_SVE)
    size_t n = len / 2;
    size_t offset = (len % 2) + n;
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, n);
    do {
        svfloat32_t vec1 = svld1_f32(predicate, data + i);
        svfloat32_t vec2 = svld1_f32(predicate, data + i + offset);
        svfloat32_t sum_vec = svadd_f32_z(predicate, vec1, vec2);
        svfloat32_t diff_vec = svsub_f32_z(predicate, vec1, vec2);
        svst1_f32(predicate, data + i, sum_vec);
        svst1_f32(predicate, data + i + offset, diff_vec);
        i += step;
        predicate = svwhilelt_b32(i, n);
    } while (svptest_first(svptrue_b32(), predicate));

    if (len % 2 != 0) {
        data[n] *= std::sqrt(2.0F);
    }
#else
    neon::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, size_t dim) {
#if defined(ENABLE_SVE)
    auto predicate_array = std::make_unique<uint8_t[]>(dim);
    const uint64_t num_bytes = dim / 8;
    for (uint64_t j = 0; j < num_bytes; ++j) {
        memcpy(&predicate_array[j * 8], g_bit_lookup_table[flip[j]].data(), 8);
    }
    if (dim % 8 != 0) {
        const uint64_t remaining_bits = dim % 8;
        memcpy(&predicate_array[num_bytes * 8],
               g_bit_lookup_table[flip[num_bytes]].data(),
               remaining_bits);
    }

    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t predicate_values = svld1ub_u32(predicate, &predicate_array[i]);
        svbool_t bit_mask = svcmpne_n_u32(predicate, predicate_values, 0);

        svfloat32_t data_vec = svld1_f32(predicate, data + i);
        svfloat32_t result_vec = svneg_f32_m(data_vec, bit_mask, data_vec);
        svst1_f32(predicate, data + i, result_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FlipSign(flip, data, dim);
#endif
}

void
VecRescale(float* data, size_t dim, float val) {
#if defined(ENABLE_SVE)
    svfloat32_t scale = svdup_f32(val);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t data_vec = svld1_f32(predicate, data + i);
        svfloat32_t result = svmul_f32_z(predicate, data_vec, scale);
        svst1_f32(predicate, data + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_SVE)
    for (int i = idx; i < dim_; i += 2 * step) {
        uint64_t j = 0;
        const uint64_t sve_step = svcntw();
        svbool_t predicate = svwhilelt_b32(j, (uint64_t)step);
        do {
            svfloat32_t x = svld1_f32(predicate, data + i + j);
            svfloat32_t y = svld1_f32(predicate, data + i + j + step);
            svst1_f32(predicate, data + i + j, svadd_f32_z(predicate, x, y));
            svst1_f32(predicate, data + i + j + step, svsub_f32_z(predicate, x, y));
            j += sve_step;
            predicate = svwhilelt_b32(j, (uint64_t)step);
        } while (svptest_first(svptrue_b32(), predicate));
    }
#else
    neon::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, size_t dim_) {
#if defined(ENABLE_SVE)
    size_t n = dim_;
    size_t step = 1;
    while (step < n) {
        sve::RotateOp(data, 0, dim_, step);
        step *= 2;
    }
#else
    neon::FHTRotate(data, dim_);
#endif
}

}  // namespace vsag::sve