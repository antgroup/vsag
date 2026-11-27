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
#if defined(ENABLE_SVE)
#include <arm_sve.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "kernels/local_sq_compute.h"
#include "kernels/rabitq_pack.h"
#include "kernels/rabitq_packed_supplement.h"
#include "simd.h"
#include "sq4_local_simd.h"
#include "sq8_local_simd.h"
#include "utils/float_utils.h"
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

namespace vsag::sve {
#if defined(ENABLE_SVE)
#if defined(ENABLE_NEON)
namespace sq4_tail = neon;
#else
namespace sq4_tail = generic;
#endif
#endif

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}
float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::INT8ComputeL2Sqr(pVect1, pVect2, qty);
}
float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - sve::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::INT8ComputeIP(pVect1, pVect2, qty);
}

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svint64_t sum = svdup_s64(0);
    uint64_t i = 0;
    const uint64_t step = svcnth();

    svbool_t predicate = svwhilelt_b16(i, dim);
    do {
        svint16_t vec_query = svld1sb_s16(predicate, query + i);
        svint16_t vec_codes = svld1sb_s16(predicate, codes + i);

        svint16_t diff = svsub_s16_x(predicate, vec_query, vec_codes);

        sum = svdot_s64(sum, diff, diff);

        i += step;
        predicate = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b16(), predicate));

    return static_cast<float>(svaddv_s64(svptrue_b64(), sum));
#else
    return neon::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svint32_t sum = svdup_s32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t predicate = svwhilelt_b8(i, dim);
    do {
        svint8_t vec1 = svld1_s8(predicate, query + i);
        svint8_t vec2 = svld1_s8(predicate, codes + i);
        sum = svdot_s32(sum, vec1, vec2);
        i += step;
        predicate = svwhilelt_b8(i, dim);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_s32(svptrue_b32(), sum));
#else
    return neon::INT8ComputeIP(query, codes, dim);
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
FP32SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const float* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    // SVE permits up to 2048 bits, i.e. at most 64 float lanes.
    float deltas[64];
    const auto width = static_cast<uint32_t>(svcntw());
    uint32_t i = 0;
    while (i < num) {
        const uint32_t remaining = num - i;
        const uint32_t count = remaining < width ? remaining : width;
        const svbool_t active = svwhilelt_b32(uint32_t{0}, count);
        const svfloat32_t values = svld1_f32(active, vals + i);
        svst1_f32(active, deltas, svmul_n_f32_x(active, values, query_val));
        // Scatter in posting order, including repeated IDs.
        for (uint32_t lane = 0; lane < count; ++lane) {
            dists[ids[i + lane]] += deltas[lane];
        }
        i += count;
    }
#else
    neon::FP32SparseAccumulate(dists, ids, vals, query_val, num);
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

    result1 += svaddv_f32(svptrue_b32(), sum1);
    result2 += svaddv_f32(svptrue_b32(), sum2);
    result3 += svaddv_f32(svptrue_b32(), sum3);
    result4 += svaddv_f32(svptrue_b32(), sum4);
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

    result1 += svaddv_f32(svptrue_b32(), sum1);
    result2 += svaddv_f32(svptrue_b32(), sum2);
    result3 += svaddv_f32(svptrue_b32(), sum3);
    result4 += svaddv_f32(svptrue_b32(), sum4);
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

        // Conversion is predicated on the widened FP32 lanes. The half loads
        // already zero inactive lanes; use all lanes even at non-power-of-two VL.
        const svbool_t convert_all = svptrue_b32();
        svfloat32_t query_f32_low = svcvt_f32_f16_x(convert_all, query_f16);
        svfloat32_t codes_f32_low = svcvt_f32_f16_x(convert_all, codes_f16);

        svfloat16_t query_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_high = svcvt_f32_f16_x(convert_all, query_f16_high);
        svfloat32_t codes_f32_high = svcvt_f32_f16_x(convert_all, codes_f16_high);

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

        // Conversion is predicated on the widened FP32 lanes. The half loads
        // already zero inactive lanes; use all lanes even at non-power-of-two VL.
        const svbool_t convert_all = svptrue_b32();
        svfloat32_t query_f32_low = svcvt_f32_f16_x(convert_all, query_f16);
        svfloat32_t codes_f32_low = svcvt_f32_f16_x(convert_all, codes_f16);

        svfloat16_t query_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_high = svcvt_f32_f16_x(convert_all, query_f16_high);
        svfloat32_t codes_f32_high = svcvt_f32_f16_x(convert_all, codes_f16_high);

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

void
FP16SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const uint16_t* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    // SVE permits up to 2048 bits, i.e. at most 64 float lanes.
    float deltas[64];
    const auto width = static_cast<uint32_t>(svcntw());
    uint32_t i = 0;
    while (i < num) {
        const uint32_t remaining = num - i;
        const uint32_t count = remaining < width ? remaining : width;
        const svbool_t active = svwhilelt_b32(uint32_t{0}, count);
        const svuint32_t half = svld1uh_u32(active, vals + i);
        // Match the sparse codec's historical bit expansion.
        const svuint32_t sign = svlsl_n_u32_x(active, svand_n_u32_x(active, half, 0x8000), 16);
        const svuint32_t magnitude = svadd_n_u32_x(
            active, svlsl_n_u32_x(active, svand_n_u32_x(active, half, 0x7fff), 13), 112U << 23);
        const svfloat32_t values = svreinterpret_f32_u32(svorr_u32_x(active, sign, magnitude));
        svst1_f32(active, deltas, svmul_n_f32_x(active, values, query_val));
        // Scatter in posting order, including repeated IDs.
        for (uint32_t lane = 0; lane < count; ++lane) {
            dists[ids[i + lane]] += deltas[lane];
        }
        i += count;
    }
#else
    neon::FP16SparseAccumulate(dists, ids, vals, query_val, num);
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

#if defined(ENABLE_SVE)
namespace {
template <bool KIsL2>
svfloat32_t
SQ8AccumulateBatch4(svbool_t active,
                    const uint8_t* code,
                    svfloat32_t query,
                    svfloat32_t lower_bound,
                    svfloat32_t diff,
                    svfloat32_t inv255,
                    svfloat32_t sum) {
    const auto values = svcvt_f32_u32_z(active, svld1ub_u32(active, code));
    const auto adjusted =
        svmla_f32_m(active, lower_bound, svmul_f32_m(active, values, inv255), diff);
    if constexpr (KIsL2) {
        const auto delta = svsub_f32_z(active, query, adjusted);
        return svmla_f32_m(active, sum, delta, delta);
    } else {
        return svmla_f32_m(active, sum, query, adjusted);
    }
}

template <bool KIsL2>
void
SQ8ComputeBatch4(const float* query,
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
    auto sum1 = svdup_f32(0.0F);
    auto sum2 = svdup_f32(0.0F);
    auto sum3 = svdup_f32(0.0F);
    auto sum4 = svdup_f32(0.0F);
    const auto inv255 = svdup_f32(1.0F / 255.0F);
    const auto width = svcntw();
    for (uint64_t i = 0; i < dim; i += width) {
        const auto active = svwhilelt_b32(i, dim);
        const auto q = svld1_f32(active, query + i);
        const auto lb = svld1_f32(active, lower_bound + i);
        const auto scale = svld1_f32(active, diff + i);
        sum1 = SQ8AccumulateBatch4<KIsL2>(active, codes1 + i, q, lb, scale, inv255, sum1);
        sum2 = SQ8AccumulateBatch4<KIsL2>(active, codes2 + i, q, lb, scale, inv255, sum2);
        sum3 = SQ8AccumulateBatch4<KIsL2>(active, codes3 + i, q, lb, scale, inv255, sum3);
        sum4 = SQ8AccumulateBatch4<KIsL2>(active, codes4 + i, q, lb, scale, inv255, sum4);
    }
    const auto all = svptrue_b32();
    dist1 = svaddv_f32(all, sum1);
    dist2 = svaddv_f32(all, sum2);
    dist3 = svaddv_f32(all, sum3);
    dist4 = svaddv_f32(all, sum4);
}
}  // namespace
#endif

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
#if defined(ENABLE_SVE)
    SQ8ComputeBatch4<true>(
        query, codes1, codes2, codes3, codes4, lower_bound, diff, dim, dist1, dist2, dist3, dist4);
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
#if defined(ENABLE_SVE)
    SQ8ComputeBatch4<false>(
        query, codes1, codes2, codes3, codes4, lower_bound, diff, dim, dist1, dist2, dist3, dist4);
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

void
SQ8SparseAccumulate(float* RESTRICT dists,
                    const uint16_t* RESTRICT ids,
                    const uint8_t* RESTRICT vals,
                    float query_val,
                    uint32_t num) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    // SVE permits up to 2048 bits, i.e. at most 64 float lanes.
    float deltas[64];
    const auto width = static_cast<uint32_t>(svcntw());
    uint32_t i = 0;
    while (i < num) {
        const uint32_t remaining = num - i;
        const uint32_t count = remaining < width ? remaining : width;
        const svbool_t active = svwhilelt_b32(uint32_t{0}, count);
        const svfloat32_t values = svcvt_f32_u32_z(active, svld1ub_u32(active, vals + i));
        svst1_f32(active, deltas, svmul_n_f32_x(active, values, query_val));
        // Scatter in posting order, including repeated IDs.
        for (uint32_t lane = 0; lane < count; ++lane) {
            dists[ids[i + lane]] += deltas[lane];
        }
        i += count;
    }
#else
    neon::SQ8SparseAccumulate(dists, ids, vals, query_val, num);
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
               sq4_tail::SQ4ComputeIP(&query[i], &codes[i / 2], &lower_bound[i], &diff[i], dim - i);
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
               sq4_tail::SQ4ComputeL2Sqr(
                   &query[i], &codes[i / 2], &lower_bound[i], &diff[i], dim - i);
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
               sq4_tail::SQ4ComputeCodesIP(
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
               sq4_tail::SQ4ComputeCodesL2Sqr(
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

void
SQ8UniformComputeCodesIPBatch(const uint8_t* RESTRICT query,
                              const uint8_t* RESTRICT codes,
                              uint64_t dim,
                              uint64_t n_codes,
                              uint64_t code_stride,
                              float* RESTRICT out) {
#if defined(ENABLE_SVE)
    const uint64_t width = svcntb();
    uint64_t row = 0;
    for (; n_codes - row >= 4; row += 4) {
        const auto* c1 = codes + row * code_stride;
        const auto* c2 = c1 + code_stride;
        const auto* c3 = c2 + code_stride;
        const auto* c4 = c3 + code_stride;
        auto a1 = svdup_u32(0);
        auto a2 = svdup_u32(0);
        auto a3 = svdup_u32(0);
        auto a4 = svdup_u32(0);
        for (uint64_t d = 0; d < dim; d += width) {
            const auto active = svwhilelt_b8(d, dim);
            const auto q = svld1_u8(active, query + d);
            a1 = svdot_u32(a1, q, svld1_u8(active, c1 + d));
            a2 = svdot_u32(a2, q, svld1_u8(active, c2 + d));
            a3 = svdot_u32(a3, q, svld1_u8(active, c3 + d));
            a4 = svdot_u32(a4, q, svld1_u8(active, c4 + d));
        }
        const auto all = svptrue_b32();
        out[row] = static_cast<float>(svaddv_u32(all, a1));
        out[row + 1] = static_cast<float>(svaddv_u32(all, a2));
        out[row + 2] = static_cast<float>(svaddv_u32(all, a3));
        out[row + 3] = static_cast<float>(svaddv_u32(all, a4));
    }
    for (; row < n_codes; ++row) {
        out[row] = sve::SQ8UniformComputeCodesIP(query, codes + row * code_stride, dim);
    }
#else
    neon::SQ8UniformComputeCodesIPBatch(query, codes, dim, n_codes, code_stride, out);
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

    // Match generic and Batch4: keep the legacy double-precision threshold.
    const bool signed_bits = inv_sqrt_d > 1e-3;
    const svfloat32_t positive_val = svdup_f32(signed_bits ? inv_sqrt_d : 1.0F);
    const svfloat32_t negative_val = svdup_f32(signed_bits ? -inv_sqrt_d : 0.0F);
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

#if defined(ENABLE_SVE)
static svbool_t
RaBitQBinaryMask(svbool_t predicate,
                 const uint8_t* bits,
                 svuint32_t byte_offsets,
                 svuint32_t bit_shifts) {
    const svuint32_t bytes = svld1ub_gather_u32offset_u32(predicate, bits, byte_offsets);
    const svuint32_t shifted = svlsr_u32_x(predicate, bytes, bit_shifts);
    return svcmpne_n_u32(predicate, svand_n_u32_x(predicate, shifted, 1U), 0U);
}
#endif

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results) {
#if defined(ENABLE_SVE)
    svfloat32_t sum1 = svdup_f32(0.0F);
    svfloat32_t sum2 = svdup_f32(0.0F);
    svfloat32_t sum3 = svdup_f32(0.0F);
    svfloat32_t sum4 = svdup_f32(0.0F);
    const svfloat32_t pos = svdup_f32(inv_sqrt_d > 1e-3 ? inv_sqrt_d : 1.0F);
    const svfloat32_t neg = svdup_f32(inv_sqrt_d > 1e-3 ? -inv_sqrt_d : 0.0F);
    const svuint32_t lanes = svindex_u32(0, 1);
    const uint64_t step = svcntw();

    for (uint64_t d = 0; d < dim; d += step) {
        const svbool_t predicate = svwhilelt_b32(d, dim);
        const svfloat32_t query = svld1_f32(predicate, vector + d);
        // Offsets are relative to the current code byte, including mid-byte starts.
        const svuint32_t bit_indices =
            svadd_n_u32_x(predicate, lanes, static_cast<uint32_t>(d & 7U));
        const svuint32_t byte_offsets = svlsr_n_u32_x(predicate, bit_indices, 3);
        const svuint32_t bit_shifts = svand_n_u32_x(predicate, bit_indices, 7U);
        const uint64_t byte_id = d >> 3;

        const svfloat32_t binary1 = svsel_f32(
            RaBitQBinaryMask(predicate, bits1 + byte_id, byte_offsets, bit_shifts), pos, neg);
        const svfloat32_t binary2 = svsel_f32(
            RaBitQBinaryMask(predicate, bits2 + byte_id, byte_offsets, bit_shifts), pos, neg);
        const svfloat32_t binary3 = svsel_f32(
            RaBitQBinaryMask(predicate, bits3 + byte_id, byte_offsets, bit_shifts), pos, neg);
        const svfloat32_t binary4 = svsel_f32(
            RaBitQBinaryMask(predicate, bits4 + byte_id, byte_offsets, bit_shifts), pos, neg);
        sum1 = svmla_f32_m(predicate, sum1, query, binary1);
        sum2 = svmla_f32_m(predicate, sum2, query, binary2);
        sum3 = svmla_f32_m(predicate, sum3, query, binary3);
        sum4 = svmla_f32_m(predicate, sum4, query, binary4);
    }

    results[0] = svaddv_f32(svptrue_b32(), sum1);
    results[1] = svaddv_f32(svptrue_b32(), sum2);
    results[2] = svaddv_f32(svptrue_b32(), sum3);
    results[3] = svaddv_f32(svptrue_b32(), sum4);
#else
    generic::RaBitQFloatBinaryIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
#endif
}

#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
namespace {
template <uint32_t Bits, uint32_t Count, bool Centered>
void
RaBitQBitPlanesIP(const float* vector,
                  const uint8_t* const (&codes)[Count],
                  uint64_t dim,
                  uint32_t reorder_bits,
                  float* results) {
    const auto plane_bytes = (dim + 7) / 8;
    const auto step = svcntw();
    const auto lanes = svindex_u32(0, 1);
    // SVE float-lane counts are multiples of four, not necessarily eight.
    const auto all = svptrue_b32();
    const auto bit_index = svand_n_u32_x(all, lanes, 7);
    auto s0 = svdup_f32(0.0F);
    auto s1 = svdup_f32(0.0F);
    auto s2 = svdup_f32(0.0F);
    auto s3 = svdup_f32(0.0F);
    for (uint64_t i = 0; i < dim; i += step) {
        const auto pg = svwhilelt_b32(i, dim);
        const auto shifted = svadd_n_u32_x(pg, bit_index, static_cast<uint32_t>(i & 7));
        const auto offsets = svlsr_n_u32_x(pg, svadd_n_u32_x(pg, lanes, i & 7), 3);
        const auto shift = svand_n_u32_x(pg, shifted, 7);
        const auto q = svld1_f32(pg, vector + i);
        const auto accumulate = [&](const uint8_t* bits, svfloat32_t sum) {
            auto weight = svdup_f32(Centered ? -0.5F * ((1U << Bits) - 1U) : 0.0F);
            for (uint32_t plane = 0; plane < Bits; ++plane) {
                const auto raw = svld1ub_gather_u32offset_u32(
                    pg, bits + plane * plane_bytes + (i >> 3), offsets);
                const auto bit = svand_n_u32_x(pg, svlsr_u32_x(pg, raw, shift), 1);
                const auto present = svcmpne_n_u32(pg, bit, 0);
                const auto value = static_cast<float>(1U << (reorder_bits + Bits - plane - 1));
                weight = svadd_n_f32_m(present, weight, value);
            }
            return svmla_f32_m(pg, sum, q, weight);
        };
        s0 = accumulate(codes[0], s0);
        if constexpr (Count == 4) {
            s1 = accumulate(codes[1], s1);
            s2 = accumulate(codes[2], s2);
            s3 = accumulate(codes[3], s3);
        }
    }
    results[0] = svaddv_f32(all, s0);
    if constexpr (Count == 4) {
        results[1] = svaddv_f32(all, s1);
        results[2] = svaddv_f32(all, s2);
        results[3] = svaddv_f32(all, s3);
    }
}
float
RaBitQPlanarIP(const float* vector,
               const uint8_t* sign,
               const uint8_t* supplement,
               uint64_t dim,
               uint32_t bits) {
    const auto plane_bytes = (dim + 7) / 8;
    const auto step = svcntw();
    const auto lanes = svindex_u32(0, 1);
    auto sum = svdup_f32(0.0F);
    for (uint64_t i = 0; i < dim; i += step) {
        const auto pg = svwhilelt_b32(i, dim);
        const auto shifted = svadd_n_u32_x(pg, lanes, i & 7);
        const auto offsets = svlsr_n_u32_x(pg, shifted, 3);
        const auto shift = svand_n_u32_x(pg, shifted, 7);
        auto code = svdup_u32(0);
        const auto accumulate = [&](const uint8_t* plane, uint32_t bit) {
            const auto raw = svld1ub_gather_u32offset_u32(pg, plane + (i >> 3), offsets);
            const auto lane = svand_n_u32_x(pg, svlsr_u32_x(pg, raw, shift), 1);
            code = svorr_u32_x(pg, code, svlsl_n_u32_x(pg, lane, bit));
        };
        if (sign != nullptr) {
            accumulate(sign, bits);
        }
        for (uint32_t bit = 0; bit < bits; ++bit) {
            accumulate(supplement + bit * plane_bytes, bit);
        }
        sum = svmla_f32_m(pg, sum, svld1_f32(pg, vector + i), svcvt_f32_u32_x(pg, code));
    }
    return svaddv_f32(svptrue_b32(), sum);
}
}  // namespace
#endif

float
RaBitQFloatTwoBitCenteredIP(const float* vector, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const uint8_t* codes[] = {bits};
    float results[1];
    RaBitQBitPlanesIP<2, 1, true>(vector, codes, dim, 0, results);
    return results[0];
#else
    return neon::RaBitQFloatTwoBitCenteredIP(vector, bits, dim);
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
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};

    RaBitQBitPlanesIP<2, 4, true>(vector, codes, dim, 0, results);

#else
    neon::RaBitQFloatTwoBitCenteredIPBatch4(vector, bits1, bits2, bits3, bits4, dim, results);
#endif
}

float
RaBitQFloatThreeBitCenteredIP(const float* vector, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const uint8_t* codes[] = {bits};
    float results[1];
    RaBitQBitPlanesIP<3, 1, true>(vector, codes, dim, 0, results);
    return results[0];
#else
    return neon::RaBitQFloatThreeBitCenteredIP(vector, bits, dim);
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
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};

    RaBitQBitPlanesIP<3, 4, true>(vector, codes, dim, 0, results);

#else
    neon::RaBitQFloatThreeBitCenteredIPBatch4(vector, bits1, bits2, bits3, bits4, dim, results);
#endif
}

float
RaBitQFloatFourBitCenteredIP(const float* vector, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const uint8_t* codes[] = {bits};
    float results[1];
    RaBitQBitPlanesIP<4, 1, true>(vector, codes, dim, 0, results);
    return results[0];
#else
    return neon::RaBitQFloatFourBitCenteredIP(vector, bits, dim);
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
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};

    RaBitQBitPlanesIP<4, 4, true>(vector, codes, dim, 0, results);

#else
    neon::RaBitQFloatFourBitCenteredIPBatch4(vector, bits1, bits2, bits3, bits4, dim, results);
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
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};
    RaBitQBitPlanesIP<3, 4, false>(vector, codes, dim, reorder_bits, results);
#else
    neon::RaBitQFloatThreeBitIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, reorder_bits, results);
#endif
}

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return RaBitQPlanarIP(vector, one_bit_code, supplement_code, dim, supplement_bits);
#else
    return neon::RaBitQFloatSplitCodeIP(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
#endif
}

float
RaBitQFloatSupplementCodeIP(const float* vector,
                            const uint8_t* supplement_code,
                            uint64_t dim,
                            uint32_t supplement_bits) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return RaBitQPlanarIP(vector, nullptr, supplement_code, dim, supplement_bits);
#else
    return neon::RaBitQFloatSupplementCodeIP(vector, supplement_code, dim, supplement_bits);
#endif
}

float
RaBitQFloatMultiBitIPByLookup(const float* lookup,
                              const uint8_t* bits,
                              uint64_t dim,
                              uint32_t reorder_bits,
                              uint32_t filter_bits) {
#if defined(ENABLE_SVE)
    const auto step = svcntw();
    const auto plane_bytes = (dim + 7) / 8;
    auto sum = svdup_f32(0.0F);
    const auto base_offsets = svindex_u32(0, 256);
    for (uint64_t block = 0; block < plane_bytes; block += step) {
        const auto pg = svwhilelt_b32(block, plane_bytes);
        const auto* table = lookup + block * 256;
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto offset = bit * plane_bytes + block;
            const auto indices = svadd_u32_x(pg, base_offsets, svld1ub_u32(pg, bits + offset));
            const auto values = svld1_gather_u32index_f32(pg, table, indices);
            const float weight = static_cast<float>(1U << (reorder_bits + filter_bits - bit - 1));
            sum = svadd_f32_m(pg, sum, svmul_n_f32_x(pg, values, weight));
        }
    }
    return svaddv_f32(svptrue_b32(), sum);
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
#if defined(ENABLE_SVE)
    const auto pg = svwhilelt_b32(uint64_t{0}, uint64_t{4});
    const auto plane_bytes = (dim + 7) / 8;
    auto sum = svdup_f32(0.0F);
    const uint8_t* codes[] = {bits1, bits2, bits3, bits4};
    for (uint64_t block = 0; block < plane_bytes; ++block) {
        const float* table = lookup + block * 256;
        for (uint32_t bit = 0; bit < filter_bits; ++bit) {
            const auto offset = bit * plane_bytes + block;
            const uint32_t indices[] = {
                codes[0][offset], codes[1][offset], codes[2][offset], codes[3][offset]};
            const auto values = svld1_gather_u32index_f32(pg, table, svld1_u32(pg, indices));
            const float weight = static_cast<float>(1U << (reorder_bits + filter_bits - bit - 1));
            sum = svadd_f32_m(pg, sum, svmul_n_f32_x(pg, values, weight));
        }
    }
    svst1_f32(pg, results, sum);
#else
    generic::RaBitQFloatMultiBitIPBatch4ByLookup(
        lookup, bits1, bits2, bits3, bits4, dim, reorder_bits, filter_bits, results);
#endif
}

float
RaBitQFloatThreeBitIPByLookup(const float* lookup,
                              const uint8_t* bits,
                              uint64_t dim,
                              uint32_t reorder_bits) {
    return sve::RaBitQFloatMultiBitIPByLookup(lookup, bits, dim, reorder_bits, 3);
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
    sve::RaBitQFloatMultiBitIPBatch4ByLookup(
        lookup, bits1, bits2, bits3, bits4, dim, reorder_bits, 3, results);
}

#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
namespace {
// A wire block is four groups of 16 dimensions, independent of SVE VL.
svuint32_t
DecodeCompactGroup(
    svbool_t pg, const uint8_t* data, uint32_t bits, uint32_t group, uint32_t offset, bool legacy) {
    const auto lanes = svindex_u32(offset, 1);
    svuint32_t code;
    if (bits == 5) {
        code = svand_n_u32_x(
            pg,
            svlsr_n_u32_x(pg, svld1ub_u32(pg, data + (group / 2) * 16 + offset), (group % 2) * 4),
            15);
    } else if (group < 3) {
        code = svand_n_u32_x(pg, svld1ub_u32(pg, data + group * 16 + offset), 63);
    } else {
        const auto a = svlsr_n_u32_x(pg, svld1ub_u32(pg, data + offset), 6);
        const auto b =
            svand_n_u32_x(pg, svlsr_n_u32_x(pg, svld1ub_u32(pg, data + 16 + offset), 4), 12);
        const auto c =
            svand_n_u32_x(pg, svlsr_n_u32_x(pg, svld1ub_u32(pg, data + 32 + offset), 2), 48);
        code = svorr_u32_x(pg, a, svorr_u32_x(pg, b, c));
    }
    if (bits != 6) {
        const auto lo = svand_n_u32_x(pg, lanes, 7);
        const auto half = svlsr_n_u32_x(pg, lanes, 3);
        const auto indices = legacy ? lo : svadd_n_u32_x(pg, half, group * 2);
        const auto shifts = legacy ? svadd_n_u32_x(pg, half, group * 2) : lo;
        const auto plane = svld1ub_gather_u32offset_u32(pg, data + (bits == 5 ? 32 : 48), indices);
        const auto bit = svand_n_u32_x(pg, svlsr_u32_x(pg, plane, shifts), 1);
        code = svorr_u32_x(pg, code, svlsl_n_u32_x(pg, bit, bits - 1));
    }
    return code;
}

float
CompactDot(const float* q, const uint8_t* packed, uint64_t blocks, uint32_t bits, bool legacy) {
    const auto step = svcntw();
    auto sum = svdup_f32(0.0F);
    for (uint64_t chunk = 0; chunk < blocks; ++chunk, packed += bits * 8) {
        for (uint32_t group = 0; group < 4; ++group) {
            for (uint32_t i = 0; i < 16; i += step) {
                const auto pg = svwhilelt_b32(i, 16U);
                const auto code = DecodeCompactGroup(pg, packed, bits, group, i, legacy);
                sum = svmla_f32_m(pg,
                                  sum,
                                  svld1_f32(pg, q + 64 * chunk + group * 16 + i),
                                  svcvt_f32_u32_x(pg, code));
            }
        }
    }
    return svaddv_f32(svptrue_b32(), sum);
}
}  // namespace
#endif

float
RaBitQFloatPackedSupplementCodeIP(const float* vector,
                                  const uint8_t* packed,
                                  uint64_t dim,
                                  uint32_t bits) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const simd::RaBitQPackedSupplementLayout layout{dim, bits};
    if (layout.PackedSize() == 0) {
        return 0.0F;
    }
    const auto value = CompactDot(vector, packed, layout.FullChunkCount(), bits, false);
    return value + RaBitQPlanarIP(vector + layout.FullDimension(),
                                  nullptr,
                                  packed + layout.FullChunksSize(),
                                  layout.TailDimension(),
                                  bits);
#elif defined(ENABLE_NEON)
    return neon::RaBitQFloatPackedSupplementCodeIP(vector, packed, dim, bits);
#else
    return generic::RaBitQFloatPackedSupplementCodeIP(vector, packed, dim, bits);
#endif
}

float
RaBitQFloatExCode7IP(const float* vector, const uint8_t* packed, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return (dim & 63U) != 0U ? 0.0F : CompactDot(vector, packed, dim / 64, 7, true);
#elif defined(ENABLE_NEON)
    return neon::RaBitQFloatExCode7IP(vector, packed, dim);
#else
    return generic::RaBitQFloatExCode7IP(vector, packed, dim);
#endif
}

#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
namespace {
svuint32_t
DecodeLegacyWithFilter(
    svbool_t pg, const uint8_t* packed, const uint8_t* filter, uint32_t group, uint32_t offset) {
    const auto lanes = svindex_u32(offset, 1);
    const auto bytes =
        svld1ub_gather_u32offset_u32(pg, filter + group * 2, svlsr_n_u32_x(pg, lanes, 3));
    const auto sign = svand_n_u32_x(pg, svlsr_u32_x(pg, bytes, svand_n_u32_x(pg, lanes, 7)), 1);
    return svorr_u32_x(
        pg, DecodeCompactGroup(pg, packed, 7, group, offset, true), svlsl_n_u32_x(pg, sign, 7));
}
}  // namespace
#endif

bool
RaBitQExCode7ToBytes(const uint8_t* filter,
                     const uint8_t* packed,
                     uint64_t dim,
                     uint8_t* out,
                     uint64_t* code_sum,
                     uint64_t* code_sq_sum) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    if (!filter || !packed || !out || !code_sum || !RaBitQExCode7SupportedDim(dim)) {
        return false;
    }
    uint64_t sum = 0, sq = 0;
    const auto step = svcntw();
    for (uint64_t block = 0; block < dim; block += 64, filter += 8, packed += 56) {
        for (uint32_t group = 0; group < 4; ++group) {
            for (uint32_t i = 0; i < 16; i += step) {
                const auto pg = svwhilelt_b32(i, 16U);
                const auto code = DecodeLegacyWithFilter(pg, packed, filter, group, i);
                svst1b_u32(pg, out + block + 16 * group + i, code);
                sum += svaddv_u32(pg, code);
                if (code_sq_sum) {
                    sq += svaddv_u32(pg, svmul_u32_x(pg, code, code));
                }
            }
        }
    }
    *code_sum = sum;
    if (code_sq_sum) {
        *code_sq_sum = sq;
    }
    return true;
#elif defined(ENABLE_NEON)
    return neon::RaBitQExCode7ToBytes(filter, packed, dim, out, code_sum, code_sq_sum);
#else
    return generic::RaBitQExCode7ToBytes(filter, packed, dim, out, code_sum, code_sq_sum);
#endif
}

bool
RaBitQExCode7ToVector(const uint8_t* filter,
                      const uint8_t* packed,
                      const float* centroid,
                      float residual_scale,
                      float full_center,
                      uint64_t dim,
                      float* out) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    if (!filter || !packed || !centroid || !out || !RaBitQExCode7SupportedDim(dim)) {
        return false;
    }
    const auto step = svcntw();
    bool finite = true;
    for (uint64_t block = 0; block < dim; block += 64, filter += 8, packed += 56) {
        for (uint32_t group = 0; group < 4; ++group) {
            for (uint32_t i = 0; i < 16; i += step) {
                const auto pg = svwhilelt_b32(i, 16U);
                const auto code = DecodeLegacyWithFilter(pg, packed, filter, group, i);
                const auto j = block + group * 16 + i;
                const auto centered = svsub_n_f32_x(pg, svcvt_f32_u32_x(pg, code), full_center);
                const auto v =
                    svmla_n_f32_x(pg, svld1_f32(pg, centroid + j), centered, residual_scale);
                svst1_f32(pg, out + j, v);
                const auto exponent = svand_n_u32_x(pg, svreinterpret_u32_f32(v), 0x7f800000U);
                finite &= !svptest_any(pg, svcmpeq_n_u32(pg, exponent, 0x7f800000U));
            }
        }
    }
    return finite;
#elif defined(ENABLE_NEON)
    return neon::RaBitQExCode7ToVector(
        filter, packed, centroid, residual_scale, full_center, dim, out);
#else
    return generic::RaBitQExCode7ToVector(
        filter, packed, centroid, residual_scale, full_center, dim, out);
#endif
}

float
RaBitQFloatSQIP(const float* vector, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const auto step = svcntw();
    auto sum = svdup_f32(0.0F);
    for (uint64_t i = 0; i < dim; i += step) {
        const auto pg = svwhilelt_b32(i, dim);
        const auto code = svcvt_f32_u32_x(pg, svld1ub_u32(pg, codes + i));
        sum = svmla_f32_m(pg, sum, svld1_f32(pg, vector + i), code);
    }
    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::RaBitQFloatSQIP(vector, codes, dim);
#endif
}

uint64_t
RaBitQCodeCodeIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    const auto step = svcntw();
    uint64_t sum = 0;
    for (uint64_t i = 0; i < dim; i += step) {
        const auto pg = svwhilelt_b32(i, dim);
        const auto a = svld1ub_u32(pg, codes1 + i);
        const auto b = svld1ub_u32(pg, codes2 + i);
        sum += svaddv_u32(pg, svmul_u32_x(pg, a, b));
    }
    return sum;
#else
    return neon::RaBitQCodeCodeIP(codes1, codes2, dim);
#endif
}

#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
namespace {
void
PackBitPlaneSVE(const uint8_t* codes, uint8_t* out, uint64_t dim, uint32_t bit) {
    const auto step = svcntb();
    const auto all = svptrue_b8();
    const auto weights = svand_n_u8_x(all, svindex_u8(0, 1), 7);
    uint8_t lanes[256];
    for (uint64_t i = 0; i < dim; i += step) {
        const auto code = svld1_u8(svwhilelt_b8(i, dim), codes + i);
        const auto set = svand_n_u8_x(all, svlsr_n_u8_x(all, code, bit), 1);
        svst1_u8(all, lanes, svlsl_u8_x(all, set, weights));
        const auto bytes = std::min((dim - i + 7) / 8, step / 8);
        for (uint64_t j = 0; j < bytes; ++j) {
            const auto* x = lanes + j * 8;
            out[i / 8 + j] = x[0] | x[1] | x[2] | x[3] | x[4] | x[5] | x[6] | x[7];
        }
    }
}

void
PackCompactSupplementSVE(const uint8_t* values, uint8_t* out, uint32_t bits) {
    const auto mask = svwhilelt_b8(uint64_t{0}, uint64_t{16});
    const auto a = svld1_u8(mask, values);
    const auto b = svld1_u8(mask, values + 16);
    const auto c = svld1_u8(mask, values + 32);
    const auto d = svld1_u8(mask, values + 48);
    if (bits == 5) {
        const auto pair = [&](svuint8_t low, svuint8_t high) {
            return svorr_u8_x(mask, svand_n_u8_x(mask, low, 15), svlsl_n_u8_x(mask, high, 4));
        };
        svst1_u8(mask, out, pair(a, b));
        svst1_u8(mask, out + 16, pair(c, d));
        PackBitPlaneSVE(values, out + 32, 64, 4);
    } else {
        const auto pair = [&](svuint8_t x, unsigned shift) {
            const auto extra = svlsr_n_u8_x(mask, d, shift);
            return svorr_u8_x(mask, svand_n_u8_x(mask, x, 63), svlsl_n_u8_x(mask, extra, 6));
        };
        svst1_u8(mask, out, pair(a, 0));
        svst1_u8(mask, out + 16, pair(b, 2));
        svst1_u8(mask, out + 32, pair(c, 4));
        if (bits == 7) {
            PackBitPlaneSVE(values, out + 48, 64, 6);
        }
    }
}
}  // namespace
#endif

void
RaBitQPackScalarToSplitCode(const uint8_t* codes,
                            uint8_t* filter,
                            uint8_t* supplement,
                            uint64_t dim,
                            uint32_t total_bits,
                            uint32_t filter_bits) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    if (dim == 0) {
        return;
    }
    const simd::RaBitQPackedSupplementLayout layout{dim, total_bits - filter_bits};
    for (uint32_t i = 0; i < filter_bits; ++i) {
        PackBitPlaneSVE(codes, filter + i * layout.PlaneBytes(), dim, total_bits - 1 - i);
    }
    if (layout.PackedSize() == 0) {
        return;
    }
    for (uint64_t i = 0; i < layout.FullChunkCount(); ++i) {
        PackCompactSupplementSVE(
            codes + i * layout.CHUNK_DIM, supplement + i * layout.ChunkBytes(), layout.bits);
    }
    if (layout.TailDimension() != 0) {
        for (uint32_t bit = 0; bit < layout.bits; ++bit) {
            PackBitPlaneSVE(codes + layout.FullDimension(),
                            supplement + layout.FullChunksSize() + bit * layout.TailPlaneBytes(),
                            layout.TailDimension(),
                            bit);
        }
    }
#elif defined(ENABLE_NEON)
    neon::RaBitQPackScalarToSplitCode(codes, filter, supplement, dim, total_bits, filter_bits);
#else
    generic::RaBitQPackScalarToSplitCode(codes, filter, supplement, dim, total_bits, filter_bits);
#endif
}

void
RaBitQPackScalarToSplitPlanes(const uint8_t* scalar_codes,
                              uint8_t* filter_planes,
                              uint8_t* supplement_planes,
                              uint64_t dim,
                              uint32_t total_bits,
                              uint32_t filter_bits) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    if (dim == 0) {
        return;
    }
    const auto plane_bytes = (dim + 7) / 8;
    for (uint32_t bit = 0; bit < total_bits; ++bit) {
        auto* out = simd::RaBitQGetSplitPlane(
            filter_planes, supplement_planes, plane_bytes, total_bits, filter_bits, bit);
        PackBitPlaneSVE(scalar_codes, out, dim, bit);
    }
#else
    neon::RaBitQPackScalarToSplitPlanes(
        scalar_codes, filter_planes, supplement_planes, dim, total_bits, filter_bits);
#endif
}

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 0;
    }

    uint32_t result = 0;
    uint64_t num_bytes = (dim + 7) / 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        uint64_t i = 0;
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

#if defined(ENABLE_SVE)
namespace {
template <size_t Count>
void
RaBitQCountWithBaseSum(const uint8_t* codes,
                       const uint8_t* const (&bases)[Count],
                       uint64_t dim,
                       uint64_t* results) {
    const auto bytes = (dim + 7) / 8;
    const auto step = svcntb();
    uint32_t sums[Count]{};
    uint32_t ips[Count]{};
    for (uint64_t i = 0; i < bytes; i += step) {
        const auto pg = svwhilelt_b8(i, bytes);
        const auto p0 = svld1_u8(pg, codes + i);
        const auto p1 = svld1_u8(pg, codes + bytes + i);
        const auto p2 = svld1_u8(pg, codes + 2 * bytes + i);
        const auto p3 = svld1_u8(pg, codes + 3 * bytes + i);
        const auto count = [&](svuint8_t v) { return svaddv_u8(pg, svcnt_u8_x(pg, v)); };
        for (size_t j = 0; j < Count; ++j) {
            const auto base = svld1_u8(pg, bases[j] + i);
            sums[j] += count(base);
            ips[j] += count(svand_u8_x(pg, p0, base)) + (count(svand_u8_x(pg, p1, base)) << 1) +
                      (count(svand_u8_x(pg, p2, base)) << 2) +
                      (count(svand_u8_x(pg, p3, base)) << 3);
        }
    }
    for (size_t j = 0; j < Count; ++j) {
        results[j] = ips[j] | (static_cast<uint64_t>(sums[j]) << 32);
    }
}
}  // namespace
#endif

uint64_t
RaBitQSQ4UBinaryIPWithBaseSum(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE)
    const uint8_t* bases[] = {bits};
    uint64_t result;
    RaBitQCountWithBaseSum(codes, bases, dim, &result);
    return result;
#else
    return neon::RaBitQSQ4UBinaryIPWithBaseSum(codes, bits, dim);
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
#if defined(ENABLE_SVE)
    const uint8_t* bases[] = {bits1, bits2, bits3, bits4};
    RaBitQCountWithBaseSum(codes, bases, dim, results);
#else
    neon::RaBitQSQ4UBinaryIPWithBaseSumBatch4(codes, bits1, bits2, bits3, bits4, dim, results);
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
#if defined(ENABLE_SVE)
    __builtin_prefetch(data, 0, 3);
#endif
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
    const float norm = std::sqrt(sve::FP32ComputeIP(from, from, dim));
    sve::DivScalar(from, to, dim, norm);
    return norm;
#else
    return neon::Normalize(from, to, dim);
#endif
}

__attribute__((no_sanitize("address", "undefined"))) void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_SVE)
    const auto step = svcntb();
    // Each 128-bit segment accumulates one byte per iteration. Drain before
    // its uint16 lanes wrap, keeping the existing loop for ordinary PQ sizes.
    const uint64_t max_groups = (UINT16_MAX / UINT8_MAX) * (step / 16);
    while (pq_dim > max_groups) {
        sve::PQFastScanLookUp32(lookup_table, codes, max_groups, result);
        lookup_table += max_groups * 16;
        codes += max_groups * 16;
        pq_dim -= max_groups;
    }
    uint64_t i = 0;
    const uint64_t total_bytes = pq_dim * 16;

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
PQFastScanLookUp32HighAcc(const uint8_t* RESTRICT low_lookup_table,
                          const uint8_t* RESTRICT high_lookup_table,
                          const uint8_t* RESTRICT codes,
                          uint64_t pq_dim,
                          int32_t* RESTRICT result) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    // Reuse the byte lookup kernel for both halves of each 16-bit LUT entry.
    // Limit each call to 128 groups so its uint16 lane accumulators cannot wrap.
    constexpr uint64_t kChunk = 128;
    for (uint64_t group = 0; group < pq_dim;) {
        const auto count = std::min(kChunk, pq_dim - group);
        int32_t low[32]{};
        int32_t high[32]{};
        const auto offset = group * 16;
        sve::PQFastScanLookUp32(low_lookup_table + offset, codes + offset, count, low);
        sve::PQFastScanLookUp32(high_lookup_table + offset, codes + offset, count, high);
        const auto step = svcntw();
        for (uint64_t lane = 0; lane < 32; lane += step) {
            const auto pg = svwhilelt_b32(lane, uint64_t{32});
            const auto hi = svlsl_n_s32_x(pg, svld1_s32(pg, high + lane), 8);
            const auto sum = svadd_s32_x(pg, svld1_s32(pg, low + lane), hi);
            svst1_s32(pg, result + lane, svadd_s32_x(pg, svld1_s32(pg, result + lane), sum));
        }
        group += count;
    }
#else
    neon::PQFastScanLookUp32HighAcc(low_lookup_table, high_lookup_table, codes, pq_dim, result);
#endif
}

void
PQFastScanLookUp32HighAccOverwrite(const uint8_t* RESTRICT low_lookup_table,
                                   const uint8_t* RESTRICT high_lookup_table,
                                   const uint8_t* RESTRICT codes,
                                   uint64_t pq_dim,
                                   int32_t* RESTRICT result) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    std::fill_n(result, 32, 0);
    sve::PQFastScanLookUp32HighAcc(low_lookup_table, high_lookup_table, codes, pq_dim, result);
#else
    neon::PQFastScanLookUp32HighAccOverwrite(
        low_lookup_table, high_lookup_table, codes, pq_dim, result);
#endif
}

uint32_t
FP32LessThan32Mask(const float* values, float limit) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    uint32_t mask = 0;
    // The result has only 32 lanes, even on a wider SVE implementation.
    uint32_t matches[32];
    const auto step = svcntw();
    for (uint64_t i = 0; i < 32; i += step) {
        const auto pg = svwhilelt_b32(i, uint64_t{32});
        const auto less = svcmplt_n_f32(pg, svld1_f32(pg, values + i), limit);
        svst1_u32(pg, matches + i, svsel_u32(less, svdup_u32(1), svdup_u32(0)));
    }
    for (uint32_t lane = 0; lane < 32; ++lane) {
        mask |= matches[lane] << lane;
    }
    return mask;
#else
    return neon::FP32LessThan32Mask(values, limit);
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
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    uint32_t computed_mask = 0;
    const float center = 0.5F * static_cast<float>((1U << filter_bits) - 1U) * query_sum;
    const auto step = svcntw();
    for (uint64_t i = 0; i < valid_size; i += step) {
        const auto pg = svwhilelt_b32(i, static_cast<uint64_t>(valid_size));
        const auto load_lookup = [&](uint32_t plane) {
            const auto acc = svld1_s32(pg, accumulators + plane * 32 + i);
            return svadd_n_f32_x(pg, svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, acc), delta), sum_vl);
        };
        auto ip = svdup_f32(0.0F);
        if (filter_bits == 1) {
            ip =
                svmul_n_f32_x(pg,
                              svsub_n_f32_x(pg, svmul_n_f32_x(pg, load_lookup(0), 2.0F), query_sum),
                              inv_sqrt_d);
        } else {
            for (uint32_t plane = 0; plane < filter_bits; ++plane) {
                const auto weight = static_cast<float>(1U << (filter_bits - plane - 1));
                ip = svadd_f32_x(pg, ip, svmul_n_f32_x(pg, load_lookup(plane), weight));
            }
            ip = svdiv_f32_x(
                pg, svsub_n_f32_x(pg, ip, center), svld1_f32(pg, filter_norm_codes + i));
        }
        const auto scale = svld1_f32(pg, f_scale + i);
        const auto add = svld1_f32(pg, f_add + i);
        const auto result = svadd_f32_x(pg,
                                        svadd_n_f32_x(pg, add, query_bucket_norm_sqr),
                                        svmul_f32_x(pg, svmul_n_f32_x(pg, scale, query_norm), ip));
        svst1_f32(pg, dists + i, result);
        if (filter_inner_products) {
            svst1_f32(pg, filter_inner_products + i, ip);
        }
        for (uint64_t lane = i; lane < std::min<uint64_t>(i + step, valid_size); ++lane) {
            const bool valid = IsFiniteFloatBits(f_add[lane]) && IsFiniteFloatBits(f_scale[lane]) &&
                               std::abs(f_scale[lane]) > std::numeric_limits<float>::epsilon() &&
                               (filter_bits == 1 || filter_norm_codes[lane] > 0.0F) &&
                               IsFiniteFloatBits(dists[lane]);
            if (valid) {
                computed_mask |= 1U << lane;
            } else {
                dists[lane] = std::numeric_limits<float>::max();
                if (filter_inner_products) {
                    filter_inner_products[lane] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
    }
    return computed_mask;
#else
    return neon::RaBitQFastScan32ResidualPostprocess(accumulators,
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
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_SVE)
    uint64_t n = len / 2;
    uint64_t offset = (len % 2) + n;
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
FlipSign(const uint8_t* flip, float* data, uint64_t dim) {
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
VecRescale(float* data, uint64_t dim, float val) {
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
FHTRotate(float* data, uint64_t dim_) {
#if defined(ENABLE_SVE)
    uint64_t n = dim_;
    uint64_t step = 1;
    while (step < n) {
        sve::RotateOp(data, 0, dim_, step);
        step *= 2;
    }
#else
    neon::FHTRotate(data, dim_);
#endif
}

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 1.0f;
    }

    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t f_vec = svld1_f32(predicate, from + i);
        svfloat32_t c_vec = svld1_f32(predicate, centroid + i);
        svfloat32_t diff = svsub_f32_z(predicate, f_vec, c_vec);
        sum = svmla_f32_m(predicate, sum, diff, diff);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    float norm_sq = svaddv_f32(svptrue_b32(), sum);
    float norm = 0;
    if (norm_sq < 1e-5f) {
        norm = 1.0f;
    } else {
        norm = std::sqrt(norm_sq);
    }

    svfloat32_t normVec = svdup_f32(norm);
    i = 0;
    predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t f_vec = svld1_f32(predicate, from + i);
        svfloat32_t c_vec = svld1_f32(predicate, centroid + i);
        svfloat32_t diff = svsub_f32_z(predicate, f_vec, c_vec);
        svfloat32_t result = svdiv_f32_z(predicate, diff, normVec);
        svst1_f32(predicate, to + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return norm;
#else
    return neon::NormalizeWithCentroid(from, centroid, to, dim);
#endif
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return;
    }

    svfloat32_t normVec = svdup_f32(norm);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t f_vec = svld1_f32(predicate, from + i);
        svfloat32_t c_vec = svld1_f32(predicate, centroid + i);
        svfloat32_t result = svmad_f32_z(predicate, f_vec, normVec, c_vec);
        svst1_f32(predicate, to + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::InverseNormalizeWithCentroid(from, centroid, to, dim, norm);
#endif
}

}  // namespace vsag::sve

namespace vsag::sve {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
namespace {
template <int Bits>
svfloat32_t
DecodeLocal(const uint8_t* codes, uint64_t i, svbool_t pg, svfloat32_t minimum, svfloat32_t step) {
    svuint32_t values;
    if constexpr (Bits == 8) {
        values = svld1ub_u32(pg, codes + i);
    } else {
        const auto indices = svindex_u32(0, 1);
        values = svld1ub_gather_u32offset_u32(pg, codes + i / 2, svlsr_n_u32_x(pg, indices, 1));
        const auto shifts = svlsl_n_u32_x(pg, svand_n_u32_x(pg, indices, 1), 2);
        values = svand_n_u32_x(pg, svlsr_u32_x(pg, values, shifts), 15);
    }
    return svmla_f32_x(pg, minimum, svcvt_f32_u32_x(pg, values), step);
}

template <int Bits, bool Adc, bool Ip>
float
LocalDistance(const float* query, const uint8_t* a, const uint8_t* b, uint64_t dim) {
    const auto fallback = [&]() {
        if constexpr (Bits == 4) {
            if constexpr (Adc) {
                return Ip ? generic::SQ4LocalComputeIP(query, b, dim)
                          : generic::SQ4LocalComputeL2Sqr(query, b, dim);
            } else {
                return Ip ? generic::SQ4LocalComputeCodesIP(a, b, dim)
                          : generic::SQ4LocalComputeCodesL2Sqr(a, b, dim);
            }
        } else if constexpr (Adc) {
            return Ip ? generic::SQ8LocalComputeIP(query, b, dim)
                      : generic::SQ8LocalComputeL2Sqr(query, b, dim);
        } else {
            return Ip ? generic::SQ8LocalComputeCodesIP(a, b, dim)
                      : generic::SQ8LocalComputeCodesL2Sqr(a, b, dim);
        }
    };
    float min_a = 0.0F, step_a = 0.0F, min_b, step_b;
    std::memcpy(&min_b, b, sizeof(float));
    std::memcpy(&step_b, b + sizeof(float), sizeof(float));
    if constexpr (!Adc) {
        std::memcpy(&min_a, a, sizeof(float));
        std::memcpy(&step_a, a + sizeof(float), sizeof(float));
    }
    if (std::abs(min_a) > 1e30F || std::abs(min_b) > 1e30F || step_a > 1e30F || step_b > 1e30F) {
        return fallback();
    }
    auto sum = svdup_f32(0.0F);
    const auto vb_min = svdup_f32(min_b), vb_step = svdup_f32(step_b);
    const auto va_min = svdup_f32(min_a), va_step = svdup_f32(step_a);
    for (uint64_t i = 0; i < dim; i += svcntw()) {
        const auto pg = svwhilelt_b32(i, dim);
        const auto vb = DecodeLocal<Bits>(b + 2 * sizeof(float), i, pg, vb_min, vb_step);
        svfloat32_t va;
        if constexpr (Adc) {
            va = svld1_f32(pg, query + i);
        } else {
            va = DecodeLocal<Bits>(a + 2 * sizeof(float), i, pg, va_min, va_step);
        }
        if constexpr (Ip) {
            sum = svmla_f32_m(pg, sum, va, vb);
        } else {
            const auto delta = svsub_f32_x(pg, va, vb);
            sum = svmla_f32_m(pg, sum, delta, delta);
        }
    }
    const float result = svaddv_f32(svptrue_b32(), sum);
    uint32_t bits;
    std::memcpy(&bits, &result, sizeof(bits));
    return (bits & 0x7f800000U) == 0x7f800000U ? fallback() : result;
}
}  // namespace
#endif

float
SQ8LocalComputeL2Sqr(const float* query, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return LocalDistance<8, true, false>(query, nullptr, codes, dim);
#else
    return neon::SQ8LocalComputeL2Sqr(query, codes, dim);
#endif
}
float
SQ8LocalComputeCodesL2Sqr(const uint8_t* a, const uint8_t* b, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return LocalDistance<8, false, false>(nullptr, a, b, dim);
#else
    return neon::SQ8LocalComputeCodesL2Sqr(a, b, dim);
#endif
}

float
SQ8LocalComputeIP(const float* query, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return LocalDistance<8, true, true>(query, nullptr, codes, dim);
#else
    return neon::SQ8LocalComputeIP(query, codes, dim);
#endif
}
float
SQ8LocalComputeCodesIP(const uint8_t* a, const uint8_t* b, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return LocalDistance<8, false, true>(nullptr, a, b, dim);
#else
    return neon::SQ8LocalComputeCodesIP(a, b, dim);
#endif
}

float
SQ4LocalComputeL2Sqr(const float* query, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return LocalDistance<4, true, false>(query, nullptr, codes, dim);
#else
    return neon::SQ4LocalComputeL2Sqr(query, codes, dim);
#endif
}
float
SQ4LocalComputeCodesL2Sqr(const uint8_t* a, const uint8_t* b, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return LocalDistance<4, false, false>(nullptr, a, b, dim);
#else
    return neon::SQ4LocalComputeCodesL2Sqr(a, b, dim);
#endif
}

float
SQ4LocalComputeIP(const float* query, const uint8_t* codes, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return LocalDistance<4, true, true>(query, nullptr, codes, dim);
#else
    return neon::SQ4LocalComputeIP(query, codes, dim);
#endif
}
float
SQ4LocalComputeCodesIP(const uint8_t* a, const uint8_t* b, uint64_t dim) {
#if defined(ENABLE_SVE) && !defined(ENABLE_NEON)
    return LocalDistance<4, false, true>(nullptr, a, b, dim);
#else
    return neon::SQ4LocalComputeCodesIP(a, b, dim);
#endif
}
}  // namespace vsag::sve
