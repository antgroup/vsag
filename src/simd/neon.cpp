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
#if defined(ENABLE_NEON)
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstdint>

#include "simd.h"

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace vsag::neon {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return neon::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return neon::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - neon::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);

    return neon::INT8ComputeL2Sqr(pVect1, pVect2, qty);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((size_t*)qty_ptr);
    return neon::INT8ComputeIP(pVect1, pVect2, qty);
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -neon::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

#if defined(ENABLE_NEON)
__inline float32x4x3_t __attribute__((__always_inline__)) vcvt3_f32_f16(const float16x4x3_t a) {
    float32x4x3_t c;
    c.val[0] = vcvt_f32_f16(a.val[0]);
    c.val[1] = vcvt_f32_f16(a.val[1]);
    c.val[2] = vcvt_f32_f16(a.val[2]);
    return c;
}

__inline float32x4x2_t __attribute__((__always_inline__)) vcvt2_f32_f16(const float16x4x2_t a) {
    float32x4x2_t c;
    c.val[0] = vcvt_f32_f16(a.val[0]);
    c.val[1] = vcvt_f32_f16(a.val[1]);
    return c;
}

__inline float32x4x3_t __attribute__((__always_inline__)) vcvt3_f32_half(const uint16x4x3_t x) {
    float32x4x3_t c;
    c.val[0] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[0]), 16));
    c.val[1] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[1]), 16));
    c.val[2] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[2]), 16));
    return c;
}

__inline float32x4x2_t __attribute__((__always_inline__)) vcvt2_f32_half(const uint16x4x2_t x) {
    float32x4x2_t c;
    c.val[0] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[0]), 16));
    c.val[1] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[1]), 16));
    return c;
}
__inline float32x4_t __attribute__((__always_inline__)) vcvt_f32_half(const uint16x4_t x) {
    return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x), 16));
}

#endif

// calculate the dist between each pq kmeans centers and corresponding pq query dim value.
void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_NEON)
    const auto* float_centers = (const float*)single_dim_centers;
    auto* float_result = (float*)result;
    for (uint64_t idx = 0; idx < 256; idx += 8) {
        float32x4x2_t v_centers_dim = vld1q_f32_x2(float_centers + idx);
        float32x4x2_t v_query_vec = {vdupq_n_f32(single_dim_val), vdupq_n_f32(single_dim_val)};

        float32x4x2_t v_diff;
        v_diff.val[0] = vsubq_f32(v_centers_dim.val[0], v_query_vec.val[0]);
        v_diff.val[1] = vsubq_f32(v_centers_dim.val[1], v_query_vec.val[1]);

        float32x4x2_t v_diff_sq;
        v_diff_sq.val[0] = vmulq_f32(v_diff.val[0], v_diff.val[0]);
        v_diff_sq.val[1] = vmulq_f32(v_diff.val[1], v_diff.val[1]);

        float32x4x2_t v_chunk_dists = vld1q_f32_x2(&float_result[idx]);
        v_chunk_dists.val[0] = vaddq_f32(v_chunk_dists.val[0], v_diff_sq.val[0]);
        v_chunk_dists.val[1] = vaddq_f32(v_chunk_dists.val[1], v_diff_sq.val[1]);
        vst1q_f32_x2(&float_result[idx], v_chunk_dists);
    }
#else
    return generic::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    float32x4_t sum_ = vdupq_n_f32(0.0f);
    auto d = dim;
    while (d >= 12) {
        float32x4x3_t a = vld1q_f32_x3(query + dim - d);
        float32x4x3_t b = vld1q_f32_x3(codes + dim - d);
        float32x4x3_t c;
        c.val[0] = vmulq_f32(a.val[0], b.val[0]);
        c.val[1] = vmulq_f32(a.val[1], b.val[1]);
        c.val[2] = vmulq_f32(a.val[2], b.val[2]);

        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        c.val[0] = vaddq_f32(c.val[0], c.val[2]);

        sum_ = vaddq_f32(sum_, c.val[0]);
        d -= 12;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(query + dim - d);
        float32x4x2_t b = vld1q_f32_x2(codes + dim - d);
        float32x4x2_t c;
        c.val[0] = vmulq_f32(a.val[0], b.val[0]);
        c.val[1] = vmulq_f32(a.val[1], b.val[1]);
        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        sum_ = vaddq_f32(sum_, c.val[0]);
        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(query + dim - d);
        float32x4_t b = vld1q_f32(codes + dim - d);
        float32x4_t c;
        c = vmulq_f32(a, b);
        sum_ = vaddq_f32(sum_, c);
        d -= 4;
    }

    float32x4_t res_x = vdupq_n_f32(0.0f);
    float32x4_t res_y = vdupq_n_f32(0.0f);
    if (d >= 3) {
        res_x = vld1q_lane_f32(query + dim - d, res_x, 2);
        res_y = vld1q_lane_f32(codes + dim - d, res_y, 2);
        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(query + dim - d, res_x, 1);
        res_y = vld1q_lane_f32(codes + dim - d, res_y, 1);
        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(query + dim - d, res_x, 0);
        res_y = vld1q_lane_f32(codes + dim - d, res_y, 0);
        d -= 1;
    }

    sum_ = vaddq_f32(sum_, vmulq_f32(res_x, res_y));
    return vaddvq_f32(sum_);
#else
    return generic::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    float32x4_t sum_ = vdupq_n_f32(0.0f);
    auto d = dim;
    while (d >= 12) {
        float32x4x3_t a = vld1q_f32_x3(query + dim - d);
        float32x4x3_t b = vld1q_f32_x3(codes + dim - d);
        float32x4x3_t c;

        c.val[0] = vsubq_f32(a.val[0], b.val[0]);
        c.val[1] = vsubq_f32(a.val[1], b.val[1]);
        c.val[2] = vsubq_f32(a.val[2], b.val[2]);

        c.val[0] = vmulq_f32(c.val[0], c.val[0]);
        c.val[1] = vmulq_f32(c.val[1], c.val[1]);
        c.val[2] = vmulq_f32(c.val[2], c.val[2]);

        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        c.val[0] = vaddq_f32(c.val[0], c.val[2]);

        sum_ = vaddq_f32(sum_, c.val[0]);
        d -= 12;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(query + dim - d);
        float32x4x2_t b = vld1q_f32_x2(codes + dim - d);
        float32x4x2_t c;
        c.val[0] = vsubq_f32(a.val[0], b.val[0]);
        c.val[1] = vsubq_f32(a.val[1], b.val[1]);

        c.val[0] = vmulq_f32(c.val[0], c.val[0]);
        c.val[1] = vmulq_f32(c.val[1], c.val[1]);

        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        sum_ = vaddq_f32(sum_, c.val[0]);
        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(query + dim - d);
        float32x4_t b = vld1q_f32(codes + dim - d);
        float32x4_t c;
        c = vsubq_f32(a, b);
        c = vmulq_f32(c, c);

        sum_ = vaddq_f32(sum_, c);
        d -= 4;
    }

    float32x4_t res_x = vdupq_n_f32(0.0f);
    float32x4_t res_y = vdupq_n_f32(0.0f);
    if (d >= 3) {
        res_x = vld1q_lane_f32(query + dim - d, res_x, 2);
        res_y = vld1q_lane_f32(codes + dim - d, res_y, 2);
        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(query + dim - d, res_x, 1);
        res_y = vld1q_lane_f32(codes + dim - d, res_y, 1);
        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(query + dim - d, res_x, 0);
        res_y = vld1q_lane_f32(codes + dim - d, res_y, 0);
        d -= 1;
    }

    sum_ = vaddq_f32(sum_, vmulq_f32(vsubq_f32(res_x, res_y), vsubq_f32(res_x, res_y)));
    return vaddvq_f32(sum_);
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
#if defined(ENABLE_NEON)
    if (dim < 4) {
        return generic::FP32ComputeIPBatch4(
            query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
    }
    float32x4_t sum1 = vdupq_n_f32(0);
    float32x4_t sum2 = vdupq_n_f32(0);
    float32x4_t sum3 = vdupq_n_f32(0);
    float32x4_t sum4 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t q = vld1q_f32(query + i);
        float32x4_t c1 = vld1q_f32(codes1 + i);
        float32x4_t c2 = vld1q_f32(codes2 + i);
        float32x4_t c3 = vld1q_f32(codes3 + i);
        float32x4_t c4 = vld1q_f32(codes4 + i);
        sum1 = vaddq_f32(sum1, vmulq_f32(q, c1));
        sum2 = vaddq_f32(sum2, vmulq_f32(q, c2));
        sum3 = vaddq_f32(sum3, vmulq_f32(q, c3));
        sum4 = vaddq_f32(sum4, vmulq_f32(q, c4));
    }

    result1 += vaddvq_f32(sum1);
    result2 += vaddvq_f32(sum2);
    result3 += vaddvq_f32(sum3);
    result4 += vaddvq_f32(sum4);

    if (i < dim) {
        generic::FP32ComputeIPBatch4(query + i,
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
#if defined(ENABLE_NEON)
    if (dim < 4) {
        return generic::FP32ComputeL2SqrBatch4(
            query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
    }
    float32x4_t sum1 = vdupq_n_f32(0);
    float32x4_t sum2 = vdupq_n_f32(0);
    float32x4_t sum3 = vdupq_n_f32(0);
    float32x4_t sum4 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t q = vld1q_f32(query + i);
        float32x4_t c1 = vld1q_f32(codes1 + i);
        float32x4_t c2 = vld1q_f32(codes2 + i);
        float32x4_t c3 = vld1q_f32(codes3 + i);
        float32x4_t c4 = vld1q_f32(codes4 + i);
        float32x4_t diff1 = vsubq_f32(q, c1);
        float32x4_t diff2 = vsubq_f32(q, c2);
        float32x4_t diff3 = vsubq_f32(q, c3);
        float32x4_t diff4 = vsubq_f32(q, c4);
        sum1 = vaddq_f32(sum1, vmulq_f32(diff1, diff1));
        sum2 = vaddq_f32(sum2, vmulq_f32(diff2, diff2));
        sum3 = vaddq_f32(sum3, vmulq_f32(diff3, diff3));
        sum4 = vaddq_f32(sum4, vmulq_f32(diff4, diff4));
    }

    result1 += vaddvq_f32(sum1);
    result2 += vaddvq_f32(sum2);
    result3 += vaddvq_f32(sum3);
    result4 += vaddvq_f32(sum4);

    if (i < dim) {
        generic::FP32ComputeL2SqrBatch4(query + i,
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
    return generic::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_NEON)
    if (dim < 4) {
        return generic::FP32Sub(x, y, z, dim);
    }
    int64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t a = vld1q_f32(x + i);
        float32x4_t b = vld1q_f32(y + i);
        float32x4_t c = vsubq_f32(a, b);
        vst1q_f32(z + i, c);
    }
    if (i < dim) {
        generic::FP32Sub(x + i, y + i, z + i, dim - i);
    }
#else
    return generic::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_NEON)
    if (dim < 4) {
        return generic::FP32Add(x, y, z, dim);
    }
    int64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t a = vld1q_f32(x + i);
        float32x4_t b = vld1q_f32(y + i);
        float32x4_t c = vaddq_f32(a, b);
        vst1q_f32(z + i, c);
    }
    if (i < dim) {
        generic::FP32Add(x + i, y + i, z + i, dim - i);
    }
#else
    return generic::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_NEON)
    if (dim < 4) {
        return generic::FP32Mul(x, y, z, dim);
    }
    int64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t a = vld1q_f32(x + i);
        float32x4_t b = vld1q_f32(y + i);
        float32x4_t c = vmulq_f32(a, b);
        vst1q_f32(z + i, c);
    }
    if (i < dim) {
        generic::FP32Mul(x + i, y + i, z + i, dim - i);
    }
#else
    return generic::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_NEON)
    if (dim < 4) {
        return generic::FP32Div(x, y, z, dim);
    }
    int64_t i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t a = vld1q_f32(x + i);
        float32x4_t b = vld1q_f32(y + i);
        float32x4_t c = vdivq_f32(a, b);
        vst1q_f32(z + i, c);
    }
    if (i < dim) {
        generic::FP32Div(x + i, y + i, z + i, dim - i);
    }
#else
    return generic::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_NEON)
    if (dim < 4) {
        return generic::FP32ReduceAdd(x, dim);
    }
    int i = 0;
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (; i + 3 < dim; i += 4) {
        float32x4_t a = vld1q_f32(x + i);
        sum = vaddq_f32(sum, a);
    }
    float result = vaddvq_f32(sum);
    if (i < dim) {
        result += generic::FP32ReduceAdd(x + i, dim - i);
    }
    return result;
#else
    return generic::FP32ReduceAdd(x, dim);
#endif
}

#if defined(ENABLE_NEON)
__inline uint16x8_t __attribute__((__always_inline__)) load_4_short(const uint16_t* data) {
    uint16_t tmp[] = {data[3], 0, data[2], 0, data[1], 0, data[0], 0};
    return vld1q_u16(tmp);
}
#endif

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    float32x4x3_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    const auto* query_bf16 = (const uint16_t*)(query);
    const auto* codes_bf16 = (const uint16_t*)(codes);
    while (dim >= 12) {
        float32x4x3_t a = vcvt3_f32_half(vld3_u16((const uint16_t*)query_bf16));
        float32x4x3_t b = vcvt3_f32_half(vld3_u16((const uint16_t*)codes_bf16));

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], b.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], b.val[2]);
        dim -= 12;
        query_bf16 += 12;
        codes_bf16 += 12;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    if (dim >= 8) {
        float32x4x2_t a = vcvt2_f32_half(vld2_u16((const uint16_t*)query_bf16));
        float32x4x2_t b = vcvt2_f32_half(vld2_u16((const uint16_t*)codes_bf16));
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], b.val[1]);
        dim -= 8;
        query_bf16 += 8;
        codes_bf16 += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (dim >= 4) {
        float32x4_t a = vcvt_f32_half(vld1_u16((const uint16_t*)query_bf16));
        float32x4_t b = vcvt_f32_half(vld1_u16((const uint16_t*)codes_bf16));
        res.val[0] = vmlaq_f32(res.val[0], a, b);
        dim -= 4;
        query_bf16 += 4;
        codes_bf16 += 4;
    }
    if (dim >= 0) {
        uint16x4_t res_x = {0, 0, 0, 0};
        uint16x4_t res_y = {0, 0, 0, 0};
        switch (dim) {
            case 3:
                res_x = vld1_lane_u16((const uint16_t*)query_bf16, res_x, 2);
                res_y = vld1_lane_u16((const uint16_t*)codes_bf16, res_y, 2);
                query_bf16++;
                codes_bf16++;
                dim--;
            case 2:
                res_x = vld1_lane_u16((const uint16_t*)query_bf16, res_x, 1);
                res_y = vld1_lane_u16((const uint16_t*)codes_bf16, res_y, 1);
                query_bf16++;
                codes_bf16++;
                dim--;
            case 1:
                res_x = vld1_lane_u16((const uint16_t*)query_bf16, res_x, 0);
                res_y = vld1_lane_u16((const uint16_t*)codes_bf16, res_y, 0);
                query_bf16++;
                codes_bf16++;
                dim--;
        }
        res.val[0] = vmlaq_f32(res.val[0], vcvt_f32_half(res_x), vcvt_f32_half(res_y));
    }
    return vaddvq_f32(res.val[0]);
#else
    return generic::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    float32x4x3_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    const auto* query_bf16 = (const uint16_t*)(query);
    const auto* codes_bf16 = (const uint16_t*)(codes);

    while (dim >= 12) {
        float32x4x3_t a = vcvt3_f32_half(vld3_u16((const uint16_t*)query_bf16));
        float32x4x3_t b = vcvt3_f32_half(vld3_u16((const uint16_t*)codes_bf16));
        a.val[0] = vsubq_f32(a.val[0], b.val[0]);
        a.val[1] = vsubq_f32(a.val[1], b.val[1]);
        a.val[2] = vsubq_f32(a.val[2], b.val[2]);

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], a.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], a.val[2]);
        dim -= 12;
        query_bf16 += 12;
        codes_bf16 += 12;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    if (dim >= 8) {
        float32x4x2_t a = vcvt2_f32_half(vld2_u16((const uint16_t*)query_bf16));
        float32x4x2_t b = vcvt2_f32_half(vld2_u16((const uint16_t*)codes_bf16));
        a.val[0] = vsubq_f32(a.val[0], b.val[0]);
        a.val[1] = vsubq_f32(a.val[1], b.val[1]);
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], a.val[1]);
        dim -= 8;
        query_bf16 += 8;
        codes_bf16 += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (dim >= 4) {
        float32x4_t a = vcvt_f32_half(vld1_u16((const uint16_t*)query_bf16));
        float32x4_t b = vcvt_f32_half(vld1_u16((const uint16_t*)codes_bf16));
        a = vsubq_f32(a, b);
        res.val[0] = vmlaq_f32(res.val[0], a, a);
        dim -= 4;
        query_bf16 += 4;
        codes_bf16 += 4;
    }
    uint16x4_t res_x = vdup_n_u16(0);
    uint16x4_t res_y = vdup_n_u16(0);
    switch (dim) {
        case 3:
            res_x = vld1_lane_u16((const uint16_t*)query_bf16, res_x, 2);
            res_y = vld1_lane_u16((const uint16_t*)codes_bf16, res_y, 2);
            query_bf16++;
            codes_bf16++;
            dim--;
        case 2:
            res_x = vld1_lane_u16((const uint16_t*)query_bf16, res_x, 1);
            res_y = vld1_lane_u16((const uint16_t*)codes_bf16, res_y, 1);
            query_bf16++;
            codes_bf16++;
            dim--;
        case 1:
            res_x = vld1_lane_u16((const uint16_t*)query_bf16, res_x, 0);
            res_y = vld1_lane_u16((const uint16_t*)codes_bf16, res_y, 0);
            query_bf16++;
            codes_bf16++;
            dim--;
    }

    float32x4_t diff = vsubq_f32(vcvt_f32_half(res_x), vcvt_f32_half(res_y));
    res.val[0] = vmlaq_f32(res.val[0], diff, diff);

    return vaddvq_f32(res.val[0]);
#else
    return generic::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    const auto* query_fp16 = (const uint16_t*)(query);
    const auto* codes_fp16 = (const uint16_t*)(codes);

    float32x4x3_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    while (dim >= 12) {
        float32x4x3_t a = vcvt3_f32_f16(vld3_f16((const __fp16*)query_fp16));
        float32x4x3_t b = vcvt3_f32_f16(vld3_f16((const __fp16*)codes_fp16));

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], b.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], b.val[2]);
        dim -= 12;
        query_fp16 += 12;
        codes_fp16 += 12;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    if (dim >= 8) {
        float32x4x2_t a = vcvt2_f32_f16(vld2_f16((const __fp16*)query_fp16));
        float32x4x2_t b = vcvt2_f32_f16(vld2_f16((const __fp16*)codes_fp16));
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], b.val[1]);
        dim -= 8;
        query_fp16 += 8;
        codes_fp16 += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (dim >= 4) {
        float32x4_t a = vcvt_f32_f16(vld1_f16((const __fp16*)query_fp16));
        float32x4_t b = vcvt_f32_f16(vld1_f16((const __fp16*)codes_fp16));
        res.val[0] = vmlaq_f32(res.val[0], a, b);
        dim -= 4;
        query_fp16 += 4;
        codes_fp16 += 4;
    }

    float16x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
    float16x4_t res_y = {0.0f, 0.0f, 0.0f, 0.0f};
    switch (dim) {
        case 3:
            res_x = vld1_lane_f16((const __fp16*)query_fp16, res_x, 2);
            res_y = vld1_lane_f16((const __fp16*)codes_fp16, res_y, 2);
            query_fp16++;
            codes_fp16++;
            dim--;
        case 2:
            res_x = vld1_lane_f16((const __fp16*)query_fp16, res_x, 1);
            res_y = vld1_lane_f16((const __fp16*)codes_fp16, res_y, 1);
            query_fp16++;
            codes_fp16++;
            dim--;
        case 1:
            res_x = vld1_lane_f16((const __fp16*)query_fp16, res_x, 0);
            res_y = vld1_lane_f16((const __fp16*)codes_fp16, res_y, 0);
            query_fp16++;
            codes_fp16++;
            dim--;
    }
    res.val[0] = vmlaq_f32(res.val[0], vcvt_f32_f16(res_x), vcvt_f32_f16(res_y));

    return vaddvq_f32(res.val[0]);
#else
    return generic::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    float32x4x3_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};

    const auto* query_fp16 = (const uint16_t*)(query);
    const auto* codes_fp16 = (const uint16_t*)(codes);

    while (dim >= 12) {
        float32x4x3_t a = vcvt3_f32_f16(vld3_f16((const __fp16*)query_fp16));
        float32x4x3_t b = vcvt3_f32_f16(vld3_f16((const __fp16*)codes_fp16));
        a.val[0] = vsubq_f32(a.val[0], b.val[0]);
        a.val[1] = vsubq_f32(a.val[1], b.val[1]);
        a.val[2] = vsubq_f32(a.val[2], b.val[2]);

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], a.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], a.val[2]);
        dim -= 12;
        query_fp16 += 12;
        codes_fp16 += 12;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    if (dim >= 8) {
        float32x4x2_t a = vcvt2_f32_f16(vld2_f16((const __fp16*)query_fp16));
        float32x4x2_t b = vcvt2_f32_f16(vld2_f16((const __fp16*)codes_fp16));
        a.val[0] = vsubq_f32(a.val[0], b.val[0]);
        a.val[1] = vsubq_f32(a.val[1], b.val[1]);
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], a.val[1]);
        dim -= 8;
        query_fp16 += 8;
        codes_fp16 += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (dim >= 4) {
        float32x4_t a = vcvt_f32_f16(vld1_f16((const __fp16*)query_fp16));
        float32x4_t b = vcvt_f32_f16(vld1_f16((const __fp16*)codes_fp16));
        a = vsubq_f32(a, b);
        res.val[0] = vmlaq_f32(res.val[0], a, a);
        dim -= 4;
        query_fp16 += 4;
        codes_fp16 += 4;
    }
    if (dim >= 0) {
        float16x4_t res_x = vdup_n_f16(0.0f);
        float16x4_t res_y = vdup_n_f16(0.0f);
        switch (dim) {
            case 3:
                res_x = vld1_lane_f16((const __fp16*)query_fp16, res_x, 2);
                res_y = vld1_lane_f16((const __fp16*)codes_fp16, res_y, 2);
                query_fp16++;
                codes_fp16++;
                dim--;
            case 2:
                res_x = vld1_lane_f16((const __fp16*)query_fp16, res_x, 1);
                res_y = vld1_lane_f16((const __fp16*)codes_fp16, res_y, 1);
                query_fp16++;
                codes_fp16++;
                dim--;
            case 1:
                res_x = vld1_lane_f16((const __fp16*)query_fp16, res_x, 0);
                res_y = vld1_lane_f16((const __fp16*)codes_fp16, res_y, 0);
                query_fp16++;
                codes_fp16++;
                dim--;
        }
        float32x4_t diff = vsubq_f32(vcvt_f32_f16(res_x), vcvt_f32_f16(res_y));

        res.val[0] = vmlaq_f32(res.val[0], diff, diff);
    }
    return vaddvq_f32(res.val[0]);
#else
    return generic::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

#if defined(ENABLE_NEON)
__inline float32x4_t __attribute__((__always_inline__)) load_4_uint8_to_float(const uint8_t* data) {
    uint32x4_t code_values = {data[0], data[1], data[2], data[3]};
    return vcvtq_f32_u32(code_values);
}

__inline void __attribute__((__always_inline__))
load_8_uint8_to_float(const uint8_t* data, float32x4_t& low, float32x4_t& high) {
    uint8x8_t code_vec = vld1_u8(data);
    uint16x8_t code_16 = vmovl_u8(code_vec);
    uint32x4_t code_32_low = vmovl_u16(vget_low_u16(code_16));
    uint32x4_t code_32_high = vmovl_u16(vget_high_u16(code_16));
    low = vcvtq_f32_u32(code_32_low);
    high = vcvtq_f32_u32(code_32_high);
}

__inline void __attribute__((__always_inline__)) load_16_uint8_to_float(
    const uint8_t* data, float32x4_t& f0, float32x4_t& f1, float32x4_t& f2, float32x4_t& f3) {
    uint8x16_t code_vec = vld1q_u8(data);
    uint16x8_t code_16_low = vmovl_u8(vget_low_u8(code_vec));
    uint16x8_t code_16_high = vmovl_u8(vget_high_u8(code_vec));

    uint32x4_t code_32_0 = vmovl_u16(vget_low_u16(code_16_low));
    uint32x4_t code_32_1 = vmovl_u16(vget_high_u16(code_16_low));
    uint32x4_t code_32_2 = vmovl_u16(vget_low_u16(code_16_high));
    uint32x4_t code_32_3 = vmovl_u16(vget_high_u16(code_16_high));

    f0 = vcvtq_f32_u32(code_32_0);
    f1 = vcvtq_f32_u32(code_32_1);
    f2 = vcvtq_f32_u32(code_32_2);
    f3 = vcvtq_f32_u32(code_32_3);
}

__inline void __attribute__((__always_inline__))
load_12_uint8_to_float(const uint8_t* data, float32x4_t& f0, float32x4_t& f1, float32x4_t& f2) {
    // Load 12 bytes
    uint8x8_t code_low = vld1_u8(data);                             // Load first 8 bytes
    uint32x4_t code_last = {data[8], data[9], data[10], data[11]};  // Load last 4 bytes

    uint16x8_t code_16 = vmovl_u8(code_low);
    uint32x4_t code_32_0 = vmovl_u16(vget_low_u16(code_16));
    uint32x4_t code_32_1 = vmovl_u16(vget_high_u16(code_16));

    f0 = vcvtq_f32_u32(code_32_0);
    f1 = vcvtq_f32_u32(code_32_1);
    f2 = vcvtq_f32_u32(code_last);
}
#endif

float
INT8ComputeL2Sqr(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    constexpr int BATCH_SIZE{8};

    const uint64_t n = dim / BATCH_SIZE;

    if (n == 0) {
        return generic::INT8ComputeL2Sqr(query, codes, dim);
    }

    int32x4_t sum_sq = vdupq_n_s32(0);
    for (uint64_t i{0}; i < n; i++) {
        int8x8_t q = vld1_s8(query + BATCH_SIZE * i);
        int8x8_t c = vld1_s8(codes + BATCH_SIZE * i);

        int16x8_t q_16 = vmovl_s8(q);
        int16x8_t c_16 = vmovl_s8(c);

        int16x8_t diff = vsubq_s16(q_16, c_16);

        sum_sq = vmlal_s16(sum_sq, vget_low_s16(diff), vget_low_s16(diff));
        sum_sq = vmlal_s16(sum_sq, vget_high_s16(diff), vget_high_s16(diff));
    }

    int32_t result[4];
    vst1q_s32(result, sum_sq);
    int64_t l2 = static_cast<int64_t>(result[0] + result[1] + result[2] + result[3]);

    l2 += generic::INT8ComputeL2Sqr(
        query + BATCH_SIZE * n, codes + BATCH_SIZE * n, dim - BATCH_SIZE * n);

    return static_cast<float>(l2);
#else
    return generic::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_NEON)
    constexpr int BATCH_SIZE = 8;

    const uint64_t n = dim / BATCH_SIZE;

    if (n == 0) {
        return generic::INT8ComputeIP(query, codes, dim);
    }

    int32x4_t sum_0 = vdupq_n_s32(0);
    int32x4_t sum_1 = vdupq_n_s32(0);

    for (uint64_t i = 0; i < n; ++i) {
        int8x8_t q_vec = vld1_s8(query + BATCH_SIZE * i);
        int8x8_t c_vec = vld1_s8(codes + BATCH_SIZE * i);

        int16x8_t q_16 = vmovl_s8(q_vec);
        int16x8_t c_16 = vmovl_s8(c_vec);

        int16x8_t prod_16 = vmulq_s16(q_16, c_16);

        int32x4_t prod_low = vmovl_s16(vget_low_s16(prod_16));
        int32x4_t prod_high = vmovl_s16(vget_high_s16(prod_16));

        sum_0 = vaddq_s32(sum_0, prod_low);
        sum_1 = vaddq_s32(sum_1, prod_high);
    }

    int32x4_t sum_total = vaddq_s32(sum_0, sum_1);

    int32_t result[4];
    vst1q_s32(result, sum_total);

    int64_t dot = static_cast<int64_t>(result[0]) + static_cast<int64_t>(result[1]) +
                  static_cast<int64_t>(result[2]) + static_cast<int64_t>(result[3]);

    dot += generic::INT8ComputeIP(
        query + BATCH_SIZE * n, codes + BATCH_SIZE * n, dim - BATCH_SIZE * n);

    return static_cast<float>(dot);
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
#if defined(ENABLE_NEON)
    const float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t i = 0;

    for (; i + 11 < dim; i += 12) {
        __builtin_prefetch(codes + i + 48, 0, 1);
        __builtin_prefetch(query + i + 24, 0, 1);

        // Load 12 codes and convert to float in one operation
        float32x4_t code_floats_0, code_floats_1, code_floats_2;
        load_12_uint8_to_float(codes + i, code_floats_0, code_floats_1, code_floats_2);

        float32x4x3_t query_vec = vld1q_f32_x3(query + i);
        float32x4x3_t diff_vec = vld1q_f32_x3(diff + i);
        float32x4x3_t lower_bound_vec = vld1q_f32_x3(lower_bound + i);

        // Normalize and adjust codes in one fused operation: lower_bound + (codes/255) * diff
        float32x4x3_t adjusted_vec;
        adjusted_vec.val[0] =
            vfmaq_f32(lower_bound_vec.val[0], vmulq_f32(code_floats_0, inv255), diff_vec.val[0]);
        adjusted_vec.val[1] =
            vfmaq_f32(lower_bound_vec.val[1], vmulq_f32(code_floats_1, inv255), diff_vec.val[1]);
        adjusted_vec.val[2] =
            vfmaq_f32(lower_bound_vec.val[2], vmulq_f32(code_floats_2, inv255), diff_vec.val[2]);

        // Compute inner product and accumulate
        sum = vfmaq_f32(sum, query_vec.val[0], adjusted_vec.val[0]);
        sum = vfmaq_f32(sum, query_vec.val[1], adjusted_vec.val[1]);
        sum = vfmaq_f32(sum, query_vec.val[2], adjusted_vec.val[2]);
    }

    uint64_t d = dim - i;

    if (d >= 8) {
        float32x4_t code_floats_low, code_floats_high;
        load_8_uint8_to_float(codes + i, code_floats_low, code_floats_high);

        float32x4x2_t query_vec = vld1q_f32_x2(query + i);
        float32x4x2_t diff_vec = vld1q_f32_x2(diff + i);
        float32x4x2_t lower_bound_vec = vld1q_f32_x2(lower_bound + i);

        float32x4x2_t adjusted_vec;
        adjusted_vec.val[0] =
            vfmaq_f32(lower_bound_vec.val[0], vmulq_f32(code_floats_low, inv255), diff_vec.val[0]);
        adjusted_vec.val[1] =
            vfmaq_f32(lower_bound_vec.val[1], vmulq_f32(code_floats_high, inv255), diff_vec.val[1]);

        sum = vfmaq_f32(sum, query_vec.val[0], adjusted_vec.val[0]);
        sum = vfmaq_f32(sum, query_vec.val[1], adjusted_vec.val[1]);
        i += 8;
        d -= 8;
    }

    if (d >= 4) {
        float32x4_t code_floats = load_4_uint8_to_float(codes + i);
        float32x4_t query_vec = vld1q_f32(query + i);
        float32x4_t diff_vec = vld1q_f32(diff + i);
        float32x4_t lower_bound_vec = vld1q_f32(lower_bound + i);

        float32x4_t adjusted = vfmaq_f32(lower_bound_vec, vmulq_f32(code_floats, inv255), diff_vec);
        sum = vfmaq_f32(sum, query_vec, adjusted);
        i += 4;
        d -= 4;
    }

    float32x4_t res_query = vdupq_n_f32(0.0f);
    float32x4_t res_codes = vdupq_n_f32(0.0f);
    float32x4_t res_diff = vdupq_n_f32(0.0f);
    float32x4_t res_lower_bound = vdupq_n_f32(0.0f);

    if (d >= 3) {
        res_query = vld1q_lane_f32(query + i, res_query, 2);
        res_codes = vsetq_lane_f32(codes[i], res_codes, 2);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 2);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 2);
        i++;
        d--;
    }

    if (d >= 2) {
        res_query = vld1q_lane_f32(query + i, res_query, 1);
        res_codes = vsetq_lane_f32(codes[i], res_codes, 1);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 1);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 1);
        i++;
        d--;
    }

    if (d >= 1) {
        res_query = vld1q_lane_f32(query + i, res_query, 0);
        res_codes = vsetq_lane_f32(codes[i], res_codes, 0);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 0);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 0);
    }

    if (dim > i) {
        float32x4_t adjusted = vfmaq_f32(res_lower_bound, vmulq_f32(res_codes, inv255), res_diff);
        sum = vfmaq_f32(sum, res_query, adjusted);
    }

    return vaddvq_f32(sum);
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
#if defined(ENABLE_NEON)
    const float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t i = 0;

    for (; i + 11 < dim; i += 12) {
        __builtin_prefetch(codes + i + 48, 0, 1);
        __builtin_prefetch(query + i + 24, 0, 1);

        // Load 12 codes and convert to float in one operation
        float32x4_t code_floats_0, code_floats_1, code_floats_2;
        load_12_uint8_to_float(codes + i, code_floats_0, code_floats_1, code_floats_2);

        float32x4x3_t query_vec = vld1q_f32_x3(query + i);
        float32x4x3_t diff_vec = vld1q_f32_x3(diff + i);
        float32x4x3_t lower_bound_vec = vld1q_f32_x3(lower_bound + i);

        // Compute adjusted codes and distance in one fused operation
        float32x4x3_t dist_vec;
        dist_vec.val[0] = vsubq_f32(
            query_vec.val[0],
            vfmaq_f32(lower_bound_vec.val[0], vmulq_f32(code_floats_0, inv255), diff_vec.val[0]));
        dist_vec.val[1] = vsubq_f32(
            query_vec.val[1],
            vfmaq_f32(lower_bound_vec.val[1], vmulq_f32(code_floats_1, inv255), diff_vec.val[1]));
        dist_vec.val[2] = vsubq_f32(
            query_vec.val[2],
            vfmaq_f32(lower_bound_vec.val[2], vmulq_f32(code_floats_2, inv255), diff_vec.val[2]));

        // Compute squared distance and accumulate
        sum = vfmaq_f32(sum, dist_vec.val[0], dist_vec.val[0]);
        sum = vfmaq_f32(sum, dist_vec.val[1], dist_vec.val[1]);
        sum = vfmaq_f32(sum, dist_vec.val[2], dist_vec.val[2]);
    }

    uint64_t d = dim - i;

    if (d >= 8) {
        float32x4_t code_floats_low, code_floats_high;
        load_8_uint8_to_float(codes + i, code_floats_low, code_floats_high);

        float32x4x2_t query_vec = vld1q_f32_x2(query + i);
        float32x4x2_t diff_vec = vld1q_f32_x2(diff + i);
        float32x4x2_t lower_bound_vec = vld1q_f32_x2(lower_bound + i);

        float32x4x2_t adjusted_vec;
        adjusted_vec.val[0] =
            vfmaq_f32(lower_bound_vec.val[0], vmulq_f32(code_floats_low, inv255), diff_vec.val[0]);
        adjusted_vec.val[1] =
            vfmaq_f32(lower_bound_vec.val[1], vmulq_f32(code_floats_high, inv255), diff_vec.val[1]);

        float32x4x2_t dist_vec;
        dist_vec.val[0] = vsubq_f32(query_vec.val[0], adjusted_vec.val[0]);
        dist_vec.val[1] = vsubq_f32(query_vec.val[1], adjusted_vec.val[1]);

        sum = vfmaq_f32(sum, dist_vec.val[0], dist_vec.val[0]);
        sum = vfmaq_f32(sum, dist_vec.val[1], dist_vec.val[1]);
        i += 8;
        d -= 8;
    }

    if (d >= 4) {
        float32x4_t code_floats = load_4_uint8_to_float(codes + i);
        float32x4_t query_vec = vld1q_f32(query + i);
        float32x4_t diff_vec = vld1q_f32(diff + i);
        float32x4_t lower_bound_vec = vld1q_f32(lower_bound + i);

        float32x4_t adjusted = vfmaq_f32(lower_bound_vec, vmulq_f32(code_floats, inv255), diff_vec);
        float32x4_t dist = vsubq_f32(query_vec, adjusted);
        sum = vfmaq_f32(sum, dist, dist);
        i += 4;
        d -= 4;
    }

    float32x4_t res_query = vdupq_n_f32(0.0f);
    float32x4_t res_codes = vdupq_n_f32(0.0f);
    float32x4_t res_diff = vdupq_n_f32(0.0f);
    float32x4_t res_lower_bound = vdupq_n_f32(0.0f);

    if (d >= 3) {
        res_query = vld1q_lane_f32(query + i, res_query, 2);
        res_codes = vsetq_lane_f32(codes[i], res_codes, 2);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 2);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 2);
        i++;
        d--;
    }

    if (d >= 2) {
        res_query = vld1q_lane_f32(query + i, res_query, 1);
        res_codes = vsetq_lane_f32(codes[i], res_codes, 1);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 1);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 1);
        i++;
        d--;
    }

    if (d >= 1) {
        res_query = vld1q_lane_f32(query + i, res_query, 0);
        res_codes = vsetq_lane_f32(codes[i], res_codes, 0);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 0);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 0);
    }

    if (dim > i) {
        float32x4_t adjusted = vfmaq_f32(res_lower_bound, vmulq_f32(res_codes, inv255), res_diff);
        float32x4_t dist = vsubq_f32(res_query, adjusted);
        sum = vfmaq_f32(sum, dist, dist);
    }

    return vaddvq_f32(sum);
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
#if defined(ENABLE_NEON)
    const float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t i = 0;

    // Process 12 elements at a time for optimal instruction usage
    for (; i + 11 < dim; i += 12) {
        __builtin_prefetch(codes1 + i + 48, 0, 1);
        __builtin_prefetch(codes2 + i + 48, 0, 1);

        // Load 12 codes from both arrays and convert to float
        float32x4_t code1_floats_0, code1_floats_1, code1_floats_2;
        float32x4_t code2_floats_0, code2_floats_1, code2_floats_2;
        load_12_uint8_to_float(codes1 + i, code1_floats_0, code1_floats_1, code1_floats_2);
        load_12_uint8_to_float(codes2 + i, code2_floats_0, code2_floats_1, code2_floats_2);

        // Load 12 elements each using x3 intrinsics
        float32x4x3_t diff_vec = vld1q_f32_x3(diff + i);
        float32x4x3_t lower_bound_vec = vld1q_f32_x3(lower_bound + i);

        // Compute adjusted codes and inner product in one fused operation
        float32x4x3_t adjusted1_vec, adjusted2_vec;
        adjusted1_vec.val[0] =
            vfmaq_f32(lower_bound_vec.val[0], vmulq_f32(code1_floats_0, inv255), diff_vec.val[0]);
        adjusted1_vec.val[1] =
            vfmaq_f32(lower_bound_vec.val[1], vmulq_f32(code1_floats_1, inv255), diff_vec.val[1]);
        adjusted1_vec.val[2] =
            vfmaq_f32(lower_bound_vec.val[2], vmulq_f32(code1_floats_2, inv255), diff_vec.val[2]);

        adjusted2_vec.val[0] =
            vfmaq_f32(lower_bound_vec.val[0], vmulq_f32(code2_floats_0, inv255), diff_vec.val[0]);
        adjusted2_vec.val[1] =
            vfmaq_f32(lower_bound_vec.val[1], vmulq_f32(code2_floats_1, inv255), diff_vec.val[1]);
        adjusted2_vec.val[2] =
            vfmaq_f32(lower_bound_vec.val[2], vmulq_f32(code2_floats_2, inv255), diff_vec.val[2]);

        // Compute inner product and accumulate
        sum = vfmaq_f32(sum, adjusted1_vec.val[0], adjusted2_vec.val[0]);
        sum = vfmaq_f32(sum, adjusted1_vec.val[1], adjusted2_vec.val[1]);
        sum = vfmaq_f32(sum, adjusted1_vec.val[2], adjusted2_vec.val[2]);
    }

    // Process remaining elements with optimized NEON instructions
    uint64_t d = dim - i;

    // Process 8 elements if remaining
    if (d >= 8) {
        float32x4_t code1_floats_low, code1_floats_high;
        float32x4_t code2_floats_low, code2_floats_high;
        load_8_uint8_to_float(codes1 + i, code1_floats_low, code1_floats_high);
        load_8_uint8_to_float(codes2 + i, code2_floats_low, code2_floats_high);

        float32x4x2_t diff_vec = vld1q_f32_x2(diff + i);
        float32x4x2_t lower_bound_vec = vld1q_f32_x2(lower_bound + i);

        float32x4x2_t adjusted1_vec, adjusted2_vec;
        adjusted1_vec.val[0] =
            vfmaq_f32(lower_bound_vec.val[0], vmulq_f32(code1_floats_low, inv255), diff_vec.val[0]);
        adjusted1_vec.val[1] = vfmaq_f32(
            lower_bound_vec.val[1], vmulq_f32(code1_floats_high, inv255), diff_vec.val[1]);
        adjusted2_vec.val[0] =
            vfmaq_f32(lower_bound_vec.val[0], vmulq_f32(code2_floats_low, inv255), diff_vec.val[0]);
        adjusted2_vec.val[1] = vfmaq_f32(
            lower_bound_vec.val[1], vmulq_f32(code2_floats_high, inv255), diff_vec.val[1]);

        sum = vfmaq_f32(sum, adjusted1_vec.val[0], adjusted2_vec.val[0]);
        sum = vfmaq_f32(sum, adjusted1_vec.val[1], adjusted2_vec.val[1]);
        i += 8;
        d -= 8;
    }

    // Process 4 elements if remaining
    if (d >= 4) {
        float32x4_t code1_floats = load_4_uint8_to_float(codes1 + i);
        float32x4_t code2_floats = load_4_uint8_to_float(codes2 + i);
        float32x4_t diff_vec = vld1q_f32(diff + i);
        float32x4_t lower_bound_vec = vld1q_f32(lower_bound + i);

        float32x4_t adjusted1 =
            vfmaq_f32(lower_bound_vec, vmulq_f32(code1_floats, inv255), diff_vec);
        float32x4_t adjusted2 =
            vfmaq_f32(lower_bound_vec, vmulq_f32(code2_floats, inv255), diff_vec);
        sum = vfmaq_f32(sum, adjusted1, adjusted2);
        i += 4;
        d -= 4;
    }

    // Process remaining 1-3 elements
    float32x4_t res_codes1 = vdupq_n_f32(0.0f);
    float32x4_t res_codes2 = vdupq_n_f32(0.0f);
    float32x4_t res_diff = vdupq_n_f32(0.0f);
    float32x4_t res_lower_bound = vdupq_n_f32(0.0f);

    if (d >= 3) {
        res_codes1 = vsetq_lane_f32(codes1[i], res_codes1, 2);
        res_codes2 = vsetq_lane_f32(codes2[i], res_codes2, 2);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 2);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 2);
        i++;
        d--;
    }

    if (d >= 2) {
        res_codes1 = vsetq_lane_f32(codes1[i], res_codes1, 1);
        res_codes2 = vsetq_lane_f32(codes2[i], res_codes2, 1);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 1);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 1);
        i++;
        d--;
    }

    if (d >= 1) {
        res_codes1 = vsetq_lane_f32(codes1[i], res_codes1, 0);
        res_codes2 = vsetq_lane_f32(codes2[i], res_codes2, 0);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 0);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 0);
    }

    if (dim > i) {
        float32x4_t adjusted1 = vfmaq_f32(res_lower_bound, vmulq_f32(res_codes1, inv255), res_diff);
        float32x4_t adjusted2 = vfmaq_f32(res_lower_bound, vmulq_f32(res_codes2, inv255), res_diff);
        sum = vfmaq_f32(sum, adjusted1, adjusted2);
    }

    return vaddvq_f32(sum);
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
#if defined(ENABLE_NEON)
    const float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t i = 0;

    // Process 12 elements at a time for optimal instruction usage
    for (; i + 11 < dim; i += 12) {
        __builtin_prefetch(codes1 + i + 48, 0, 1);
        __builtin_prefetch(codes2 + i + 48, 0, 1);

        // Load 12 codes and compute difference at integer level for efficiency
        uint8x8_t codes1_low = vld1_u8(codes1 + i);
        uint8x8_t codes2_low = vld1_u8(codes2 + i);
        uint32x4_t codes1_last = {codes1[i + 8], codes1[i + 9], codes1[i + 10], codes1[i + 11]};
        uint32x4_t codes2_last = {codes2[i + 8], codes2[i + 9], codes2[i + 10], codes2[i + 11]};

        // Compute code difference at integer level
        int16x8_t diff_low = vreinterpretq_s16_u16(vsubl_u8(codes1_low, codes2_low));
        int32x4_t diff_last =
            vsubq_s32(vreinterpretq_s32_u32(codes1_last), vreinterpretq_s32_u32(codes2_last));

        // Convert to float32x4x3_t
        float32x4x3_t code_diff_vec;
        code_diff_vec.val[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(diff_low)));
        code_diff_vec.val[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(diff_low)));
        code_diff_vec.val[2] = vcvtq_f32_s32(diff_last);

        // Load 12 diff values using x3 intrinsics
        float32x4x3_t diff_vec = vld1q_f32_x3(diff + i);

        // Compute final distance in one fused operation: (code_diff / 255) * diff
        float32x4x3_t scaled_diff_vec;
        scaled_diff_vec.val[0] =
            vmulq_f32(vmulq_f32(code_diff_vec.val[0], inv255), diff_vec.val[0]);
        scaled_diff_vec.val[1] =
            vmulq_f32(vmulq_f32(code_diff_vec.val[1], inv255), diff_vec.val[1]);
        scaled_diff_vec.val[2] =
            vmulq_f32(vmulq_f32(code_diff_vec.val[2], inv255), diff_vec.val[2]);

        // Compute squared distance and accumulate
        sum = vfmaq_f32(sum, scaled_diff_vec.val[0], scaled_diff_vec.val[0]);
        sum = vfmaq_f32(sum, scaled_diff_vec.val[1], scaled_diff_vec.val[1]);
        sum = vfmaq_f32(sum, scaled_diff_vec.val[2], scaled_diff_vec.val[2]);
    }

    // Process remaining elements with optimized NEON instructions
    uint64_t d = dim - i;

    // Process 8 elements if remaining
    if (d >= 8) {
        // Load 8 codes and compute difference at integer level
        uint8x8_t codes1_vec = vld1_u8(codes1 + i);
        uint8x8_t codes2_vec = vld1_u8(codes2 + i);

        // Compute code difference at integer level
        int16x8_t diff_codes = vreinterpretq_s16_u16(vsubl_u8(codes1_vec, codes2_vec));

        // Convert to float32x4x2_t
        float32x4x2_t code_diff_vec;
        code_diff_vec.val[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(diff_codes)));
        code_diff_vec.val[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(diff_codes)));

        // Load 8 diff values
        float32x4x2_t diff_vec = vld1q_f32_x2(diff + i);

        // Compute scaled distance: (code_diff / 255) * diff
        float32x4x2_t scaled_diff_vec;
        scaled_diff_vec.val[0] =
            vmulq_f32(vmulq_f32(code_diff_vec.val[0], inv255), diff_vec.val[0]);
        scaled_diff_vec.val[1] =
            vmulq_f32(vmulq_f32(code_diff_vec.val[1], inv255), diff_vec.val[1]);

        // Compute squared distance and accumulate
        sum = vfmaq_f32(sum, scaled_diff_vec.val[0], scaled_diff_vec.val[0]);
        sum = vfmaq_f32(sum, scaled_diff_vec.val[1], scaled_diff_vec.val[1]);
        i += 8;
        d -= 8;
    }

    // Process 4 elements if remaining
    if (d >= 4) {
        float32x4_t code1_floats = load_4_uint8_to_float(codes1 + i);
        float32x4_t code2_floats = load_4_uint8_to_float(codes2 + i);
        float32x4_t diff_vec = vld1q_f32(diff + i);
        float32x4_t lower_bound_vec = vld1q_f32(lower_bound + i);

        float32x4_t adjusted1 =
            vfmaq_f32(lower_bound_vec, vmulq_f32(code1_floats, inv255), diff_vec);
        float32x4_t adjusted2 =
            vfmaq_f32(lower_bound_vec, vmulq_f32(code2_floats, inv255), diff_vec);
        float32x4_t dist = vsubq_f32(adjusted1, adjusted2);
        sum = vfmaq_f32(sum, dist, dist);
        i += 4;
        d -= 4;
    }

    // Process remaining 1-3 elements
    float32x4_t res_codes1 = vdupq_n_f32(0.0f);
    float32x4_t res_codes2 = vdupq_n_f32(0.0f);
    float32x4_t res_diff = vdupq_n_f32(0.0f);
    float32x4_t res_lower_bound = vdupq_n_f32(0.0f);

    if (d >= 3) {
        res_codes1 = vsetq_lane_f32(codes1[i], res_codes1, 2);
        res_codes2 = vsetq_lane_f32(codes2[i], res_codes2, 2);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 2);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 2);
        i++;
        d--;
    }

    if (d >= 2) {
        res_codes1 = vsetq_lane_f32(codes1[i], res_codes1, 1);
        res_codes2 = vsetq_lane_f32(codes2[i], res_codes2, 1);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 1);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 1);
        i++;
        d--;
    }

    if (d >= 1) {
        res_codes1 = vsetq_lane_f32(codes1[i], res_codes1, 0);
        res_codes2 = vsetq_lane_f32(codes2[i], res_codes2, 0);
        res_diff = vld1q_lane_f32(diff + i, res_diff, 0);
        res_lower_bound = vld1q_lane_f32(lower_bound + i, res_lower_bound, 0);
    }

    if (dim > i) {
        float32x4_t adjusted1 = vfmaq_f32(res_lower_bound, vmulq_f32(res_codes1, inv255), res_diff);
        float32x4_t adjusted2 = vfmaq_f32(res_lower_bound, vmulq_f32(res_codes2, inv255), res_diff);
        float32x4_t dist = vsubq_f32(adjusted1, adjusted2);
        sum = vfmaq_f32(sum, dist, dist);
    }

    return vaddvq_f32(sum);
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
#if defined(ENABLE_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t d = 0;

    const float inv15 = 1.0f / 15.0f;
    float32x4_t v_inv15 = vdupq_n_f32(inv15);

    for (; d + 7 < dim; d += 8) {
        uint8_t byte0 = codes[d >> 1];
        uint8_t byte1 = codes[(d >> 1) + 1];
        uint8_t byte2 = codes[(d >> 1) + 2];
        uint8_t byte3 = codes[(d >> 1) + 3];

        float32x4_t code_low = {
            static_cast<float>(byte0 & 0x0f),
            static_cast<float>(byte0 >> 4),
            static_cast<float>(byte1 & 0x0f),
            static_cast<float>(byte1 >> 4),
        };

        float32x4_t code_high = {
            static_cast<float>(byte2 & 0x0f),
            static_cast<float>(byte2 >> 4),
            static_cast<float>(byte3 & 0x0f),
            static_cast<float>(byte3 >> 4),
        };

        code_low = vmulq_f32(code_low, v_inv15);
        code_high = vmulq_f32(code_high, v_inv15);

        float32x4_t query_low = vld1q_f32(query + d);
        float32x4_t query_high = vld1q_f32(query + d + 4);
        float32x4_t diff_low = vld1q_f32(diff + d);
        float32x4_t diff_high = vld1q_f32(diff + d + 4);
        float32x4_t lb_low = vld1q_f32(lower_bound + d);
        float32x4_t lb_high = vld1q_f32(lower_bound + d + 4);

        code_low = vmlaq_f32(lb_low, code_low, diff_low);
        code_high = vmlaq_f32(lb_high, code_high, diff_high);

        sum = vmlaq_f32(sum, query_low, code_low);
        sum = vmlaq_f32(sum, query_high, code_high);
    }

    float result = vaddvq_f32(sum);

    if (d < dim) {
        result +=
            generic::SQ4ComputeIP(query + d, codes + (d >> 1), lower_bound + d, diff + d, dim - d);
    }

    return result;
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
#if defined(ENABLE_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t d = 0;

    const float inv15 = 1.0f / 15.0f;
    float32x4_t v_inv15 = vdupq_n_f32(inv15);

    for (; d + 7 < dim; d += 8) {
        uint8_t byte0 = codes[d >> 1];
        uint8_t byte1 = codes[(d >> 1) + 1];
        uint8_t byte2 = codes[(d >> 1) + 2];
        uint8_t byte3 = codes[(d >> 1) + 3];

        float32x4_t code_low = {
            static_cast<float>(byte0 & 0x0f),
            static_cast<float>(byte0 >> 4),
            static_cast<float>(byte1 & 0x0f),
            static_cast<float>(byte1 >> 4),
        };

        float32x4_t code_high = {
            static_cast<float>(byte2 & 0x0f),
            static_cast<float>(byte2 >> 4),
            static_cast<float>(byte3 & 0x0f),
            static_cast<float>(byte3 >> 4),
        };

        code_low = vmulq_f32(code_low, v_inv15);
        code_high = vmulq_f32(code_high, v_inv15);

        float32x4_t query_low = vld1q_f32(query + d);
        float32x4_t query_high = vld1q_f32(query + d + 4);
        float32x4_t diff_low = vld1q_f32(diff + d);
        float32x4_t diff_high = vld1q_f32(diff + d + 4);
        float32x4_t lb_low = vld1q_f32(lower_bound + d);
        float32x4_t lb_high = vld1q_f32(lower_bound + d + 4);

        code_low = vmlaq_f32(lb_low, code_low, diff_low);
        code_high = vmlaq_f32(lb_high, code_high, diff_high);

        float32x4_t diff_vec_low = vsubq_f32(query_low, code_low);
        float32x4_t diff_vec_high = vsubq_f32(query_high, code_high);

        sum = vmlaq_f32(sum, diff_vec_low, diff_vec_low);
        sum = vmlaq_f32(sum, diff_vec_high, diff_vec_high);
    }

    float result = vaddvq_f32(sum);

    if (d < dim) {
        result += generic::SQ4ComputeL2Sqr(
            query + d, codes + (d >> 1), lower_bound + d, diff + d, dim - d);
    }

    return result;
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
#if defined(ENABLE_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t d = 0;
    const float inv15 = 1.0f / 15.0f;
    float32x4_t v_inv15 = vdupq_n_f32(inv15);

    for (; d + 7 < dim; d += 8) {
        uint8_t byte1_0 = codes1[d >> 1];
        uint8_t byte1_1 = codes1[(d >> 1) + 1];
        uint8_t byte1_2 = codes1[(d >> 1) + 2];
        uint8_t byte1_3 = codes1[(d >> 1) + 3];

        uint8_t byte2_0 = codes2[d >> 1];
        uint8_t byte2_1 = codes2[(d >> 1) + 1];
        uint8_t byte2_2 = codes2[(d >> 1) + 2];
        uint8_t byte2_3 = codes2[(d >> 1) + 3];

        float32x4_t code1_low = {static_cast<float>(byte1_0 & 0x0f),
                                 static_cast<float>(byte1_0 >> 4),
                                 static_cast<float>(byte1_1 & 0x0f),
                                 static_cast<float>(byte1_1 >> 4)};

        float32x4_t code1_high = {static_cast<float>(byte1_2 & 0x0f),
                                  static_cast<float>(byte1_2 >> 4),
                                  static_cast<float>(byte1_3 & 0x0f),
                                  static_cast<float>(byte1_3 >> 4)};

        float32x4_t code2_low = {static_cast<float>(byte2_0 & 0x0f),
                                 static_cast<float>(byte2_0 >> 4),
                                 static_cast<float>(byte2_1 & 0x0f),
                                 static_cast<float>(byte2_1 >> 4)};

        float32x4_t code2_high = {static_cast<float>(byte2_2 & 0x0f),
                                  static_cast<float>(byte2_2 >> 4),
                                  static_cast<float>(byte2_3 & 0x0f),
                                  static_cast<float>(byte2_3 >> 4)};

        code1_low = vmulq_f32(code1_low, v_inv15);
        code1_high = vmulq_f32(code1_high, v_inv15);
        code2_low = vmulq_f32(code2_low, v_inv15);
        code2_high = vmulq_f32(code2_high, v_inv15);

        float32x4_t diff_low = vld1q_f32(diff + d);
        float32x4_t diff_high = vld1q_f32(diff + d + 4);
        float32x4_t lb_low = vld1q_f32(lower_bound + d);
        float32x4_t lb_high = vld1q_f32(lower_bound + d + 4);

        code1_low = vmlaq_f32(lb_low, code1_low, diff_low);
        code1_high = vmlaq_f32(lb_high, code1_high, diff_high);
        code2_low = vmlaq_f32(lb_low, code2_low, diff_low);
        code2_high = vmlaq_f32(lb_high, code2_high, diff_high);

        sum = vmlaq_f32(sum, code1_low, code2_low);
        sum = vmlaq_f32(sum, code1_high, code2_high);
    }

    float result = vaddvq_f32(sum);

    if (d < dim) {
        result += generic::SQ4ComputeCodesIP(
            codes1 + (d >> 1), codes2 + (d >> 1), lower_bound + d, diff + d, dim - d);
    }

    return result;
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
#if defined(ENABLE_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t d = 0;

    const float inv15 = 1.0f / 15.0f;
    float32x4_t v_inv15 = vdupq_n_f32(inv15);

    for (; d + 7 < dim; d += 8) {
        uint8_t byte1_0 = codes1[d >> 1];
        uint8_t byte1_1 = codes1[(d >> 1) + 1];
        uint8_t byte1_2 = codes1[(d >> 1) + 2];
        uint8_t byte1_3 = codes1[(d >> 1) + 3];

        uint8_t byte2_0 = codes2[d >> 1];
        uint8_t byte2_1 = codes2[(d >> 1) + 1];
        uint8_t byte2_2 = codes2[(d >> 1) + 2];
        uint8_t byte2_3 = codes2[(d >> 1) + 3];

        float32x4_t code1_low = {static_cast<float>(byte1_0 & 0x0f),
                                 static_cast<float>(byte1_0 >> 4),
                                 static_cast<float>(byte1_1 & 0x0f),
                                 static_cast<float>(byte1_1 >> 4)};

        float32x4_t code1_high = {static_cast<float>(byte1_2 & 0x0f),
                                  static_cast<float>(byte1_2 >> 4),
                                  static_cast<float>(byte1_3 & 0x0f),
                                  static_cast<float>(byte1_3 >> 4)};

        float32x4_t code2_low = {static_cast<float>(byte2_0 & 0x0f),
                                 static_cast<float>(byte2_0 >> 4),
                                 static_cast<float>(byte2_1 & 0x0f),
                                 static_cast<float>(byte2_1 >> 4)};

        float32x4_t code2_high = {static_cast<float>(byte2_2 & 0x0f),
                                  static_cast<float>(byte2_2 >> 4),
                                  static_cast<float>(byte2_3 & 0x0f),
                                  static_cast<float>(byte2_3 >> 4)};

        code1_low = vmulq_f32(code1_low, v_inv15);
        code1_high = vmulq_f32(code1_high, v_inv15);
        code2_low = vmulq_f32(code2_low, v_inv15);
        code2_high = vmulq_f32(code2_high, v_inv15);

        float32x4_t diff_low = vld1q_f32(diff + d);
        float32x4_t diff_high = vld1q_f32(diff + d + 4);
        float32x4_t lb_low = vld1q_f32(lower_bound + d);
        float32x4_t lb_high = vld1q_f32(lower_bound + d + 4);

        code1_low = vmlaq_f32(lb_low, code1_low, diff_low);
        code1_high = vmlaq_f32(lb_high, code1_high, diff_high);
        code2_low = vmlaq_f32(lb_low, code2_low, diff_low);
        code2_high = vmlaq_f32(lb_high, code2_high, diff_high);

        float32x4_t diff_vec_low = vsubq_f32(code1_low, code2_low);
        float32x4_t diff_vec_high = vsubq_f32(code1_high, code2_high);

        sum = vmlaq_f32(sum, diff_vec_low, diff_vec_low);
        sum = vmlaq_f32(sum, diff_vec_high, diff_vec_high);
    }

    float result = vaddvq_f32(sum);

    if (d < dim) {
        result += generic::SQ4ComputeCodesL2Sqr(
            codes1 + (d >> 1), codes2 + (d >> 1), lower_bound + d, diff + d, dim - d);
    }

    return result;
#else
    return generic::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

#if defined(ENABLE_NEON)
__inline void __attribute__((__always_inline__))
compute_part(const uint8x16_t& a_vec, const uint8x16_t& b_vec, uint32x4_t& sum) {
    uint8x8_t a_lo = vget_low_u8(a_vec);
    uint8x8_t a_hi = vget_high_u8(a_vec);
    uint8x8_t b_lo = vget_low_u8(b_vec);
    uint8x8_t b_hi = vget_high_u8(b_vec);

    uint16x8_t prod_lo = vmull_u8(a_lo, b_lo);
    uint16x8_t prod_hi = vmull_u8(a_hi, b_hi);

    uint32x4_t sum_lo = vaddl_u16(vget_low_u16(prod_lo), vget_high_u16(prod_lo));
    uint32x4_t sum_hi = vaddl_u16(vget_low_u16(prod_hi), vget_high_u16(prod_hi));

    sum = vaddq_u32(sum, sum_lo);
    sum = vaddq_u32(sum, sum_hi);
}
#endif

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_NEON)
    if (dim == 0) {
        return 0.0f;
    }

    uint32x4_t sum = vdupq_n_u32(0);
    uint64_t d = 0;

    for (; d + 31 < dim; d += 32) {
        uint8x16_t a = vld1q_u8(codes1 + (d >> 1));
        uint8x16_t b = vld1q_u8(codes2 + (d >> 1));
        uint8x16_t mask = vdupq_n_u8(0x0f);

        uint8x16_t a_low = vandq_u8(a, mask);
        uint8x16_t a_high = vandq_u8(vshrq_n_u8(a, 4), mask);
        uint8x16_t b_low = vandq_u8(b, mask);
        uint8x16_t b_high = vandq_u8(vshrq_n_u8(b, 4), mask);

        compute_part(a_low, b_low, sum);
        compute_part(a_high, b_high, sum);
    }
    int scalar_sum =
        generic::SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);

    return static_cast<float>(vaddvq_u32(sum) + scalar_sum);
#else
    return generic::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t d) {
#if defined(ENABLE_NEON)
    uint32x4_t sum_ = vdupq_n_u32(0);
    while (d >= 16) {
        uint8x16_t a = vld1q_u8(codes1);
        uint8x16_t b = vld1q_u8(codes2);

        uint16x8_t a_low = vmovl_u8(vget_low_u8(a));
        uint16x8_t a_high = vmovl_u8(vget_high_u8(a));
        uint16x8_t b_low = vmovl_u8(vget_low_u8(b));
        uint16x8_t b_high = vmovl_u8(vget_high_u8(b));

        uint32x4_t a_low_low = vmovl_u16(vget_low_u16(a_low));
        uint32x4_t a_low_high = vmovl_u16(vget_high_u16(a_low));
        uint32x4_t a_high_low = vmovl_u16(vget_low_u16(a_high));
        uint32x4_t a_high_high = vmovl_u16(vget_high_u16(a_high));

        uint32x4_t b_low_low = vmovl_u16(vget_low_u16(b_low));
        uint32x4_t b_low_high = vmovl_u16(vget_high_u16(b_low));
        uint32x4_t b_high_low = vmovl_u16(vget_low_u16(b_high));
        uint32x4_t b_high_high = vmovl_u16(vget_high_u16(b_high));

        sum_ = vaddq_u32(sum_, vmulq_u32(a_low_low, b_low_low));
        sum_ = vaddq_u32(sum_, vmulq_u32(a_low_high, b_low_high));
        sum_ = vaddq_u32(sum_, vmulq_u32(a_high_low, b_high_low));
        sum_ = vaddq_u32(sum_, vmulq_u32(a_high_high, b_high_high));

        codes1 += 16;
        codes2 += 16;
        d -= 16;
    }

    if (d >= 8) {
        uint8x8_t a = vld1_u8(codes1);
        uint8x8_t b = vld1_u8(codes2);

        uint16x8_t a_ext = vmovl_u8(a);
        uint16x8_t b_ext = vmovl_u8(b);

        uint32x4_t a_low = vmovl_u16(vget_low_u16(a_ext));
        uint32x4_t a_high = vmovl_u16(vget_high_u16(a_ext));
        uint32x4_t b_low = vmovl_u16(vget_low_u16(b_ext));
        uint32x4_t b_high = vmovl_u16(vget_high_u16(b_ext));

        sum_ = vaddq_u32(sum_, vmulq_u32(a_low, b_low));
        sum_ = vaddq_u32(sum_, vmulq_u32(a_high, b_high));

        codes1 += 8;
        codes2 += 8;
        d -= 8;
    }

    if (d >= 4) {
        uint8x8_t a = vld1_u8(codes1);
        uint8x8_t b = vld1_u8(codes2);

        uint16x8_t a_ext = vmovl_u8(a);
        uint16x8_t b_ext = vmovl_u8(b);

        uint32x4_t a_low = vmovl_u16(vget_low_u16(a_ext));
        uint32x4_t b_low = vmovl_u16(vget_low_u16(b_ext));

        sum_ = vaddq_u32(sum_, vmulq_u32(a_low, b_low));

        codes1 += 4;
        codes2 += 4;
        d -= 4;
    }
    int32_t rem_sum = 0;
    for (uint64_t i = 0; i < d; ++i) {
        rem_sum += static_cast<int32_t>(codes1[i]) * static_cast<int32_t>(codes2[i]);
    }

    // accumulate the total sum
    return static_cast<float>(vaddvq_u32(sum_) + rem_sum);
#else
    return generic::SQ8UniformComputeCodesIP(codes1, codes2, d);
#endif
}

#if defined(ENABLE_NEON)
__inline void __attribute__((__always_inline__)) extract_12_bits_to_mask(const uint8_t* bits,
                                                                         size_t bit_offset,
                                                                         uint32x4_t& mask0,
                                                                         uint32x4_t& mask1,
                                                                         uint32x4_t& mask2) {
    size_t byte_idx = bit_offset / 8;
    size_t bit_start = bit_offset % 8;

    uint32_t mask_bits;
    if (bit_start <= 4) {
        // 12 bits span at most 2 bytes
        mask_bits =
            (bits[byte_idx] >> bit_start) | ((uint32_t)bits[byte_idx + 1] << (8 - bit_start));
    } else {
        // 12 bits span 3 bytes
        mask_bits = (bits[byte_idx] >> bit_start) |
                    ((uint32_t)bits[byte_idx + 1] << (8 - bit_start)) |
                    ((uint32_t)bits[byte_idx + 2] << (16 - bit_start));
    }
    mask_bits &= 0xFFF;  // Keep only 12 bits

    // Create mask vectors for 12 elements (4+4+4)
    mask0 = (uint32x4_t){(mask_bits & 0x001) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x002) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x004) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x008) ? 0xFFFFFFFF : 0};
    mask1 = (uint32x4_t){(mask_bits & 0x010) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x020) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x040) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x080) ? 0xFFFFFFFF : 0};
    mask2 = (uint32x4_t){(mask_bits & 0x100) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x200) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x400) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x800) ? 0xFFFFFFFF : 0};
}

__inline void __attribute__((__always_inline__)) extract_8_bits_to_mask(const uint8_t* bits,
                                                                        size_t bit_offset,
                                                                        uint32x4_t& mask0,
                                                                        uint32x4_t& mask1) {
    size_t byte_idx = bit_offset / 8;
    size_t bit_start = bit_offset % 8;

    uint16_t mask_bits;
    if (bit_start == 0) {
        mask_bits = bits[byte_idx];
    } else {
        mask_bits =
            (bits[byte_idx] >> bit_start) | ((uint16_t)bits[byte_idx + 1] << (8 - bit_start));
    }
    mask_bits &= 0xFF;  // Keep only 8 bits

    // Create mask vectors for 8 elements (4+4)
    mask0 = (uint32x4_t){(mask_bits & 0x01) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x02) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x04) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x08) ? 0xFFFFFFFF : 0};
    mask1 = (uint32x4_t){(mask_bits & 0x10) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x20) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x40) ? 0xFFFFFFFF : 0,
                         (mask_bits & 0x80) ? 0xFFFFFFFF : 0};
}

__inline uint32x4_t __attribute__((__always_inline__))
extract_4_bits_to_mask(const uint8_t* bits, uint64_t bit_offset) {
    uint64_t byte_idx = bit_offset / 8;
    uint64_t bit_start = bit_offset % 8;

    uint8_t mask_bits;
    if (bit_start <= 4) {
        mask_bits = (bits[byte_idx] >> bit_start) & 0xF;
    } else {
        mask_bits = ((bits[byte_idx] >> bit_start) | (bits[byte_idx + 1] << (8 - bit_start))) & 0xF;
    }

    return (uint32x4_t){(mask_bits & 0x1) ? 0xFFFFFFFF : 0,
                        (mask_bits & 0x2) ? 0xFFFFFFFF : 0,
                        (mask_bits & 0x4) ? 0xFFFFFFFF : 0,
                        (mask_bits & 0x8) ? 0xFFFFFFFF : 0};
}
#endif
uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_NEON)
    if (dim == 0)
        return 0;

    uint32_t result = 0;
    size_t num_bytes = (dim + 7) / 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        const uint8_t* codes_ptr = codes + bit_pos * num_bytes;
        uint32x4_t popcnt_sum = vdupq_n_u32(0);

        size_t i = 0;
        for (; i + 15 < num_bytes; i += 16) {
            uint8x16_t code_vec = vld1q_u8(codes_ptr + i);
            uint8x16_t bits_vec = vld1q_u8(bits + i);
            uint8x16_t and_vec = vandq_u8(code_vec, bits_vec);
            uint8x16_t cnt_vec = vcntq_u8(and_vec);
            uint16x8_t sum_low = vpaddlq_u8(cnt_vec);
            uint32x4_t sum_32 = vpaddlq_u16(sum_low);
            popcnt_sum = vaddq_u32(popcnt_sum, sum_32);
        }

        uint32_t bit_count = vaddvq_u32(popcnt_sum);

        for (; i < num_bytes; i++) {
            uint8_t bitwise_and = codes_ptr[i] & bits[i];
            bit_count += __builtin_popcount(bitwise_and);
        }

        result += bit_count << bit_pos;
    }

    return result;
#else
    return generic::RaBitQSQ4UBinaryIP(codes, bits, dim);
#endif
}
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_NEON)
    if (dim == 0) {
        return 0.0f;
    }

    if (dim < 4) {
        return generic::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
    }

    uint64_t d = 0;
    float32x4_t sum = vdupq_n_f32(0.0f);
    const float32x4_t inv_sqrt_d_vec = vdupq_n_f32(inv_sqrt_d);
    const float32x4_t neg_inv_sqrt_d_vec = vdupq_n_f32(-inv_sqrt_d);

    for (; d + 11 < dim; d += 12) {
        __builtin_prefetch(vector + d + 24, 0, 1);

        float32x4x3_t vec_values = vld1q_f32_x3(vector + d);

        // Extract 12 bits and create mask vectors
        uint32x4_t bit_mask0, bit_mask1, bit_mask2;
        extract_12_bits_to_mask(bits, d, bit_mask0, bit_mask1, bit_mask2);

        // Create conditional selection vectors for all 12 elements
        float32x4x3_t b_vec;
        b_vec.val[0] = vbslq_f32(bit_mask0, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);
        b_vec.val[1] = vbslq_f32(bit_mask1, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);
        b_vec.val[2] = vbslq_f32(bit_mask2, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);

        // Fused multiply-accumulate for all 12 elements
        sum = vfmaq_f32(sum, b_vec.val[0], vec_values.val[0]);
        sum = vfmaq_f32(sum, b_vec.val[1], vec_values.val[1]);
        sum = vfmaq_f32(sum, b_vec.val[2], vec_values.val[2]);
    }

    uint64_t remaining = dim - d;

    if (remaining >= 8) {
        float32x4x2_t vec_values = vld1q_f32_x2(vector + d);

        uint32x4_t bit_mask0, bit_mask1;
        extract_8_bits_to_mask(bits, d, bit_mask0, bit_mask1);

        float32x4x2_t b_vec;
        b_vec.val[0] = vbslq_f32(bit_mask0, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);
        b_vec.val[1] = vbslq_f32(bit_mask1, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);

        sum = vfmaq_f32(sum, b_vec.val[0], vec_values.val[0]);
        sum = vfmaq_f32(sum, b_vec.val[1], vec_values.val[1]);
        d += 8;
        remaining -= 8;
    }

    if (remaining >= 4) {
        float32x4_t vec_values = vld1q_f32(vector + d);
        uint32x4_t bit_mask = extract_4_bits_to_mask(bits, d);
        float32x4_t b_vec = vbslq_f32(bit_mask, inv_sqrt_d_vec, neg_inv_sqrt_d_vec);
        sum = vfmaq_f32(sum, b_vec, vec_values);
        d += 4;
        remaining -= 4;
    }

    float32x4_t res_vec = vdupq_n_f32(0.0f);
    float32x4_t res_b = vdupq_n_f32(0.0f);

    if (remaining >= 3) {
        res_vec = vld1q_lane_f32(vector + d, res_vec, 2);
        uint64_t byte_idx = d / 8;
        uint64_t bit_idx = d % 8;
        bool bit_set = (bits[byte_idx] & (1 << bit_idx)) != 0;
        res_b = vsetq_lane_f32(bit_set ? inv_sqrt_d : -inv_sqrt_d, res_b, 2);
        d++;
        remaining--;
    }

    if (remaining >= 2) {
        res_vec = vld1q_lane_f32(vector + d, res_vec, 1);
        uint64_t byte_idx = d / 8;
        uint64_t bit_idx = d % 8;
        bool bit_set = (bits[byte_idx] & (1 << bit_idx)) != 0;
        res_b = vsetq_lane_f32(bit_set ? inv_sqrt_d : -inv_sqrt_d, res_b, 1);
        d++;
        remaining--;
    }

    if (remaining >= 1) {
        res_vec = vld1q_lane_f32(vector + d, res_vec, 0);
        uint64_t byte_idx = d / 8;
        uint64_t bit_idx = d % 8;
        bool bit_set = (bits[byte_idx] & (1 << bit_idx)) != 0;
        res_b = vsetq_lane_f32(bit_set ? inv_sqrt_d : -inv_sqrt_d, res_b, 0);
    }

    if (dim > d) {
        sum = vfmaq_f32(sum, res_b, res_vec);
    }

    return vaddvq_f32(sum);
#else
    return generic::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
#endif
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_NEON)
    if (dim == 0)
        return;
    if (scalar == 0)
        scalar = 1.0f;
    int i = 0;
    float32x4_t scalarVec = vdupq_n_f32(scalar);
    for (; i + 3 < dim; i += 4) {
        float32x4_t vec = vld1q_f32(from + i);
        vec = vdivq_f32(vec, scalarVec);
        vst1q_f32(to + i, vec);
    }
    generic::DivScalar(from + i, to + i, dim - i, scalar);
#else
    generic::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
    float norm = std::sqrt(neon::FP32ComputeIP(from, from, dim));
    neon::DivScalar(from, to, dim, norm);
    return norm;
}

#if defined(ENABLE_NEON)
__inline uint16x8_t __attribute__((__always_inline__))
shuffle_16_char(const uint8x16_t* a, const uint8x16_t* b) {
    int8x16_t tbl = vreinterpretq_s8_u8(*a);
    uint8x16_t idx = *b;
    uint8x16_t idx_masked = vandq_u8(idx, vdupq_n_u8(0x8F));  // avoid using meaningless bits

    return vreinterpretq_u16_s8(vqtbl1q_s8(tbl, idx_masked));
}
#endif

void
Prefetch(const void* data) {
#if defined(ENABLE_NEON)
    __builtin_prefetch(data, 0, 3);
#endif
};

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_NEON)
    uint32x4_t sum[4];
    for (uint64_t i = 0; i < 4; ++i) {
        sum[i] = vdupq_n_u32(0);
    }
    const auto sign4 = vdupq_n_u8(0x0F);
    const auto sign8 = vdupq_n_u16(0xFF);

    for (uint64_t i = 0; i < pq_dim; ++i) {
        auto dict = vld1q_u8(lookup_table);
        auto code = vld1q_u8(codes);
        lookup_table += 16;
        codes += 16;

        auto code1 = vandq_u8(code, sign4);
        auto code2 = vandq_u8(vshrq_n_u8(code, 4), sign4);
        auto res1 = shuffle_16_char(&dict, &code1);
        auto res2 = shuffle_16_char(&dict, &code2);
        sum[0] = vaddq_u32(sum[0], vreinterpretq_u32_u16(vandq_u16(res1, sign8)));
        sum[1] = vaddq_u32(sum[1], vreinterpretq_u32_u16(vshrq_n_u16(res1, 8)));
        sum[2] = vaddq_u32(sum[2], vreinterpretq_u32_u16(vandq_u16(res2, sign8)));
        sum[3] = vaddq_u32(sum[3], vreinterpretq_u32_u16(vshrq_n_u16(res2, 8)));
    }
    alignas(128) uint16_t temp[8];
    for (int64_t i = 0; i < 4; ++i) {
        vst1q_u16(temp, vreinterpretq_u16_u32(sum[i]));
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
#if defined(ENABLE_NEON)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 16) {
        return generic::BitAnd(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 15 < num_byte; i += 16) {
        uint8x16_t x_vec = vld1q_u8(x + i);
        uint8x16_t y_vec = vld1q_u8(y + i);
        uint8x16_t result_vec = vandq_u8(x_vec, y_vec);
        vst1q_u8(result + i, result_vec);
    }
    if (i < num_byte) {
        generic::BitAnd(x + i, y + i, num_byte - i, result + i);
    }
#else
    return generic::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_NEON)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 16) {
        return generic::BitOr(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 15 < num_byte; i += 16) {
        uint8x16_t x_vec = vld1q_u8(x + i);
        uint8x16_t y_vec = vld1q_u8(y + i);
        uint8x16_t result_vec = vorrq_u8(x_vec, y_vec);
        vst1q_u8(result + i, result_vec);
    }
    if (i < num_byte) {
        generic::BitOr(x + i, y + i, num_byte - i, result + i);
    }
#else
    return generic::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_NEON)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 16) {
        return generic::BitXor(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 15 < num_byte; i += 16) {
        uint8x16_t x_vec = vld1q_u8(x + i);
        uint8x16_t y_vec = vld1q_u8(y + i);
        uint8x16_t result_vec = veorq_u8(x_vec, y_vec);
        vst1q_u8(result + i, result_vec);
    }
    if (i < num_byte) {
        generic::BitXor(x + i, y + i, num_byte - i, result + i);
    }
#else
    return generic::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_NEON)
    if (num_byte == 0) {
        return;
    }
    if (num_byte < 16) {
        return generic::BitNot(x, num_byte, result);
    }
    int64_t i = 0;
    for (; i + 15 < num_byte; i += 16) {
        uint8x16_t x_vec = vld1q_u8(x + i);
        uint8x16_t result_vec = veorq_u8(x_vec, vdupq_n_u8(0xFF));
        vst1q_u8(result + i, result_vec);
    }
    if (i < num_byte) {
        generic::BitNot(x + i, num_byte - i, result + i);
    }
#else
    return generic::BitNot(x, num_byte, result);
#endif
}
void
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_NEON)
    size_t base = len % 2;
    size_t offset = base + (len / 2);

    size_t i = 0;
    for (; i + 3 < len / 2; i += 4) {
        float32x4_t first = vld1q_f32(data + i);
        float32x4_t second = vld1q_f32(data + i + offset);

        float32x4_t add = vaddq_f32(first, second);
        float32x4_t sub = vsubq_f32(first, second);

        vst1q_f32(data + i, add);
        vst1q_f32(data + i + offset, sub);
    }

    for (; i < len / 2; i++) {
        float add = data[i] + data[i + offset];
        float sub = data[i] - data[i + offset];
        data[i] = add;
        data[i + offset] = sub;
    }

    if (base != 0) {
        data[len / 2] *= std::sqrt(2.0f);
    }
#else
    generic::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim) {
#if defined(ENABLE_NEON)
    size_t i = 0;

    for (; i + 3 < dim; i += 4) {
        uint8_t byte_val = flip[i / 8];
        uint8_t bit_offset = i % 8;

        uint8_t four_bits = byte_val >> bit_offset;
        if (bit_offset > 4 && (i / 8 + 1) < (dim + 7) / 8) {
            four_bits |= flip[i / 8 + 1] << (8 - bit_offset);
        }

        float32x4_t vec = vld1q_f32(data + i);

        uint32x4_t sign_mask = {(four_bits & 1) ? 0x80000000 : 0,
                                (four_bits & 2) ? 0x80000000 : 0,
                                (four_bits & 4) ? 0x80000000 : 0,
                                (four_bits & 8) ? 0x80000000 : 0};

        vec = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vec), sign_mask));
        vst1q_f32(data + i, vec);
    }

    for (; i < dim; i++) {
        bool mask = (flip[i / 8] & (1 << (i % 8))) != 0;
        if (mask) {
            data[i] = -data[i];
        }
    }
#else
    generic::FlipSign(flip, data, dim);
#endif
}

void
VecRescale(float* data, uint64_t dim, float val) {
#if defined(ENABLE_NEON)
    float32x4_t scale = vdupq_n_f32(val);
    size_t i = 0;

    for (; i + 3 < dim; i += 4) {
        float32x4_t vec = vld1q_f32(data + i);
        vec = vmulq_f32(vec, scale);
        vst1q_f32(data + i, vec);
    }

    for (; i < dim; i++) {
        data[i] *= val;
    }
#else
    generic::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_NEON)
    for (int i = idx; i < dim_; i += 2 * step) {
        int j = 0;

        for (; j + 3 < step; j += 4) {
            float32x4_t x = vld1q_f32(data + i + j);
            float32x4_t y = vld1q_f32(data + i + j + step);

            float32x4_t sum = vaddq_f32(x, y);
            float32x4_t diff = vsubq_f32(x, y);

            vst1q_f32(data + i + j, sum);
            vst1q_f32(data + i + j + step, diff);
        }

        for (; j < step; j++) {
            float x = data[i + j];
            float y = data[i + j + step];
            data[i + j] = x + y;
            data[i + j + step] = x - y;
        }
    }
#else
    generic::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, uint64_t dim_) {
#if defined(ENABLE_NEON)
    size_t n = dim_;
    size_t step = 1;
    while (step < n) {
        neon::RotateOp(data, 0, dim_, step);
        step *= 2;
    }
#else
    generic::FHTRotate(data, dim_);
#endif
}
}  // namespace vsag::neon
