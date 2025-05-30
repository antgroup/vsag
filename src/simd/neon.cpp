
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
    auto qty = *((size_t*)qty_ptr);
    return neon::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((size_t*)qty_ptr);
    return neon::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - neon::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return generic::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
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
    for (size_t idx = 0; idx < 256; idx += 8) {
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
FP32ComputeIPBatch4(const float* query,
                    uint64_t dim,
                    const float* codes1,
                    const float* codes2,
                    const float* codes3,
                    const float* codes4,
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
FP32ComputeL2SqrBatch4(const float* query,
                       uint64_t dim,
                       const float* codes1,
                       const float* codes2,
                       const float* codes3,
                       const float* codes4,
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

#if defined(ENABLE_NEON)
__inline uint16x8_t __attribute__((__always_inline__)) load_4_short(const uint16_t* data) {
    uint16_t tmp[] = {data[3], 0, data[2], 0, data[1], 0, data[0], 0};
    return vld1q_u16(tmp);
}
#endif

float
BF16ComputeIP(const uint8_t* query, const uint8_t* codes, uint64_t dim) {
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
BF16ComputeL2Sqr(const uint8_t* query, const uint8_t* codes, uint64_t dim) {
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
        a.val[3] = vsubq_f32(a.val[3], b.val[3]);

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], a.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], a.val[2]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[3], a.val[3]);
        dim -= 16;
        query_bf16 += 16;
        codes_bf16 += 16;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    res.val[2] = vaddq_f32(res.val[2], res.val[3]);
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
    if (dim >= 0) {
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
    }
    return vaddvq_f32(res.val[0]);
#else
    return generic::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* query, const uint8_t* codes, uint64_t dim) {
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
    res.val[2] = vaddq_f32(res.val[2], res.val[3]);
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
    if (dim >= 0) {
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
    }
    return vaddvq_f32(res.val[0]);
#else
    return generic::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* query, const uint8_t* codes, uint64_t dim) {
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
__inline uint8x16_t __attribute__((__always_inline__)) load_4_char(const uint8_t* data) {
    uint8x16_t vec = vdupq_n_u8(0);
    vec = vsetq_lane_u8(data[0], vec, 0);
    vec = vsetq_lane_u8(data[1], vec, 1);
    vec = vsetq_lane_u8(data[2], vec, 2);
    vec = vsetq_lane_u8(data[3], vec, 3);
    return vec;
}

__inline float32x4_t __attribute__((__always_inline__)) get_4_float(uint8x16_t* code_vec) {
    uint8x8_t code_low = vget_low_u8(*code_vec);
    uint16x8_t code_low_16 = vmovl_u8(code_low);
    uint16x4_t code_low_16_low = vget_low_u16(code_low_16);
    uint32x4_t code_values = vmovl_u16(code_low_16_low);
    float32x4_t code_floats = vcvtq_f32_u32(code_values);
    return code_floats;
}
#endif

float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             uint64_t dim) {
#if defined(ENABLE_NEON)
    float32x4_t sum_ = vdupq_n_f32(0.0f);
    uint64_t i = 0;

    for (; i + 3 < dim; i += 4) {
        // load 8bit * 16, front 4B are data
        uint8x16_t code_vec = load_4_char(codes + i);
        float32x4_t code_floats = get_4_float(&code_vec);

        float32x4_t query_values = vld1q_f32(query + i);
        float32x4_t diff_values = vld1q_f32(diff + i);
        float32x4_t lower_bound_values = vld1q_f32(lower_bound + i);

        float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
        float32x4_t scaled_codes = vmulq_f32(code_floats, inv255);
        scaled_codes = vmulq_f32(scaled_codes, diff_values);

        float32x4_t adjusted_codes = vaddq_f32(scaled_codes, lower_bound_values);

        float32x4_t val = vmulq_f32(query_values, adjusted_codes);
        sum_ = vaddq_f32(sum_, val);
    }

    for (; i < dim; ++i) {
        float q = query[i];
        float c = codes[i];
        float lb = lower_bound[i];
        float d = diff[i];
        sum_ += q * ((c / 255.0f) * d + lb);
    }

    return vaddvq_f32(sum_) +
           generic::SQ8ComputeIP(query + i, codes + i, lower_bound + i, diff + i, dim - i);
#else
    return generic::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                uint64_t dim) {
#if defined(ENABLE_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t i = 0;

    for (; i + 3 < dim; i += 4) {
        uint8x16_t code_vec = load_4_char(codes + i);
        float32x4_t code_floats = get_4_float(&code_vec);
        code_floats = vdivq_f32(code_floats, vdupq_n_f32(255.0f));

        float32x4_t diff_values = vld1q_f32(diff + i);
        float32x4_t lower_bound_values = vld1q_f32(lower_bound + i);
        float32x4_t query_values = vld1q_f32(query + i);

        float32x4_t scaled_codes = vmulq_f32(code_floats, diff_values);
        scaled_codes = vaddq_f32(scaled_codes, lower_bound_values);
        float32x4_t val = vsubq_f32(query_values, scaled_codes);

        val = vmulq_f32(val, val);
        sum = vaddq_f32(sum, val);
    }

    return vaddvq_f32(sum) +
           generic::SQ8ComputeL2Sqr(query + i, codes + i, lower_bound + i, diff + i, dim - i);
#else
    return generic::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  uint64_t dim) {
#if defined(ENABLE_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t i = 0;

    for (; i + 3 < dim; i += 4) {
        uint8x16_t code1_vec = load_4_char(codes1 + i);
        uint8x16_t code2_vec = load_4_char(codes2 + i);

        float32x4_t code1_floats = get_4_float(&code1_vec);
        float32x4_t code2_floats = get_4_float(&code2_vec);

        code1_floats = vdivq_f32(code1_floats, vdupq_n_f32(255.0f));
        code2_floats = vdivq_f32(code2_floats, vdupq_n_f32(255.0f));

        float32x4_t diff_values = vld1q_f32(diff + i);
        float32x4_t lower_bound_values = vld1q_f32(lower_bound + i);

        float32x4_t scaled_codes1 =
            vaddq_f32(vmulq_f32(code1_floats, diff_values), lower_bound_values);
        float32x4_t scaled_codes2 =
            vaddq_f32(vmulq_f32(code2_floats, diff_values), lower_bound_values);
        float32x4_t val = vmulq_f32(scaled_codes1, scaled_codes2);
        sum = vaddq_f32(sum, val);
    }

    return vaddvq_f32(sum) +
           generic::SQ8ComputeCodesIP(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
#else
    return generic::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     uint64_t dim) {
#if defined(ENABLE_NEON)

    float32x4_t sum = vdupq_n_f32(0.0f);
    uint64_t i = 0;

    for (; i + 3 < dim; i += 4) {
        // Load data into registers
        uint8x16_t code1_vec = load_4_char(codes1 + i);
        uint8x16_t code2_vec = load_4_char(codes2 + i);

        float32x4_t code1_floats = get_4_float(&code1_vec);
        float32x4_t code2_floats = get_4_float(&code2_vec);

        code1_floats = vdivq_f32(code1_floats, vdupq_n_f32(255.0f));
        code2_floats = vdivq_f32(code2_floats, vdupq_n_f32(255.0f));

        float32x4_t diff_values = vld1q_f32(diff + i);
        float32x4_t lower_bound_values = vld1q_f32(lower_bound + i);

        float32x4_t scaled_codes1 =
            vaddq_f32(vmulq_f32(code1_floats, diff_values), lower_bound_values);
        float32x4_t scaled_codes2 =
            vaddq_f32(vmulq_f32(code2_floats, diff_values), lower_bound_values);
        float32x4_t val = vsubq_f32(scaled_codes1, scaled_codes2);
        val = vmulq_f32(val, val);
        sum = vaddq_f32(sum, val);
    }
    return vaddvq_f32(sum) + generic::SQ8ComputeCodesL2Sqr(
                                 codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
#else
    return generic::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
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
#if defined(ENABLE_NEON)
    uint16x8_t sum_ = vdupq_n_u16(0);

    while (dim >= 32) {
        // get (2*4)bit * (16)
        uint8x16_t a = vld1q_u8(codes1);
        uint8x16_t b = vld1q_u8(codes2);
        uint8x16_t mask = vdupq_n_u8(0x0f);

        // get 4bit * 16
        uint8x16_t a_low = vandq_u8(a, mask);
        uint8x16_t a_high = vandq_u8(vshrq_n_u8(a, 4), mask);
        uint8x16_t b_low = vandq_u8(b, mask);
        uint8x16_t b_high = vandq_u8(vshrq_n_u8(b, 4), mask);

        // Multiply and zero-extend to 16 bits
        uint16x8_t sum_low1 = vmovl_u8(vget_low_u8(a_low));
        uint16x8_t sum_low2 = vmovl_u8(vget_high_u8(a_low));
        uint16x8_t sum_high1 = vmovl_u8(vget_low_u8(a_high));
        uint16x8_t sum_high2 = vmovl_u8(vget_high_u8(a_high));

        // Multiply with b
        sum_low1 = vmulq_u16(sum_low1, vmovl_u8(vget_low_u8(b_low)));
        sum_low2 = vmulq_u16(sum_low2, vmovl_u8(vget_high_u8(b_low)));
        sum_high1 = vmulq_u16(sum_high1, vmovl_u8(vget_low_u8(b_high)));
        sum_high2 = vmulq_u16(sum_high2, vmovl_u8(vget_high_u8(b_high)));

        // Accumulate
        sum_ = vaddq_u16(sum_, sum_low1);
        sum_ = vaddq_u16(sum_, sum_low2);
        sum_ = vaddq_u16(sum_, sum_high1);
        sum_ = vaddq_u16(sum_, sum_high2);

        codes1 += 16;
        codes2 += 16;
        dim -= 32;
    }

    int32_t rem_sum = 0;
    for (size_t i = 0; i < dim; i += 2) {
        uint8_t x_lo = codes1[dim >> 1] & 0x0f;
        uint8_t x_hi = (codes1[dim >> 1] & 0xf0) >> 4;
        uint8_t y_lo = codes2[dim >> 1] & 0x0f;
        uint8_t y_hi = (codes2[dim >> 1] & 0xf0) >> 4;

        rem_sum += (x_lo * y_lo + x_hi * y_hi);
    }

    return static_cast<float>(vaddvq_u16(sum_) + rem_sum);
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
    for (size_t i = 0; i < d; ++i) {
        rem_sum += static_cast<int32_t>(codes1[i]) * static_cast<int32_t>(codes2[i]);
    }

    // accumulate the total sum
    return static_cast<float>(vaddvq_u32(sum_) + rem_sum);
#else
    return generic::SQ8UniformComputeCodesIP(codes1, codes2, d);
#endif
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_NEON)
    return generic::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);  // TODO(zxy): implement
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
    return generic::Prefetch(data);
};

void
PQFastScanLookUp32(const uint8_t* lookup_table,
                   const uint8_t* codes,
                   uint64_t pq_dim,
                   int32_t* result) {
#if defined(ENABLE_NEON)
    uint32x4_t sum[4];
    for (size_t i = 0; i < 4; ++i) {
        sum[i] = vdupq_n_u32(0);
    }
    const auto sign4 = vdupq_n_u8(0x0F);
    const auto sign8 = vdupq_n_u16(0xFF);

    for (size_t i = 0; i < pq_dim; ++i) {
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

}  // namespace vsag::neon
