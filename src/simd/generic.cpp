
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

#include "simd.h"
#include "simd/int8_simd.h"
#include "simd/kernels/kernels.h"
#include "simd/traits/simd_traits_generic.h"

namespace vsag::generic {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    uint64_t qty = *((uint64_t*)qty_ptr);

    float res = 0.0f;
    for (uint64_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return res;
}

float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    uint64_t qty = *((uint64_t*)qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float*)pVect1)[i] * ((float*)pVect2)[i];
    }
    return res;
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    uint64_t qty = *((uint64_t*)qty_ptr);

    float res = 0.0f;
    for (uint64_t i = 0; i < qty; ++i) {
        float t = static_cast<float>(*pVect1 - *pVect2);
        pVect1++;
        pVect2++;
        res += t * t;
    }

    return res;
}

float
INT8InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    uint64_t qty = *((uint64_t*)qty_ptr);
    auto* vec1 = (int8_t*)pVect1;
    auto* vec2 = (int8_t*)pVect2;
    double res = 0;
    for (uint64_t i = 0; i < qty; i++) {
        res += vec1[i] * vec2[i];
    }
    return static_cast<float>(res);
}

float
INT8InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -INT8InnerProduct(pVect1, pVect2, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
    const auto* float_centers = (const float*)single_dim_centers;
    auto* float_result = (float*)result;
    for (uint64_t idx = 0; idx < 256; idx++) {
        double diff = float_centers[idx] - single_dim_val;
        float_result[idx] += (float)(diff * diff);
    }
}

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
    return simd::ComputeIPImpl<simd::SimdTraits<simd::Generic_Tag>>(query, codes, dim);
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::Generic_Tag>>(query, codes, dim);
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
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::Generic_Tag>, simd::Batch4Kind::IP>(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
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
    simd::ComputeBatch4Impl<simd::SimdTraits<simd::Generic_Tag>, simd::Batch4Kind::L2>(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
    simd::BinaryOpImpl<simd::SimdTraits<simd::Generic_Tag>, simd::BinaryOp::Sub>(x, y, z, dim);
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
    simd::BinaryOpImpl<simd::SimdTraits<simd::Generic_Tag>, simd::BinaryOp::Add>(x, y, z, dim);
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
    simd::BinaryOpImpl<simd::SimdTraits<simd::Generic_Tag>, simd::BinaryOp::Mul>(x, y, z, dim);
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
    simd::BinaryOpImpl<simd::SimdTraits<simd::Generic_Tag>, simd::BinaryOp::Div>(x, y, z, dim);
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
    return simd::ReduceAddImpl<simd::SimdTraits<simd::Generic_Tag>>(x, dim);
}

union FP32Struct {
    uint32_t int_value;
    float float_value;
};

float
INT8ComputeL2Sqr(const int8_t* query, const int8_t* codes, uint64_t dim) {
    return simd::Int8ComputeL2SqrImpl<simd::Int8Traits<simd::Generic_Int8_Tag>>(query, codes, dim);
}

float
INT8ComputeIP(const int8_t* query, const int8_t* codes, uint64_t dim) {
    return simd::Int8ComputeIPImpl<simd::Int8Traits<simd::Generic_Int8_Tag>>(query, codes, dim);
}

float
BF16ToFloat(const uint16_t bf16_value) {
    FP32Struct fp32;
    fp32.int_value = (static_cast<uint32_t>(bf16_value) << 16);
    return fp32.float_value;
}

uint16_t
FloatToBF16(const float fp32_value) {
    FP32Struct fp32;
    fp32.float_value = fp32_value;
    return static_cast<uint16_t>((fp32.int_value + 0x8000) >> 16);
}

float
FP16ToFloat(const uint16_t fp16_value) {
    uint32_t sign = (fp16_value >> 15) & 0x1;
    int32_t exp = ((fp16_value >> 10) & 0x1F) - 15;
    uint32_t mantissa = (fp16_value & 0x3FF) << 13;
    FP32Struct fp32;
    fp32.int_value = (sign << 31) | ((exp + 127) << 23) | mantissa;
    return fp32.float_value;
}

uint16_t
FloatToFP16(const float fp32_value) {
    FP32Struct fp32;
    fp32.float_value = fp32_value;
    uint16_t sign = (fp32.int_value >> 31) & 0x1;
    int32_t exp = ((fp32.int_value >> 23) & 0xFF) - 127;
    uint32_t mantissa = fp32.int_value & 0x007FFFFF;

    if (exp > 15) {
        exp = 15;
    } else if (exp < -14) {
        exp = -14;
    }
    return (sign << 15) | ((exp + 15) << 10) | (mantissa >> 13);
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    return simd::HalfComputeIPImpl<simd::BF16Traits<simd::Generic_BF16_Tag>>(
        query, codes, dim, nullptr);
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    return simd::HalfComputeL2SqrImpl<simd::BF16Traits<simd::Generic_BF16_Tag>>(
        query, codes, dim, nullptr);
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    return simd::HalfComputeIPImpl<simd::FP16Traits<simd::Generic_FP16_Tag>>(
        query, codes, dim, nullptr);
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    return simd::HalfComputeL2SqrImpl<simd::FP16Traits<simd::Generic_FP16_Tag>>(
        query, codes, dim, nullptr);
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
    return simd::SQ8ComputeIPImpl<simd::SQ8Traits<simd::Generic_SQ8_Tag>>(
        query, codes, lower_bound, diff, dim);
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
    return simd::SQ8ComputeL2SqrImpl<simd::SQ8Traits<simd::Generic_SQ8_Tag>>(
        query, codes, lower_bound, diff, dim);
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
    return simd::SQ8ComputeCodesIPImpl<simd::SQ8Traits<simd::Generic_SQ8_Tag>>(
        codes1, codes2, lower_bound, diff, dim);
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
    return simd::SQ8ComputeCodesL2SqrImpl<simd::SQ8Traits<simd::Generic_SQ8_Tag>>(
        codes1, codes2, lower_bound, diff, dim);
}

namespace {
float
SQ4ScalarIP(const float* q, const uint8_t* c, const float* lb, const float* d, uint64_t dim) {
    float result = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        uint8_t nibble = (i & 1) ? (c[i >> 1] >> 4) : (c[i >> 1] & 0x0f);
        result += q[i] * (nibble * (1.0f / 15.0f) * d[i] + lb[i]);
    }
    return result;
}
float
SQ4ScalarL2(const float* q, const uint8_t* c, const float* lb, const float* d, uint64_t dim) {
    float result = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        uint8_t nibble = (i & 1) ? (c[i >> 1] >> 4) : (c[i >> 1] & 0x0f);
        float v = nibble * (1.0f / 15.0f) * d[i] + lb[i];
        float diff = q[i] - v;
        result += diff * diff;
    }
    return result;
}
float
SQ4ScalarCodesIP(
    const uint8_t* c1, const uint8_t* c2, const float* lb, const float* d, uint64_t dim) {
    float result = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        uint8_t n1 = (i & 1) ? (c1[i >> 1] >> 4) : (c1[i >> 1] & 0x0f);
        uint8_t n2 = (i & 1) ? (c2[i >> 1] >> 4) : (c2[i >> 1] & 0x0f);
        float v1 = n1 * (1.0f / 15.0f) * d[i] + lb[i];
        float v2 = n2 * (1.0f / 15.0f) * d[i] + lb[i];
        result += v1 * v2;
    }
    return result;
}
float
SQ4ScalarCodesL2(
    const uint8_t* c1, const uint8_t* c2, const float* lb, const float* d, uint64_t dim) {
    float result = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        uint8_t n1 = (i & 1) ? (c1[i >> 1] >> 4) : (c1[i >> 1] & 0x0f);
        uint8_t n2 = (i & 1) ? (c2[i >> 1] >> 4) : (c2[i >> 1] & 0x0f);
        float v1 = n1 * (1.0f / 15.0f) * d[i] + lb[i];
        float v2 = n2 * (1.0f / 15.0f) * d[i] + lb[i];
        float diff = v1 - v2;
        result += diff * diff;
    }
    return result;
}
}  // namespace

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
    return simd::SQ4ComputeIPImpl<simd::SQ4Traits<simd::Generic_SQ4_Tag>>(
        query, codes, lower_bound, diff, dim, &SQ4ScalarIP);
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
    return simd::SQ4ComputeL2SqrImpl<simd::SQ4Traits<simd::Generic_SQ4_Tag>>(
        query, codes, lower_bound, diff, dim, &SQ4ScalarL2);
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
    return simd::SQ4ComputeCodesIPImpl<simd::SQ4Traits<simd::Generic_SQ4_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &SQ4ScalarCodesIP);
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
    return simd::SQ4ComputeCodesL2SqrImpl<simd::SQ4Traits<simd::Generic_SQ4_Tag>>(
        codes1, codes2, lower_bound, diff, dim, &SQ4ScalarCodesL2);
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
    int32_t result = 0;

    for (uint64_t d = 0; d < dim; d += 2) {
        float x_lo = codes1[d >> 1] & 0x0f;
        float x_hi = (codes1[d >> 1] & 0xf0) >> 4;
        float y_lo = codes2[d >> 1] & 0x0f;
        float y_hi = (codes2[d >> 1] & 0xf0) >> 4;

        result += (x_lo * y_lo + x_hi * y_hi);
    }

    return result;
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
    int32_t result = 0;
    for (uint64_t d = 0; d < dim; d++) {
        result += codes1[d] * codes2[d];
    }
    return static_cast<float>(result);
}

void
SQ8UniformComputeCodesIPBatch(const uint8_t* RESTRICT query,
                              const uint8_t* RESTRICT codes,
                              uint64_t dim,
                              uint64_t n_codes,
                              uint64_t code_stride,
                              float* RESTRICT out) {
    for (uint64_t i = 0; i < n_codes; ++i) {
        out[i] = generic::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
    if (dim == 0) {
        return 0.0f;
    }

    float result = 0.0f;

    float neg = 0, pos = 0;
    if (inv_sqrt_d > 1e-3) {
        pos = inv_sqrt_d;
        neg = -inv_sqrt_d;
    } else {
        pos = 1.0f;
        neg = 0;
    }

    for (uint64_t d = 0; d < dim; ++d) {
        bool bit = ((bits[d / 8] >> (d % 8)) & 1) != 0;
        float b_i = bit ? pos : neg;
        result += b_i * vector[d];
    }

    return result;
}

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results) {
    results[0] = 0.0F;
    results[1] = 0.0F;
    results[2] = 0.0F;
    results[3] = 0.0F;
    if (dim == 0) {
        return;
    }

    const float pos = inv_sqrt_d > 1e-3F ? inv_sqrt_d : 1.0F;
    const float neg = inv_sqrt_d > 1e-3F ? -inv_sqrt_d : 0.0F;
    for (uint64_t d = 0; d < dim; ++d) {
        const uint64_t byte_id = d >> 3;
        const uint8_t bit_mask = static_cast<uint8_t>(1U << (d & 7));
        const float value = vector[d];
        results[0] += ((bits1[byte_id] & bit_mask) != 0U ? pos : neg) * value;
        results[1] += ((bits2[byte_id] & bit_mask) != 0U ? pos : neg) * value;
        results[2] += ((bits3[byte_id] & bit_mask) != 0U ? pos : neg) * value;
        results[3] += ((bits4[byte_id] & bit_mask) != 0U ? pos : neg) * value;
    }
}

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits) {
    if (dim == 0) {
        return 0.0f;
    }

    const uint64_t plane_bytes = (dim + 7) / 8;
    const uint32_t one_bit_weight = 1U << supplement_bits;
    float result = 0.0f;

    for (uint64_t d = 0; d < dim; ++d) {
        const uint64_t byte_idx = d >> 3;
        const uint8_t bit_mask = static_cast<uint8_t>(1U << (d & 7));
        uint32_t code = (one_bit_code[byte_idx] & bit_mask) != 0 ? one_bit_weight : 0U;
        for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
            const auto* plane = supplement_code + static_cast<uint64_t>(bit) * plane_bytes;
            if ((plane[byte_idx] & bit_mask) != 0) {
                code += 1U << bit;
            }
        }
        result += vector[d] * static_cast<float>(code);
    }

    return result;
}

float
RaBitQFloatSQIP(const float* vector, const uint8_t* codes, uint64_t dim) {
    if (dim == 0) {
        return 0.0f;
    }

    float result = 0.0f;
    for (uint64_t d = 0; d < dim; ++d) {
        result += float(codes[d]) * vector[d];
    }

    return result;
}

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
    // note that this func requiere the redident part in codes and bits is 0
    // e.g., suppose dim = 10, then the value of bit pos = 11 to 15 should be 0
    if (dim == 0) {
        return 0.0f;
    }

    uint32_t result = 0;
    uint64_t num_bytes = (dim + 7) / 8;
    uint64_t num_blocks = num_bytes / 8;
    uint64_t remainder = num_bytes % 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        const uint64_t* codes_block =
            reinterpret_cast<const uint64_t*>(codes + bit_pos * num_bytes);
        const uint64_t* bits_block = reinterpret_cast<const uint64_t*>(bits);

        for (uint64_t i = 0; i < num_blocks; ++i) {
            uint64_t bitwise_and = codes_block[i] & bits_block[i];
            result += __builtin_popcountll(bitwise_and) << bit_pos;
        }

        if (remainder > 0) {
            uint64_t leftover_code = 0;
            uint64_t leftover_bits = 0;

            for (uint64_t i = 0; i < remainder; ++i) {
                leftover_code |=
                    static_cast<uint64_t>(codes[bit_pos * num_bytes + num_blocks * 8 + i])
                    << (i * 8);
                leftover_bits |= static_cast<uint64_t>(bits[num_blocks * 8 + i]) << (i * 8);
            }

            uint64_t bitwise_and = leftover_code & leftover_bits;
            result += __builtin_popcountll(bitwise_and) << bit_pos;
        }
    }

    return result;
}

float
Normalize(const float* from, float* to, uint64_t dim) {
    float norm = std::sqrt(FP32ComputeIP(from, from, dim));
    generic::DivScalar(from, to, dim, norm);
    return norm;
}

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
    return simd::NormalizeWithCentroidImpl<simd::SimdTraits<simd::Generic_Tag>>(
        from, centroid, to, dim);
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
    simd::InverseNormalizeWithCentroidImpl<simd::SimdTraits<simd::Generic_Tag>>(
        from, centroid, to, dim, norm);
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
    simd::DivScalarImpl<simd::SimdTraits<simd::Generic_Tag>>(from, to, dim, scalar);
}

void
Prefetch(const void* data){};

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
    for (uint64_t i = 0; i < pq_dim; i++) {
        const auto* dict = lookup_table;
        lookup_table += 16;
        const auto* code = codes;
        codes += 16;
        for (uint64_t j = 0; j < 16; j++) {
            if (j % 2 == 0) {
                result[j / 2] += static_cast<uint32_t>(dict[code[j] & 0x0F]);
                result[16 + j / 2] += static_cast<uint32_t>(dict[(code[j] >> 4)]);
            } else {
                result[8 + j / 2] += static_cast<uint32_t>(dict[code[j] & 0x0F]);
                result[24 + j / 2] += static_cast<uint32_t>(dict[(code[j] >> 4)]);
            }
        }
    }
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
    for (uint64_t i = 0; i < num_byte; i++) {
        result[i] = x[i] & y[i];
    }
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
    for (uint64_t i = 0; i < num_byte; i++) {
        result[i] = x[i] | y[i];
    }
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
    for (uint64_t i = 0; i < num_byte; i++) {
        result[i] = x[i] ^ y[i];
    }
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
    for (uint64_t i = 0; i < num_byte; i++) {
        result[i] = ~x[i];
    }
}

void
KacsWalk(float* data, uint64_t len) {
    simd::KacsWalkImpl<simd::SimdTraits<simd::Generic_Tag>>(data, len);
}

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim) {
    for (uint64_t i = 0; i < dim; i++) {
        bool mask = (flip[i / 8] & (1 << (i % 8))) != 0;
        if (mask) {
            data[i] = -data[i];
        }
    }
}

void
VecRescale(float* data, uint64_t dim, float val) {
    simd::VecRescaleImpl<simd::SimdTraits<simd::Generic_Tag>>(data, dim, val);
}

void
RotateOp(float* data, int idx, int dim_, int step) {
    simd::RotateOpImpl<simd::SimdTraits<simd::Generic_Tag>>(data, idx, dim_, step);
}

void
FHTRotate(float* data, uint64_t dim_) {
    uint64_t n = dim_;
    uint64_t step = 1;
    while (step < n) {
        generic::RotateOp(data, 0, dim_, step);
        step *= 2;
    }
}

}  // namespace vsag::generic
