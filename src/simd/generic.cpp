
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
    float result = 0.0f;

    for (uint64_t i = 0; i < dim; ++i) {
        result += query[i] * codes[i];
    }
    return result;
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
    float result = 0.0f;
    for (uint64_t i = 0; i < dim; ++i) {
        auto val = query[i] - codes[i];
        result += val * val;
    }
    return result;
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
    for (uint64_t i = 0; i < dim; ++i) {
        result1 += query[i] * codes1[i];
        result2 += query[i] * codes2[i];
        result3 += query[i] * codes3[i];
        result4 += query[i] * codes4[i];
    }
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
    for (uint64_t i = 0; i < dim; ++i) {
        result1 += (query[i] - codes1[i]) * (query[i] - codes1[i]);
        result2 += (query[i] - codes2[i]) * (query[i] - codes2[i]);
        result3 += (query[i] - codes3[i]) * (query[i] - codes3[i]);
        result4 += (query[i] - codes4[i]) * (query[i] - codes4[i]);
    }
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
    for (uint64_t i = 0; i < dim; ++i) {
        z[i] = x[i] - y[i];
    }
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
    for (uint64_t i = 0; i < dim; ++i) {
        z[i] = x[i] + y[i];
    }
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
    for (uint64_t i = 0; i < dim; ++i) {
        z[i] = x[i] * y[i];
    }
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
    for (uint64_t i = 0; i < dim; ++i) {
        z[i] = x[i] / y[i];
    }
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
    float result = 0.0F;
    for (uint64_t i = 0; i < dim; ++i) {
        result += x[i];
    }
    return result;
}

union FP32Struct {
    uint32_t int_value;
    float float_value;
};

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
    float result = 0.0f;
    for (uint64_t i = 0; i < dim; ++i) {
        auto val = static_cast<float>(query[i] - codes[i]);
        result += val * val;
    }
    return result;
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
    float result = 0.0f;
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);
    for (uint64_t i = 0; i < dim; ++i) {
        result += BF16ToFloat(query_bf16[i]) * BF16ToFloat(codes_bf16[i]);
    }
    return result;
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    float result = 0.0f;
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);
    for (uint64_t i = 0; i < dim; ++i) {
        auto val = BF16ToFloat(query_bf16[i]) - BF16ToFloat(codes_bf16[i]);
        result += val * val;
    }
    return result;
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    float result = 0.0f;
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);
    for (uint64_t i = 0; i < dim; ++i) {
        result += FP16ToFloat(query_bf16[i]) * FP16ToFloat(codes_bf16[i]);
    }
    return result;
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    float result = 0.0f;
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);
    for (uint64_t i = 0; i < dim; ++i) {
        auto val = FP16ToFloat(query_bf16[i]) - FP16ToFloat(codes_bf16[i]);
        result += val * val;
    }
    return result;
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
    float result = 0.0f;
    for (uint64_t i = 0; i < dim; ++i) {
        result += query[i] * static_cast<float>(static_cast<float>(codes[i]) / 255.0 * diff[i] +
                                                lower_bound[i]);
    }
    return result;
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
    float result = 0.0f;
    for (uint64_t i = 0; i < dim; ++i) {
        auto val = (query[i] - static_cast<float>(static_cast<float>(codes[i]) / 255.0 * diff[i] +
                                                  lower_bound[i]));
        result += val * val;
    }
    return result;
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
    float result = 0.0f;
    for (uint64_t i = 0; i < dim; ++i) {
        auto val1 =
            static_cast<float>(static_cast<float>(codes1[i]) / 255.0 * diff[i] + lower_bound[i]);
        auto val2 =
            static_cast<float>(static_cast<float>(codes2[i]) / 255.0 * diff[i] + lower_bound[i]);
        result += val1 * val2;
    }
    return result;
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
    float result = 0.0f;
    for (uint64_t i = 0; i < dim; ++i) {
        auto val1 =
            static_cast<float>(static_cast<float>(codes1[i]) / 255.0 * diff[i] + lower_bound[i]);
        auto val2 =
            static_cast<float>(static_cast<float>(codes2[i]) / 255.0 * diff[i] + lower_bound[i]);
        result += (val1 - val2) * (val1 - val2);
    }
    return result;
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
    float result = 0;
    float x_lo = 0, x_hi = 0, y_lo = 0, y_hi = 0;

    for (uint64_t d = 0; d < dim; d += 2) {
        x_lo = query[d];
        y_lo = (codes[d >> 1] & 0x0f) / 15.0 * diff[d] + lower_bound[d];
        if (d + 1 < dim) {
            x_hi = query[d + 1];
            y_hi = (codes[d >> 1] >> 4) / 15.0 * diff[d + 1] + lower_bound[d + 1];
        } else {
            x_hi = 0;
            y_hi = 0;
        }

        result += (x_lo * y_lo + x_hi * y_hi);
    }

    return result;
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
    float result = 0;
    float x_lo = 0, x_hi = 0, y_lo = 0, y_hi = 0;

    for (uint64_t d = 0; d < dim; d += 2) {
        x_lo = query[d];
        y_lo = (codes[d >> 1] & 0x0f) / 15.0 * diff[d] + lower_bound[d];
        if (d + 1 < dim) {
            x_hi = query[d + 1];
            y_hi = ((codes[d >> 1] & 0xf0) >> 4) / 15.0 * diff[d + 1] + lower_bound[d + 1];
        } else {
            x_hi = 0;
            y_hi = 0;
        }

        result += (x_lo - y_lo) * (x_lo - y_lo) + (x_hi - y_hi) * (x_hi - y_hi);
    }

    return result;
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
    float result = 0, delta = 0;
    float x_lo = 0, x_hi = 0, y_lo = 0, y_hi = 0;

    for (uint64_t d = 0; d < dim; d += 2) {
        x_lo = (codes1[d >> 1] & 0x0f) / 15.0 * diff[d] + lower_bound[d];
        y_lo = (codes2[d >> 1] & 0x0f) / 15.0 * diff[d] + lower_bound[d];
        if (d + 1 < dim) {
            x_hi = ((codes1[d >> 1] & 0xf0) >> 4) / 15.0 * diff[d + 1] + lower_bound[d + 1];
            y_hi = ((codes2[d >> 1] & 0xf0) >> 4) / 15.0 * diff[d + 1] + lower_bound[d + 1];
        } else {
            x_hi = 0;
            y_hi = 0;
        }

        result += (x_lo * y_lo + x_hi * y_hi);
    }

    return result;
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
    float result = 0, delta = 0;
    float x_lo = 0, x_hi = 0, y_lo = 0, y_hi = 0;

    for (uint64_t d = 0; d < dim; d += 2) {
        x_lo = (codes1[d >> 1] & 0x0f) / 15.0 * diff[d] + lower_bound[d];
        y_lo = (codes2[d >> 1] & 0x0f) / 15.0 * diff[d] + lower_bound[d];
        if (d + 1 < dim) {
            x_hi = ((codes1[d >> 1] & 0xf0) >> 4) / 15.0 * diff[d + 1] + lower_bound[d + 1];
            y_hi = ((codes2[d >> 1] & 0xf0) >> 4) / 15.0 * diff[d + 1] + lower_bound[d + 1];
        } else {
            x_hi = 0;
            y_hi = 0;
        }

        result += (x_lo - y_lo) * (x_lo - y_lo) + (x_hi - y_hi) * (x_hi - y_hi);
    }

    return result;
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

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
    if (dim == 0) {
        return 0.0f;
    }

    float result = 0.0f;

    for (std::size_t d = 0; d < dim; ++d) {
        bool bit = ((bits[d / 8] >> (d % 8)) & 1) != 0;
        float b_i = bit ? inv_sqrt_d : -inv_sqrt_d;
        result += b_i * vector[d];
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
    size_t num_bytes = (dim + 7) / 8;
    size_t num_blocks = num_bytes / 8;
    size_t remainder = num_bytes % 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        const uint64_t* codes_block =
            reinterpret_cast<const uint64_t*>(codes + bit_pos * num_bytes);
        const uint64_t* bits_block = reinterpret_cast<const uint64_t*>(bits);

        for (size_t i = 0; i < num_blocks; ++i) {
            uint64_t bitwise_and = codes_block[i] & bits_block[i];
            result += __builtin_popcountll(bitwise_and) << bit_pos;
        }

        if (remainder > 0) {
            uint64_t leftover_code = 0;
            uint64_t leftover_bits = 0;

            for (size_t i = 0; i < remainder; ++i) {
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
    float norm = 0;
    for (uint64_t d = 0; d < dim; ++d) {
        norm += (from[d] - centroid[d]) * (from[d] - centroid[d]);
    }

    if (norm < 1e-5) {
        norm = 1;
    } else {
        norm = std::sqrt(norm);
    }

    for (int d = 0; d < dim; d++) {
        to[d] = (from[d] - centroid[d]) / norm;
    }

    return norm;
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
    for (int d = 0; d < dim; d++) {
        to[d] = from[d] * norm + centroid[d];
    }
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
    if (dim == 0) {
        return;
    }
    if (scalar == 0) {
        scalar = 1.0f;  // TODO(LHT): logger?
    }
    for (uint64_t i = 0; i < dim; ++i) {
        to[i] = from[i] / scalar;
    }
}

void
Prefetch(const void* data){};

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
    for (size_t i = 0; i < pq_dim; i++) {
        const auto* dict = lookup_table;
        lookup_table += 16;
        const auto* code = codes;
        codes += 16;
        for (size_t j = 0; j < 16; j++) {
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
KacsWalk(float* data, size_t len) {
    size_t base = len % 2;
    size_t offset = base + (len / 2);  // for odd dim
    for (size_t i = 0; i < len / 2; i++) {
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
}

void
FlipSign(const uint8_t* flip, float* data, size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        bool mask = (flip[i / 8] & (1 << (i % 8))) != 0;
        if (mask) {
            data[i] = -data[i];
        }
    }
}

void
VecRescale(float* data, size_t dim, float val) {
    for (int i = 0; i < dim; i++) {
        data[i] *= val;
    }
}

void
RotateOp(float* data, int idx, int dim_, int step) {
    for (int i = idx; i < dim_; i += 2 * step) {
        for (int j = 0; j < step; j++) {
            float x = data[i + j];
            float y = data[i + j + step];
            data[i + j] = x + y;
            data[i + j + step] = x - y;
        }
    }
}

void
FHTRotate(float* data, size_t dim_) {
    size_t n = dim_;
    size_t step = 1;
    while (step < n) {
        generic::RotateOp(data, 0, dim_, step);
        step *= 2;
    }
}

}  // namespace vsag::generic
