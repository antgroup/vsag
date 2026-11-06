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

#include "vector_math.h"

#include <algorithm>
#include <cmath>

namespace vsag::dataset_profile {

double
l2_sqr(const float* a, const float* b, uint64_t dim) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    uint64_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const double d0 = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        const double d1 = static_cast<double>(a[i + 1]) - static_cast<double>(b[i + 1]);
        const double d2 = static_cast<double>(a[i + 2]) - static_cast<double>(b[i + 2]);
        const double d3 = static_cast<double>(a[i + 3]) - static_cast<double>(b[i + 3]);
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    for (; i < dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s0 += d * d;
    }
    return (s0 + s1) + (s2 + s3);
}

double
inner_product(const float* a, const float* b, uint64_t dim) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    uint64_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        s0 += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        s1 += static_cast<double>(a[i + 1]) * static_cast<double>(b[i + 1]);
        s2 += static_cast<double>(a[i + 2]) * static_cast<double>(b[i + 2]);
        s3 += static_cast<double>(a[i + 3]) * static_cast<double>(b[i + 3]);
    }
    for (; i < dim; ++i) {
        s0 += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return (s0 + s1) + (s2 + s3);
}

double
squared_norm(const float* a, uint64_t dim) {
    return inner_product(a, a, dim);
}

void
inner_product_and_l2_sqr(const float* a, const float* b, uint64_t dim, double& ip, double& l2) {
    double p0 = 0.0;
    double p1 = 0.0;
    double q0 = 0.0;
    double q1 = 0.0;
    uint64_t i = 0;
    for (; i + 2 <= dim; i += 2) {
        const double a0 = a[i];
        const double a1 = a[i + 1];
        const double b0 = b[i];
        const double b1 = b[i + 1];
        p0 += a0 * b0;
        p1 += a1 * b1;
        q0 += (a0 - b0) * (a0 - b0);
        q1 += (a1 - b1) * (a1 - b1);
    }
    for (; i < dim; ++i) {
        const double a0 = a[i];
        const double b0 = b[i];
        p0 += a0 * b0;
        q0 += (a0 - b0) * (a0 - b0);
    }
    ip = p0 + p1;
    l2 = q0 + q1;
}

double
metric_distance(
    Metric metric, const float* a, const float* b, uint64_t dim, double norm_a, double norm_b) {
    switch (metric) {
        case Metric::L2:
            return l2_sqr(a, b, dim);
        case Metric::IP:
            return 1.0 - inner_product(a, b, dim);
        case Metric::COSINE: {
            const double denominator = norm_a * norm_b;
            if (denominator <= 0.0) {
                return 1.0;
            }
            return 1.0 - inner_product(a, b, dim) / denominator;
        }
    }
    return l2_sqr(a, b, dim);
}

double
geometric_distance(
    Metric metric, const float* a, const float* b, uint64_t dim, double norm_a, double norm_b) {
    if (metric == Metric::COSINE) {
        double geometric = 0.0;
        geometric_from_metric(
            metric, metric_distance(metric, a, b, dim, norm_a, norm_b), geometric);
        return geometric;
    }
    return std::sqrt(l2_sqr(a, b, dim));
}

bool
geometric_from_metric(Metric metric, double distance, double& geometric) {
    if (metric == Metric::L2) {
        geometric = std::sqrt(std::max(0.0, distance));
        return true;
    }
    if (metric == Metric::COSINE) {
        // ||a/|a| - b/|b|||^2 = 2 (1 - cos) = 2 * cosine_distance.
        geometric = std::sqrt(std::max(0.0, 2.0 * distance));
        return true;
    }
    return false;
}

namespace {

float
float_from_bits(uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t
float_to_half_bits(float value) {
    uint32_t bits = float_bits(value);
    const uint32_t sign = bits & 0x80000000U;
    bits ^= sign;
    uint32_t half = 0;
    if (bits >= 0x47800000U) {
        // 65536 or larger, Inf or NaN.
        half = bits > 0x7F800000U ? 0x7E00U : 0x7C00U;
    } else if (bits < 0x38800000U) {
        // Result is subnormal or zero: align the mantissa with a magic addend so that the
        // FPU performs the round-to-nearest-even step.
        const float magic = float_from_bits(0x3F000000U);
        const float aligned = float_from_bits(bits) + magic;
        half = float_bits(aligned) - float_bits(magic);
    } else {
        const uint32_t mantissa_odd = (bits >> 13U) & 1U;
        bits += 0xC8000FFFU;  // rebias the exponent (15 - 127) and add the rounding bias
        bits += mantissa_odd;
        half = bits >> 13U;
    }
    return static_cast<uint16_t>(half | (sign >> 16U));
}

float
half_bits_to_float(uint16_t half) {
    const uint32_t sign = (static_cast<uint32_t>(half) & 0x8000U) << 16U;
    const uint32_t exponent = (static_cast<uint32_t>(half) >> 10U) & 0x1FU;
    const uint32_t mantissa = static_cast<uint32_t>(half) & 0x3FFU;
    if (exponent == 0) {
        const float magnitude = static_cast<float>(mantissa) * 5.9604644775390625e-8F;
        return sign != 0 ? -magnitude : magnitude;
    }
    if (exponent == 31) {
        return float_from_bits(sign | 0x7F800000U | (mantissa << 13U));
    }
    return float_from_bits(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

}  // namespace

float
round_through_fp16(float value) {
    return half_bits_to_float(float_to_half_bits(value));
}

float
round_through_bf16(float value) {
    if (!is_finite_value(value)) {
        return value;
    }
    const uint32_t rounded = ((float_bits(value) + 0x8000U) >> 16U) << 16U;
    return float_from_bits(rounded);
}

}  // namespace vsag::dataset_profile
