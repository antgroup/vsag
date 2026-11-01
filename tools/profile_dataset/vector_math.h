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

#pragma once

#include <cstdint>
#include <cstring>

#include "profile_types.h"

namespace vsag::dataset_profile {

// Every kernel accumulates in double with a fixed association order. The profiler target is
// compiled with -fno-fast-math and -ffp-contract=off, so results do not depend on the SIMD
// width, thread count or FMA availability of the machine.

double
l2_sqr(const float* a, const float* b, uint64_t dim);

double
inner_product(const float* a, const float* b, uint64_t dim);

double
squared_norm(const float* a, uint64_t dim);

/// Computes <a, b> and ||a - b||^2 in one pass.
void
inner_product_and_l2_sqr(const float* a, const float* b, uint64_t dim, double& ip, double& l2);

/// Distance in VSAG's convention for the metric. `norm_a` and `norm_b` are L2 norms and are
/// used only for cosine.
double
metric_distance(
    Metric metric, const float* a, const float* b, uint64_t dim, double norm_a, double norm_b);

/// Euclidean distance between the rows (chord distance between normalized rows for cosine).
/// Intrinsic-dimensionality and contrast estimators need a true metric, which `ip` is not.
double
geometric_distance(
    Metric metric, const float* a, const float* b, uint64_t dim, double norm_a, double norm_b);

/// Converts a VSAG metric distance to the geometric distance when it is a monotone function
/// of it (l2 and cosine). Returns false for ip.
bool
geometric_from_metric(Metric metric, double distance, double& geometric);

/// IEEE-754 classification from the bit pattern, immune to -ffinite-math-only.
inline uint32_t
float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline bool
is_finite_value(float value) {
    return (float_bits(value) & 0x7F800000U) != 0x7F800000U;
}

inline bool
is_nan_value(float value) {
    const uint32_t bits = float_bits(value);
    return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
}

/// Bits used for duplicate hashing: -0.0 maps to +0.0 and every NaN to one quiet NaN.
inline uint32_t
canonical_bits(float value) {
    if (value == 0.0F) {
        return 0;
    }
    if (is_nan_value(value)) {
        return 0x7FC00000U;
    }
    return float_bits(value);
}

/// Round-trips through IEEE binary16 with round-to-nearest-even (F16C behavior).
float
round_through_fp16(float value);

/// Round-trips through bfloat16 the way VSAG's FloatToBF16 does (round half up).
float
round_through_bf16(float value);

}  // namespace vsag::dataset_profile
