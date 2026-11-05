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
#include <limits>

#include "impl/inner_search_param.h"

namespace vsag {

inline uint32_t
float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline bool
is_finite_distance(float distance) {
    constexpr uint32_t k_exponent_mask = 0x7F800000U;
    return (float_bits(distance) & k_exponent_mask) != k_exponent_mask;
}

inline bool
is_nan_distance(float distance) {
    constexpr uint32_t k_exponent_mask = 0x7F800000U;
    constexpr uint32_t k_mantissa_mask = 0x007FFFFFU;
    const auto bits = float_bits(distance);
    return (bits & k_exponent_mask) == k_exponent_mask and (bits & k_mantissa_mask) != 0U;
}

template <InnerSearchMode mode>
inline bool
is_result_distance_eligible(float distance, const InnerSearchParam& search_param) {
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        return not is_nan_distance(distance);
    }
    // The finite threshold bound is applied after bounded selection. Only non-finite values must
    // be rejected here because negative infinity could otherwise displace eligible finite results.
    return not is_nan_distance(distance) and
           (not search_param.distance_threshold.has_value() or is_finite_distance(distance));
}

inline float
traversal_priority(float distance) {
    // Keep unordered/non-finite bridge nodes traversable without ranking them by their distance;
    // the max-heap consumes this sentinel promptly while result eligibility remains independent.
    return is_finite_distance(distance) ? -distance : std::numeric_limits<float>::max();
}

}  // namespace vsag
