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

#include <limits>

#include "utils/float_utils.h"

namespace vsag {

inline bool
is_result_distance_eligible(float distance) {
    uint32_t bits = 0;
    std::memcpy(&bits, &distance, sizeof(bits));
    // Admit infinities as before, but reject NaNs even under -ffast-math.
    return (bits & 0x7FFFFFFFU) <= 0x7F800000U;
}

inline float
traversal_priority(float distance) {
    // Non-finite nodes can still connect the entry point to valid candidates. Give them a
    // deterministic priority so NaN does not violate the heap's ordering requirements.
    return IsFiniteFloatBits(distance) ? -distance : std::numeric_limits<float>::max();
}

}  // namespace vsag
