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

namespace vsag {

// Release builds on Linux compile with -Ofast, which implies -ffast-math and
// lets the compiler assume that no NaN or infinity is ever produced. Under that
// assumption std::isfinite/std::isnan fold to a constant and the distance
// filters silently disappear, so test the IEEE-754 exponent bits instead. A
// value is finite when not all exponent bits are set.
inline bool
IsFiniteFloatBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7F800000U) != 0x7F800000U;
}

}  // namespace vsag
