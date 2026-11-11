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

namespace vsag::half_input {

// Expand input values without changing stored-code quantization semantics.
inline float
FP16ToFloat(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000U) << 16;
    uint32_t exponent = (value >> 10) & 0x1FU;
    uint32_t mantissa = value & 0x3FFU;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t unbiased = -14;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1;
                --unbiased;
            }
            bits =
                sign | (static_cast<uint32_t>(unbiased + 127) << 23) | ((mantissa & 0x3FFU) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace vsag::half_input
