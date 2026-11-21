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

#include "pipnn_half_simd.h"
#include "utils/half_input.h"

namespace vsag::pipnn_half {

// IEEE binary16/bfloat16 decode, including signed zero, subnormals, infinities and NaNs.
// Deliberately independent of generic::FP16ToFloat, which only rebiases exponents.
inline float
Decode(uint16_t value, Format format) {
    if (format == Format::FP16) {
        return half_input::FP16ToFloat(value);
    }
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

namespace detail {

// Select the format once per row, including on targets without x86 intrinsics.
template <Format format>
inline void
DecodeRowScalar(const uint16_t* source, float* destination, uint64_t dim) {
    for (uint64_t k = 0; k < dim; ++k) {
        destination[k] = Decode(source[k], format);
    }
}

// Keep contraction control local to this fallback, never to the including TU.
// GCC function options are retained even when the caller uses fast-math.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
inline float
PreparationScalar(const uint16_t* a, const float* b, uint64_t dim, Format format) {
    // Preserve four independent lanes and separate multiply/add on every platform.
    const auto product = [a, b, format](uint64_t k) {
        const float av = Decode(a[k], format);
#if defined(__GNUC__) && !defined(__clang__)
        return av * (b == nullptr ? av : b[k]);
#else
        // Clang fast-math can contract across lexical contract(off) pragmas.
        // Retain the portable barrier until equivalent codegen is proven.
        volatile float value = av * (b == nullptr ? av : b[k]);
        return value;
#endif
    };
    float s0 = 0.0F;
    float s1 = 0.0F;
    float s2 = 0.0F;
    float s3 = 0.0F;
    uint64_t k = 0;
    for (; dim - k >= 4; k += 4) {
        s0 += product(k);
        s1 += product(k + 1);
        s2 += product(k + 2);
        s3 += product(k + 3);
    }
    float result = (s0 + s2) + (s1 + s3);
    for (; k < dim; ++k) {
        result += product(k);
    }
    return result;
}

}  // namespace detail

}  // namespace vsag::pipnn_half
