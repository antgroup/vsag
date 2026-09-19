// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "simd/traits/simd_traits_generic.h"

namespace vsag::lite::detail {

inline uint16_t
encode_fp16(float value) {
    if (not std::isfinite(value)) {
        throw std::invalid_argument("non-finite FP16 input");
    }
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const auto sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t exponent = (bits >> 23) & 0xffU;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0) {
        return sign;
    }
    int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        throw std::invalid_argument("FP16 range overflow");
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000U;
        const auto shift = static_cast<uint32_t>(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1);
        if (remainder > halfway or (remainder == halfway and (rounded & 1U) != 0)) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U or (remainder == 0x1000U and (rounded & 1U) != 0)) {
        ++rounded;
    }
    if (rounded == 0x400U) {
        rounded = 0;
        ++half_exponent;
    }
    if (half_exponent >= 31) {
        throw std::invalid_argument("FP16 rounded range overflow");
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10) | rounded);
}

inline float
decode_fp16(uint16_t value) {
    return simd::FP16Traits<simd::GenericFP16Tag>::load_half(&value);
}

}  // namespace vsag::lite::detail
