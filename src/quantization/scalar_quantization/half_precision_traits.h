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

#include "inner_string_params.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"

namespace vsag {

struct FP16Format {
    static constexpr const char* TYPE_NAME = "fp16";

    static uint16_t
    FloatToHalf(float f) {
        return generic::FloatToFP16(f);
    }

    static float
    HalfToFloat(uint16_t h) {
        return generic::FP16ToFloat(h);
    }

    static float
    ComputeIP(const uint8_t* a, const uint8_t* b, uint64_t dim) {
        return FP16ComputeIP(a, b, dim);
    }

    static float
    ComputeL2Sqr(const uint8_t* a, const uint8_t* b, uint64_t dim) {
        return FP16ComputeL2Sqr(a, b, dim);
    }
};

struct BF16Format {
    static constexpr const char* TYPE_NAME = "bf16";

    static uint16_t
    FloatToHalf(float f) {
        return generic::FloatToBF16(f);
    }

    static float
    HalfToFloat(uint16_t h) {
        return generic::BF16ToFloat(h);
    }

    static float
    ComputeIP(const uint8_t* a, const uint8_t* b, uint64_t dim) {
        return BF16ComputeIP(a, b, dim);
    }

    static float
    ComputeL2Sqr(const uint8_t* a, const uint8_t* b, uint64_t dim) {
        return BF16ComputeL2Sqr(a, b, dim);
    }
};

}  // namespace vsag
