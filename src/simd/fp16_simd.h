
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

#include "simd_marco.h"
namespace vsag {

#define DECLARE_FP16_FUNCTIONS(ns)                                                                \
    namespace ns {                                                                                \
    float                                                                                         \
    FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim);    \
    float                                                                                         \
    FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim); \
    }  // namespace ns

namespace generic {
float
FP16ToFloat(const uint16_t bf16_value);
uint16_t
FloatToFP16(const float fp32_value);
}  // namespace generic

DECLARE_FP16_FUNCTIONS(generic)
DECLARE_FP16_FUNCTIONS(sse)
DECLARE_FP16_FUNCTIONS(avx)
DECLARE_FP16_FUNCTIONS(avx2)
DECLARE_FP16_FUNCTIONS(avx512)
DECLARE_FP16_FUNCTIONS(neon)
DECLARE_FP16_FUNCTIONS(sve)

#undef DECLARE_FP16_FUNCTIONS

using FP16ComputeType = float (*)(const uint8_t* RESTRICT query,
                                  const uint8_t* RESTRICT codes,
                                  uint64_t dim);
extern FP16ComputeType FP16ComputeIP;
extern FP16ComputeType FP16ComputeL2Sqr;

}  // namespace vsag
