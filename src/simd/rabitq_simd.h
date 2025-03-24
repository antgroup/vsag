
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

#include "simd_status.h"

namespace vsag {
namespace avx512 {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);
}  // namespace avx512

namespace avx2 {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);
}  // namespace avx2

namespace avx {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);
}  // namespace avx

namespace sse {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);
}  // namespace sse

namespace generic {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);
}  // namespace generic

using RaBitQFloatBinaryType = float (*)(const float* vector,
                                        const uint8_t* bits,
                                        uint64_t dim,
                                        float inv_sqrt_d);

extern RaBitQFloatBinaryType RaBitQFloatBinaryIP;
}  // namespace vsag
