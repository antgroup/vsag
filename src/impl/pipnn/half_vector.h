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

#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "simd/pipnn_half_scalar.h"
#include "simd/pipnn_half_simd.h"

namespace vsag::pipnn_half {

// Decode a nonoverlapping row into dim writable floats. Pointers need only their
// element alignment, not SIMD alignment; dim == 0 permits null pointers. No
// padding is read or written, and all output bits match Decode, including sNaNs.
inline void
DecodeRow(const uint16_t* source, float* destination, uint64_t dim, Format format) {
    if (dim == 0) {
        return;
    }
    const auto& dispatch = GetDispatch();
    (format == Format::FP16 ? dispatch.decode_fp16 : dispatch.decode_bf16)(
        source, destination, dim);
}

// Mixed half/FP32 projection and squared norm for preparation. This avoids
// scalar decode in every arithmetic operation while retaining FP32 accumulation.
inline float
Preparation(const uint16_t* a, const float* b, uint64_t dim, Format format) {
    return GetDispatch().preparation(a, b, dim, format);
}

// FP32 accumulation throughout. Vector reduction/FMA can differ in rounding
// from scalar summation. No half-precision accumulating instructions are used.
inline float
Dot(const uint16_t* a, const uint16_t* b, uint64_t dim, Format format) {
    if (dim == 0) {
        return 0.0F;
    }
    const auto* a_bytes = reinterpret_cast<const uint8_t*>(a);
    const auto* b_bytes = reinterpret_cast<const uint8_t*>(b);
    // Dispatched kernels use IEEE-correct half traits, also for scalar tails.
    // They do not use generic::FP16ToFloat.
    return format == Format::FP16 ? FP16ComputeIP(a_bytes, b_bytes, dim)
                                  : BF16ComputeIP(a_bytes, b_bytes, dim);
}

}  // namespace vsag::pipnn_half
