// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cstdint>

namespace vsag::simd {

inline uint8_t*
RaBitQGetSplitPlane(uint8_t* filter_planes,
                    uint8_t* supplement_planes,
                    uint64_t plane_bytes,
                    uint32_t total_bits,
                    uint32_t filter_bits,
                    uint32_t logical_bit) {
    const uint32_t supplement_bits = total_bits - filter_bits;
    if (logical_bit < supplement_bits) {
        return supplement_planes + static_cast<uint64_t>(logical_bit) * plane_bytes;
    }
    const uint32_t filter_plane = total_bits - 1 - logical_bit;
    return filter_planes + static_cast<uint64_t>(filter_plane) * plane_bytes;
}

inline void
RaBitQPackScalarToSplitPlanesTail(const uint8_t* scalar_codes,
                                  uint8_t* filter_planes,
                                  uint8_t* supplement_planes,
                                  uint64_t dim,
                                  uint32_t total_bits,
                                  uint32_t filter_bits,
                                  uint64_t begin) {
    const uint64_t plane_bytes = (dim + 7) / 8;
    for (uint64_t d = begin; d < dim; d += 8) {
        const uint64_t lanes = std::min<uint64_t>(8, dim - d);
        const uint64_t byte_index = d / 8;
        for (uint32_t bit = 0; bit < total_bits; ++bit) {
            uint8_t packed = 0;
            for (uint64_t lane = 0; lane < lanes; ++lane) {
                packed |= static_cast<uint8_t>(((scalar_codes[d + lane] >> bit) & 1U) << lane);
            }
            auto* plane = RaBitQGetSplitPlane(
                filter_planes, supplement_planes, plane_bytes, total_bits, filter_bits, bit);
            plane[byte_index] = packed;
        }
    }
}

}  // namespace vsag::simd
