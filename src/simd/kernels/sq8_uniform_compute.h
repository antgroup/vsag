
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

namespace vsag::simd {

// T must satisfy UniformCodeTraits: IntVec, ByteWidth, loadu, zero, set1_epi16,
// and_si, srli_epi16, madd_epi16, add_epi32, reduce_add_epi32.
template <typename T>
inline float
SQ8UniformComputeCodesIPImpl(const uint8_t* codes1,
                             const uint8_t* codes2,
                             uint64_t dim,
                             float (*fallback)(const uint8_t*, const uint8_t*, uint64_t)) {
    if (dim == 0) {
        return 0.0f;
    }

    constexpr uint64_t kElemsPerIter = T::ByteWidth;
    uint64_t d = 0;
    auto sum = T::zero();
    auto mask = T::set1_epi16(0xff);

    for (; d + kElemsPerIter - 1 < dim; d += kElemsPerIter) {
        auto xx = T::loadu(codes1 + d);
        auto yy = T::loadu(codes2 + d);

        auto xx1 = T::and_si(xx, mask);
        auto xx2 = T::srli_epi16(xx, 8);
        auto yy1 = T::and_si(yy, mask);
        auto yy2 = T::srli_epi16(yy, 8);

        sum = T::add_epi32(sum, T::madd_epi16(xx1, yy1));
        sum = T::add_epi32(sum, T::madd_epi16(xx2, yy2));
    }

    int32_t result = T::reduce_add_epi32(sum);

    if (d < dim) {
        result += static_cast<int32_t>(fallback(codes1 + d, codes2 + d, dim - d));
    }

    return static_cast<float>(result);
}

}  // namespace vsag::simd
