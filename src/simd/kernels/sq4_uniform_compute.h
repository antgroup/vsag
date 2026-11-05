
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

// T must satisfy UniformCodeTraits: IntVec, ByteWidth, loadu, zero, set1_epi8,
// and_si, srli_epi16, maddubs_epi16, add_epi16, reduce_add_epi16.
template <typename T>
inline float
SQ4UniformComputeCodesIPImpl(const uint8_t* codes1,
                             const uint8_t* codes2,
                             uint64_t dim,
                             float (*fallback)(const uint8_t*, const uint8_t*, uint64_t)) {
    if (dim == 0) {
        return 0;
    }

    constexpr uint64_t kElemsPerIter = T::ByteWidth * 2;
    // Each iteration adds at most ~900 to each int16 lane; overflow at ~36 iters.
    constexpr int kFlushInterval = 32;
    constexpr uint64_t kFlushElems = kElemsPerIter * kFlushInterval;
    int32_t result = 0;
    uint64_t d = 0;
    auto sum = T::zero();
    auto mask = T::set1_epi8(0x0f);

    for (; d + kElemsPerIter - 1 < dim; d += kElemsPerIter) {
        auto xx = T::loadu(codes1 + (d >> 1));
        auto yy = T::loadu(codes2 + (d >> 1));
        auto xx1 = T::and_si(xx, mask);
        auto xx2 = T::and_si(T::srli_epi16(xx, 4), mask);
        auto yy1 = T::and_si(yy, mask);
        auto yy2 = T::and_si(T::srli_epi16(yy, 4), mask);

        sum = T::add_epi16(sum, T::maddubs_epi16(xx1, yy1));
        sum = T::add_epi16(sum, T::maddubs_epi16(xx2, yy2));

        if ((d + kElemsPerIter) % kFlushElems == 0) {
            result += T::reduce_add_epi16(sum);
            sum = T::zero();
        }
    }

    result += T::reduce_add_epi16(sum);

    if (d < dim) {
        result += fallback(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);
    }

    return result;
}

}  // namespace vsag::simd
