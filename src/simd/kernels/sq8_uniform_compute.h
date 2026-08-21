
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

#include "simd/simd_marco.h"

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

// Four-way batched variant: the query block is loaded and split into low/high
// lanes once per iteration and shared across all four accumulators, instead of
// four independent single-code passes. Integer accumulation order per code
// matches SQ8UniformComputeCodesIPImpl exactly, so results are bit-identical.
template <typename T>
inline void
SQ8UniformComputeCodesIPBatch4Impl(const uint8_t* RESTRICT query,
                                   const uint8_t* RESTRICT code1,
                                   const uint8_t* RESTRICT code2,
                                   const uint8_t* RESTRICT code3,
                                   const uint8_t* RESTRICT code4,
                                   uint64_t dim,
                                   float& result1,
                                   float& result2,
                                   float& result3,
                                   float& result4) {
    constexpr uint64_t kElemsPerIter = T::ByteWidth;
    auto sum1 = T::zero();
    auto sum2 = T::zero();
    auto sum3 = T::zero();
    auto sum4 = T::zero();
    auto mask = T::set1_epi16(0xff);

    uint64_t d = 0;
    for (; d + kElemsPerIter - 1 < dim; d += kElemsPerIter) {
        auto xx = T::loadu(query + d);
        auto xx1 = T::and_si(xx, mask);
        auto xx2 = T::srli_epi16(xx, 8);

        auto yy = T::loadu(code1 + d);
        sum1 = T::add_epi32(sum1, T::madd_epi16(xx1, T::and_si(yy, mask)));
        sum1 = T::add_epi32(sum1, T::madd_epi16(xx2, T::srli_epi16(yy, 8)));

        yy = T::loadu(code2 + d);
        sum2 = T::add_epi32(sum2, T::madd_epi16(xx1, T::and_si(yy, mask)));
        sum2 = T::add_epi32(sum2, T::madd_epi16(xx2, T::srli_epi16(yy, 8)));

        yy = T::loadu(code3 + d);
        sum3 = T::add_epi32(sum3, T::madd_epi16(xx1, T::and_si(yy, mask)));
        sum3 = T::add_epi32(sum3, T::madd_epi16(xx2, T::srli_epi16(yy, 8)));

        yy = T::loadu(code4 + d);
        sum4 = T::add_epi32(sum4, T::madd_epi16(xx1, T::and_si(yy, mask)));
        sum4 = T::add_epi32(sum4, T::madd_epi16(xx2, T::srli_epi16(yy, 8)));
    }

    int32_t tail1 = T::reduce_add_epi32(sum1);
    int32_t tail2 = T::reduce_add_epi32(sum2);
    int32_t tail3 = T::reduce_add_epi32(sum3);
    int32_t tail4 = T::reduce_add_epi32(sum4);

    for (; d < dim; ++d) {
        tail1 += static_cast<int32_t>(query[d]) * static_cast<int32_t>(code1[d]);
        tail2 += static_cast<int32_t>(query[d]) * static_cast<int32_t>(code2[d]);
        tail3 += static_cast<int32_t>(query[d]) * static_cast<int32_t>(code3[d]);
        tail4 += static_cast<int32_t>(query[d]) * static_cast<int32_t>(code4[d]);
    }

    result1 = static_cast<float>(tail1);
    result2 = static_cast<float>(tail2);
    result3 = static_cast<float>(tail3);
    result4 = static_cast<float>(tail4);
}

}  // namespace vsag::simd
