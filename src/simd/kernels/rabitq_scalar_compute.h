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

#include <cstdint>

namespace vsag::simd {

template <typename T>
inline float
RaBitQFloatScalarIPImpl(const float* query,
                        const uint8_t* codes,
                        uint64_t dim,
                        float (*fallback)(const float*, const uint8_t*, uint64_t)) {
    using V = typename T::FloatVec;
    constexpr uint64_t kWidth = T::Width;

    V sum = T::zero();
    uint64_t i = 0;
    for (; i + kWidth <= dim; i += kWidth) {
        const V code_values = T::load_u8_as_float(codes + i);
        sum = T::fmadd(T::load(query + i), code_values, sum);
    }

    float result = T::reduce_add(sum);
    if (i < dim) {
        result += fallback(query + i, codes + i, dim - i);
    }
    return result;
}

template <typename T>
inline uint64_t
RaBitQScalarCodesIPImpl(const uint8_t* codes1,
                        const uint8_t* codes2,
                        uint64_t dim,
                        uint64_t (*fallback)(const uint8_t*, const uint8_t*, uint64_t)) {
    constexpr uint64_t kBytesPerIteration = T::ByteWidth;
    constexpr uint64_t kIterationsPerBlock = 128;
    const auto mask = T::set1_epi16(0xff);
    uint64_t result = 0;
    uint64_t i = 0;

    while (i + kBytesPerIteration <= dim) {
        auto sum = T::zero();
        uint64_t iterations = 0;
        for (; iterations < kIterationsPerBlock and i + kBytesPerIteration <= dim;
             ++iterations, i += kBytesPerIteration) {
            const auto code1 = T::loadu(codes1 + i);
            const auto code2 = T::loadu(codes2 + i);
            const auto code1_low = T::and_si(code1, mask);
            const auto code1_high = T::srli_epi16(code1, 8);
            const auto code2_low = T::and_si(code2, mask);
            const auto code2_high = T::srli_epi16(code2, 8);
            sum = T::add_epi32(sum, T::madd_epi16(code1_low, code2_low));
            sum = T::add_epi32(sum, T::madd_epi16(code1_high, code2_high));
        }
        result += static_cast<uint64_t>(T::reduce_add_epi32(sum));
    }

    if (i < dim) {
        result += fallback(codes1 + i, codes2 + i, dim - i);
    }
    return result;
}

}  // namespace vsag::simd
