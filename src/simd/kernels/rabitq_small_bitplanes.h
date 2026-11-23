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

template <typename T, uint32_t Bits, bool Centered, uint32_t Count>
inline void
ComputeSmallBitplaneIP(const float* query,
                       const uint8_t* const (&codes)[Count],
                       uint64_t dim,
                       uint32_t reorder_bits,
                       float* result) {
    constexpr uint64_t width = 4;
    static_assert(T::Width == width);
    const uint64_t plane_bytes = (dim + 7) / 8;
    typename T::FloatVec accum[Count];
    typename T::FloatVec positive[Bits];
    float weights[Bits];
    const float offset = Centered ? (static_cast<float>((1U << Bits) - 1U) * -0.5F) : 0.0F;
    const auto origin = T::set1(offset);
    for (uint32_t b = 0; b < Bits; ++b) {
        weights[b] = static_cast<float>(1U << (reorder_bits + Bits - b - 1));
        positive[b] = T::set1(weights[b]);
    }
    for (auto& lane : accum) {
        lane = T::zero();
    }
    uint64_t d = 0;
    for (; d + width <= dim; d += width) {
        const auto q = T::load(query + d);
        for (uint32_t c = 0; c < Count; ++c) {
            auto value = origin;
            for (uint32_t b = 0; b < Bits; ++b) {
                const uint8_t packed = codes[c][b * plane_bytes + d / 8];
                value = T::add(value, T::SelectBits(packed >> (d & 7), positive[b]));
            }
            accum[c] = T::fmadd(value, q, accum[c]);
        }
    }
    for (uint32_t c = 0; c < Count; ++c) {
        result[c] = T::reduce_add(accum[c]);
        for (uint64_t i = d; i < dim; ++i) {
            float value = offset;
            for (uint32_t b = 0; b < Bits; ++b) {
                if ((codes[c][b * plane_bytes + i / 8] & (1U << (i & 7))) != 0) {
                    value += weights[b];
                }
            }
            result[c] += value * query[i];
        }
    }
}

// Narrow SIMD has no indexed float gather. Gather four scalar values into a
// vector; for batch4 each lane corresponds to a candidate, preserving its
// accumulation order and sharing the LUT/weight loop.
template <typename T, uint32_t Count>
void
ComputeSmallLookupIP(const float* lookup,
                     const uint8_t* const (&codes)[Count],
                     uint64_t dim,
                     uint32_t reorder_bits,
                     uint32_t filter_bits,
                     float* results) {
    static_assert(T::Width == 4 && (Count == 1 || Count == 4));
    const auto plane_bytes = (dim + 7) / 8;
    auto sum = T::zero();
    uint64_t b = 0;
    if constexpr (Count == 1) {
        for (; b + 4 <= plane_bytes; b += 4) {
            const auto* table = lookup + b * 256;
            for (uint32_t bit = 0; bit < filter_bits; ++bit) {
                const auto* p = codes[0] + bit * plane_bytes + b;
                const float values[4]{
                    table[p[0]], table[256 + p[1]], table[512 + p[2]], table[768 + p[3]]};
                const auto weight =
                    T::set1(static_cast<float>(1U << (reorder_bits + filter_bits - bit - 1)));
                sum = T::add(sum, T::mul(T::load(values), weight));
            }
        }
        float result = T::reduce_add(sum);
        for (; b < plane_bytes; ++b) {
            for (uint32_t bit = 0; bit < filter_bits; ++bit) {
                const float weight =
                    static_cast<float>(1U << (reorder_bits + filter_bits - bit - 1));
                result += lookup[b * 256 + codes[0][bit * plane_bytes + b]] * weight;
            }
        }
        results[0] = result;
    } else {
        for (; b < plane_bytes; ++b) {
            const auto* table = lookup + b * 256;
            for (uint32_t bit = 0; bit < filter_bits; ++bit) {
                const auto offset = bit * plane_bytes + b;
                const float values[4]{table[codes[0][offset]],
                                      table[codes[1][offset]],
                                      table[codes[2][offset]],
                                      table[codes[3][offset]]};
                const auto weight =
                    T::set1(static_cast<float>(1U << (reorder_bits + filter_bits - bit - 1)));
                sum = T::add(sum, T::mul(T::load(values), weight));
            }
        }
        T::store(results, sum);
    }
}

template <typename T>
float
ComputeSmallPlanarIP(const float* query,
                     const uint8_t* high_plane,
                     const uint8_t* supplement,
                     uint64_t dim,
                     uint32_t supplement_bits) {
    static_assert(T::Width == 4);
    const uint64_t plane_bytes = (dim + 7) / 8;
    auto sum = T::zero();
    const float high_weight =
        high_plane == nullptr ? 0.0F : static_cast<float>(1U << supplement_bits);
    uint64_t d = 0;
    for (; d + 4 <= dim; d += 4) {
        const auto byte = d / 8;
        const auto shift = d & 7;
        auto code = T::zero();
        for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
            code = T::add(code,
                          T::SelectBits(supplement[bit * plane_bytes + byte] >> shift,
                                        T::set1(static_cast<float>(1U << bit))));
        }
        if (high_plane != nullptr) {
            code = T::add(code, T::SelectBits(high_plane[byte] >> shift, T::set1(high_weight)));
        }
        sum = T::fmadd(code, T::load(query + d), sum);
    }
    float result = T::reduce_add(sum);
    for (; d < dim; ++d) {
        const auto byte = d / 8;
        const auto mask = 1U << (d & 7);
        float code = high_plane != nullptr && (high_plane[byte] & mask) != 0 ? high_weight : 0.0F;
        for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
            if ((supplement[bit * plane_bytes + byte] & mask) != 0) {
                code += static_cast<float>(1U << bit);
            }
        }
        result += query[d] * code;
    }
    return result;
}

}  // namespace vsag::simd
