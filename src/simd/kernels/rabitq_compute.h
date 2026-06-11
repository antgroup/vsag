
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

// T must satisfy RaBitQTraits: inherits SimdTraits + provides:
//   static FloatVec bits_to_signed(const uint8_t* bit_ptr, FloatVec pos, FloatVec neg)
//   static FloatVec bits_select(const uint8_t* bit_ptr, FloatVec weight)
//
// For RaBitQFloatBinaryIP: load float vec, decode bits to ±inv_sqrt_d, fmadd accumulate.
template <typename T>
inline float
RaBitQFloatBinaryIPImpl(const float* vector,
                        const uint8_t* bits,
                        uint64_t dim,
                        float inv_sqrt_d,
                        float (*fallback)(const float*, const uint8_t*, uint64_t, float)) {
    if (dim == 0) {
        return 0.0f;
    }

    constexpr int W = T::Width;
    if (dim < static_cast<uint64_t>(W)) {
        return fallback(vector, bits, dim, inv_sqrt_d);
    }

    auto sum = T::zero();
    auto pos = inv_sqrt_d > 1e-3f ? T::set1(inv_sqrt_d) : T::set1(1.0f);
    auto neg = inv_sqrt_d > 1e-3f ? T::set1(-inv_sqrt_d) : T::zero();

    uint64_t d = 0;
    for (; d + W <= dim; d += W) {
        auto vec = T::load(vector + d);
        auto b_vec = T::bits_to_signed(bits + (d >> 3), pos, neg);
        sum = T::fmadd(b_vec, vec, sum);
    }

    float result = T::reduce_add(sum);

    if (d < dim) {
        result += fallback(vector + d, bits + (d >> 3), dim - d, inv_sqrt_d);
    }

    return result;
}

template <typename T>
inline void
RaBitQFloatBinaryIPBatch4Impl(const float* vector,
                              const uint8_t* bits1,
                              const uint8_t* bits2,
                              const uint8_t* bits3,
                              const uint8_t* bits4,
                              uint64_t dim,
                              float inv_sqrt_d,
                              float* results,
                              void (*fallback)(const float*,
                                               const uint8_t*,
                                               const uint8_t*,
                                               const uint8_t*,
                                               const uint8_t*,
                                               uint64_t,
                                               float,
                                               float*)) {
    if (dim == 0) {
        results[0] = results[1] = results[2] = results[3] = 0.0f;
        return;
    }

    constexpr int W = T::Width;
    if (dim < static_cast<uint64_t>(W)) {
        fallback(vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
        return;
    }

    auto pos = inv_sqrt_d > 1e-3f ? T::set1(inv_sqrt_d) : T::set1(1.0f);
    auto neg = inv_sqrt_d > 1e-3f ? T::set1(-inv_sqrt_d) : T::zero();
    typename T::FloatVec sums[4] = {T::zero(), T::zero(), T::zero(), T::zero()};
    const uint8_t* all_bits[4] = {bits1, bits2, bits3, bits4};

    uint64_t d = 0;
    for (; d + W <= dim; d += W) {
        auto vec = T::load(vector + d);
        uint64_t byte_id = d >> 3;
        for (uint32_t i = 0; i < 4; ++i) {
            auto binary = T::bits_to_signed(all_bits[i] + byte_id, pos, neg);
            sums[i] = T::fmadd(binary, vec, sums[i]);
        }
    }

    for (uint32_t i = 0; i < 4; ++i) {
        results[i] = T::reduce_add(sums[i]);
    }

    if (d < dim) {
        float tail[4];
        fallback(vector + d,
                 bits1 + (d >> 3),
                 bits2 + (d >> 3),
                 bits3 + (d >> 3),
                 bits4 + (d >> 3),
                 dim - d,
                 inv_sqrt_d,
                 tail);
        for (uint32_t i = 0; i < 4; ++i) {
            results[i] += tail[i];
        }
    }
}

template <typename T>
inline float
RaBitQFloatSplitCodeIPImpl(const float* vector,
                           const uint8_t* one_bit_code,
                           const uint8_t* supplement_code,
                           uint64_t dim,
                           uint32_t supplement_bits) {
    if (dim == 0) {
        return 0.0f;
    }

    constexpr int W = T::Width;
    const uint64_t plane_bytes = (dim + 7) / 8;
    auto sum = T::zero();

    uint64_t d = 0;
    for (; d + W <= dim; d += W) {
        const uint64_t byte_idx = d >> 3;
        auto code = T::zero();

        for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
            const auto* plane = supplement_code + static_cast<uint64_t>(bit) * plane_bytes;
            auto weight = T::set1(static_cast<float>(1U << bit));
            code = T::add(code, T::bits_select(plane + byte_idx, weight));
        }

        auto one_bit_weight = T::set1(static_cast<float>(1U << supplement_bits));
        code = T::add(code, T::bits_select(one_bit_code + byte_idx, one_bit_weight));

        auto vec = T::load(vector + d);
        sum = T::fmadd(code, vec, sum);
    }

    float result = T::reduce_add(sum);

    const uint32_t one_bit_scalar_weight = 1U << supplement_bits;
    for (; d < dim; ++d) {
        const uint64_t byte_idx = d >> 3;
        const uint8_t bit_mask = static_cast<uint8_t>(1U << (d & 7));
        uint32_t code = (one_bit_code[byte_idx] & bit_mask) != 0 ? one_bit_scalar_weight : 0U;
        for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
            const auto* plane = supplement_code + static_cast<uint64_t>(bit) * plane_bytes;
            if ((plane[byte_idx] & bit_mask) != 0) {
                code += 1U << bit;
            }
        }
        result += vector[d] * static_cast<float>(code);
    }

    return result;
}

}  // namespace vsag::simd
