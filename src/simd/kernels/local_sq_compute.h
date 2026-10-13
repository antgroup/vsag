
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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
namespace vsag::simd {
// Clamp rounding at the finite FP32 endpoints. Double intermediates also avoid
// overflow of code * step before adding a negative minimum.
inline float
LocalSQDecode(uint8_t code, float step, float minimum) {
    const double value = static_cast<double>(code) * step + minimum;
    const double limit = std::numeric_limits<float>::max();
    if (value > limit) {
        return std::numeric_limits<float>::max();
    }
    if (value < -limit) {
        return -std::numeric_limits<float>::max();
    }
    return std::fma(static_cast<float>(code), step, minimum);
}

// Packed SQ4 stores the even component in the low nibble.
template <int bit>
inline uint8_t
LocalSQCode(const uint8_t* codes, uint64_t i) {
    return (codes[i * bit / 8] >> (i * bit % 8)) & ((1U << bit) - 1);
}

template <typename T, int bit>
inline typename T::FloatVec
LocalSQLoad(const uint8_t* codes, uint64_t i) {
    if constexpr (bit == 8) {
        return T::load_u8_as_float(codes + i);
    } else {
        uint8_t unpacked[T::Width];
        for (uint64_t j = 0; j < T::Width; ++j) {
            unpacked[j] = LocalSQCode<bit>(codes, i + j);
        }
        return T::load_u8_as_float(unpacked);
    }
}

// Match DecodeOne's single-rounding reconstruction. SSE/AVX do not have
// native FMA: reconstruct their lanes scalarly before SIMD distance accumulation.
// The caller handles extreme metadata before reaching this normal-range helper.
template <typename T, int bit, bool fused>
inline typename T::FloatVec
LocalSQReconstruct(const uint8_t* codes, uint64_t i, float step, float minimum) {
    if constexpr (fused) {
        return T::fmadd(LocalSQLoad<T, bit>(codes, i), T::set1(step), T::set1(minimum));
    } else {
        float values[T::Width];
        for (uint64_t j = 0; j < T::Width; ++j) {
            values[j] = std::fma(static_cast<float>(LocalSQCode<bit>(codes, i + j)), step, minimum);
        }
        return T::load(values);
    }
}

// Both operands are reconstructed independently, including their offsets.
// memcpy permits odd dimensions and unaligned records; tails never overread codes.
template <typename T, bool adc, int bit, bool ip, bool fused = false>
inline float
LocalSQCompute(const float* query, const uint8_t* a, const uint8_t* b, uint64_t dim) {
    float min_a = 0;
    float step_a = 0;
    float min_b;
    float step_b;
    memcpy(&min_b, b, sizeof(float));
    memcpy(&step_b, b + sizeof(float), sizeof(float));
    b += 2 * sizeof(float);
    if constexpr (!adc) {
        memcpy(&min_a, a, sizeof(float));
        memcpy(&step_a, a + sizeof(float), sizeof(float));
        a += 2 * sizeof(float);
    }
    // Rare extreme records use the same finite reconstruction on every ISA.
    // The normal path retains vectorized FP32 arithmetic.
    const auto wide_compute = [&]() {
        double total = 0;
        for (uint64_t j = 0; j < dim; ++j) {
            const double va = adc ? query[j] : LocalSQDecode(LocalSQCode<bit>(a, j), step_a, min_a);
            const double vb = LocalSQDecode(LocalSQCode<bit>(b, j), step_b, min_b);
            const double delta = va - vb;
            total += ip ? va * vb : delta * delta;
        }
        if (total > std::numeric_limits<float>::max()) {
            return std::numeric_limits<float>::infinity();
        }
        if (total < -std::numeric_limits<float>::max()) {
            return -std::numeric_limits<float>::infinity();
        }
        return static_cast<float>(total);
    };
    if (std::abs(min_b) > 1e30F || step_b > 1e30F || std::abs(min_a) > 1e30F || step_a > 1e30F) {
        return wide_compute();
    }
    auto sum = T::zero();
    uint64_t i = 0;
    for (; i + T::Width <= dim; i += T::Width) {
        auto vb = LocalSQReconstruct<T, bit, fused>(b, i, step_b, min_b);
        typename T::FloatVec va;
        if constexpr (adc) {
            va = T::load(query + i);
        } else {
            va = LocalSQReconstruct<T, bit, fused>(a, i, step_a, min_a);
        }
        if constexpr (ip) {
            sum = T::fmadd(va, vb, sum);
        } else {
            auto delta = T::sub(va, vb);
            sum = T::fmadd(delta, delta, sum);
        }
    }
    float result = T::reduce_add(sum);
    for (; i < dim; ++i) {
        float va =
            adc ? query[i] : std::fma(static_cast<float>(LocalSQCode<bit>(a, i)), step_a, min_a);
        float vb = std::fma(static_cast<float>(LocalSQCode<bit>(b, i)), step_b, min_b);
        float delta = va - vb;
        result += ip ? va * vb : delta * delta;
    }
    // Recompute overflow/cancellation in double, also for extreme FP32 queries.
    // Inspect bits because release builds may assume finite arithmetic.
    uint32_t result_bits;
    memcpy(&result_bits, &result, sizeof(result_bits));
    return (result_bits & 0x7f800000U) == 0x7f800000U ? wide_compute() : result;
}
}  // namespace vsag::simd
