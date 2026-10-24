
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
#include <type_traits>
#include <utility>
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

template <typename T, typename = void>
struct LocalSQ4Loader {
    static inline typename T::FloatVec
    Load(const uint8_t* codes, uint64_t first_coordinate) {
        uint8_t unpacked[T::Width];
        for (uint64_t j = 0; j < T::Width; ++j) {
            unpacked[j] = LocalSQCode<4>(codes, first_coordinate + j);
        }
        return T::load_u8_as_float(unpacked);
    }
};

template <typename T>
struct LocalSQ4Loader<T,
                      std::void_t<decltype(static_cast<void>(
                          T::load_nibbles_as_float(std::declval<const uint8_t*>())))>> {
    static inline typename T::FloatVec
    Load(const uint8_t* codes, uint64_t first_coordinate) {
        return T::load_nibbles_as_float(codes + first_coordinate / 2);
    }
};

template <typename T, int bit>
inline typename T::FloatVec
LocalSQLoad(const uint8_t* codes, uint64_t i) {
    if constexpr (bit == 8) {
        return T::load_u8_as_float(codes + i);
    } else {
        return LocalSQ4Loader<T>::Load(codes, i);
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

template <typename T, bool adc, bool ip>
inline float
LocalSQ8CoarseCompute(const float* query,
                      const uint8_t* first,
                      const uint8_t* second,
                      uint64_t dim) {
    float first_minimum = 0.0F;
    float first_step = 0.0F;
    if constexpr (!adc) {
        memcpy(&first_minimum, first, sizeof(float));
        memcpy(&first_step, first + sizeof(float), sizeof(float));
        first += 2 * sizeof(float);
        first_step *= 17.0F;
    }
    float second_minimum;
    float second_step;
    memcpy(&second_minimum, second, sizeof(float));
    memcpy(&second_step, second + sizeof(float), sizeof(float));
    second += 2 * sizeof(float);
    second_step *= 17.0F;
    auto sum = T::zero();
    uint64_t coordinate = 0;
    for (; coordinate + T::Width <= dim; coordinate += T::Width) {
        const auto second_codes = T::load_u8_as_coarse4_float(second + coordinate);
        const auto second_value =
            T::fmadd(second_codes, T::set1(second_step), T::set1(second_minimum));
        typename T::FloatVec first_value;
        if constexpr (adc) {
            first_value = T::load(query + coordinate);
        } else {
            const auto first_codes = T::load_u8_as_coarse4_float(first + coordinate);
            first_value = T::fmadd(first_codes, T::set1(first_step), T::set1(first_minimum));
        }
        if constexpr (ip) {
            sum = T::fmadd(first_value, second_value, sum);
        } else {
            const auto delta = T::sub(first_value, second_value);
            sum = T::fmadd(delta, delta, sum);
        }
    }
    float total = T::reduce_add(sum);
    for (; coordinate < dim; ++coordinate) {
        const float second_value = LocalSQDecode(
            static_cast<uint8_t>((second[coordinate] + 8) / 17), second_step, second_minimum);
        const float first_value =
            adc ? query[coordinate]
                : LocalSQDecode(static_cast<uint8_t>((first[coordinate] + 8) / 17),
                                first_step,
                                first_minimum);
        const float delta = first_value - second_value;
        total += ip ? first_value * second_value : delta * delta;
    }
    return ip ? 1.0F - total : total;
}

template <typename T, bool ip>
inline void
LocalSQ8CoarseComputeBatch4(const float* query,
                            const uint8_t* const* codes,
                            uint64_t dim,
                            float* distances) {
    float minima[4];
    float steps[4];
    for (uint64_t candidate = 0; candidate < 4; ++candidate) {
        memcpy(minima + candidate, codes[candidate], sizeof(float));
        memcpy(steps + candidate, codes[candidate] + sizeof(float), sizeof(float));
        steps[candidate] *= 17.0F;
    }
    const auto minimum0 = T::set1(minima[0]);
    const auto minimum1 = T::set1(minima[1]);
    const auto minimum2 = T::set1(minima[2]);
    const auto minimum3 = T::set1(minima[3]);
    const auto step0 = T::set1(steps[0]);
    const auto step1 = T::set1(steps[1]);
    const auto step2 = T::set1(steps[2]);
    const auto step3 = T::set1(steps[3]);
    auto sum0 = T::zero();
    auto sum1 = T::zero();
    auto sum2 = T::zero();
    auto sum3 = T::zero();
    uint64_t coordinate = 0;
    for (; coordinate + T::Width <= dim; coordinate += T::Width) {
        const auto query_values = T::load(query + coordinate);
        const auto accumulate = [&](const uint8_t* code,
                                    typename T::FloatVec minimum,
                                    typename T::FloatVec step,
                                    typename T::FloatVec& sum) {
            const auto coarse = T::load_u8_as_coarse4_float(code + 2 * sizeof(float) + coordinate);
            const auto value = T::fmadd(coarse, step, minimum);
            if constexpr (ip) {
                sum = T::fmadd(query_values, value, sum);
            } else {
                const auto delta = T::sub(query_values, value);
                sum = T::fmadd(delta, delta, sum);
            }
        };
        accumulate(codes[0], minimum0, step0, sum0);
        accumulate(codes[1], minimum1, step1, sum1);
        accumulate(codes[2], minimum2, step2, sum2);
        accumulate(codes[3], minimum3, step3, sum3);
    }
    distances[0] = T::reduce_add(sum0);
    distances[1] = T::reduce_add(sum1);
    distances[2] = T::reduce_add(sum2);
    distances[3] = T::reduce_add(sum3);
    for (; coordinate < dim; ++coordinate) {
        for (uint64_t candidate = 0; candidate < 4; ++candidate) {
            const auto coarse = static_cast<uint8_t>((codes[candidate][8 + coordinate] + 8) / 17);
            const float value = LocalSQDecode(coarse, steps[candidate], minima[candidate]);
            const float delta = query[coordinate] - value;
            distances[candidate] += ip ? query[coordinate] * value : delta * delta;
        }
    }
    if constexpr (ip) {
        for (uint64_t candidate = 0; candidate < 4; ++candidate) {
            distances[candidate] = 1.0F - distances[candidate];
        }
    }
}
}  // namespace vsag::simd
