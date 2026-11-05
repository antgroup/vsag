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

// BF16 distance kernels.
//
// BF16ComputeIP:     sum += bf16_to_fp32(query[i]) * bf16_to_fp32(codes[i])
// BF16ComputeL2Sqr:  sum += (bf16_to_fp32(query[i]) - bf16_to_fp32(codes[i]))^2
//
// Parameterized on BF16Traits<ISA>, which must expose:
//   FloatVec, Width, zero/fmadd/sub/reduce_add  (inherited from SimdTraits)
//   load_half(const uint16_t* p) -> FloatVec
//     Load Width bf16 values from p, convert to fp32.

namespace vsag::simd {

template <typename T>
inline float
HalfComputeIPImpl(const uint8_t* query,
                  const uint8_t* codes,
                  uint64_t dim,
                  float (*fallback)(const uint8_t*, const uint8_t*, uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;
    const auto* q16 = reinterpret_cast<const uint16_t*>(query);
    const auto* c16 = reinterpret_cast<const uint16_t*>(codes);

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback ? fallback(query, codes, dim) : 0.0f;
        }
    }

    V sum = T::zero();
    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V qv = T::load_half(q16 + i);
        V cv = T::load_half(c16 + i);
        sum = T::fmadd(qv, cv, sum);
    }
    float result = T::reduce_add(sum);
    if (i < dim && fallback) {
        result += fallback(query + i * 2, codes + i * 2, dim - i);
    }
    return result;
}

template <typename T>
inline float
HalfComputeL2SqrImpl(const uint8_t* query,
                     const uint8_t* codes,
                     uint64_t dim,
                     float (*fallback)(const uint8_t*, const uint8_t*, uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;
    const auto* q16 = reinterpret_cast<const uint16_t*>(query);
    const auto* c16 = reinterpret_cast<const uint16_t*>(codes);

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback ? fallback(query, codes, dim) : 0.0f;
        }
    }

    V sum = T::zero();
    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V qv = T::load_half(q16 + i);
        V cv = T::load_half(c16 + i);
        V d = T::sub(qv, cv);
        sum = T::fmadd(d, d, sum);
    }
    float result = T::reduce_add(sum);
    if (i < dim && fallback) {
        result += fallback(query + i * 2, codes + i * 2, dim - i);
    }
    return result;
}

}  // namespace vsag::simd
