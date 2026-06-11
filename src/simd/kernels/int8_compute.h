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

// INT8 inner-product and L2-squared kernels.
//
// These are parameterized on Int8Traits<ISA>, which must expose:
//   Int8HalfVec   - the raw loaded int8 vector (half-width of full vector)
//   Int16Vec      - int8 widened to int16
//   Int32Vec      - int32 accumulator
//   BatchSize     - number of int8 elements per iteration
//   load_i8(ptr)          -> Int8HalfVec
//   cvt_i8_to_i16(v)     -> Int16Vec
//   madd_i16(a, b)        -> Int32Vec   (a[i]*b[i] + a[i+1]*b[i+1] pairwise)
//   sub_i16(a, b)         -> Int16Vec
//   add_i32(a, b)         -> Int32Vec
//   zero_i32()            -> Int32Vec
//   reduce_add_i32(v)     -> int32_t

#include <cstdint>

namespace vsag::simd {

using Int8Fallback = float (*)(const int8_t*, const int8_t*, uint64_t);

template <typename T>
inline float
Int8ComputeIPImpl(const int8_t* query,
                  const int8_t* codes,
                  uint64_t dim,
                  Int8Fallback fallback = nullptr) {
    using I32 = typename T::Int32Vec;
    constexpr int B = T::BatchSize;

    const uint64_t n = dim / B;
    if (n == 0) {
        if (fallback) {
            return fallback(query, codes, dim);
        }
        return 0.0f;
    }

    I32 sum = T::zero_i32();
    for (uint64_t i = 0; i < n; ++i) {
        auto q = T::cvt_i8_to_i16(T::load_i8(query + B * i));
        auto c = T::cvt_i8_to_i16(T::load_i8(codes + B * i));
        sum = T::add_i32(sum, T::madd_i16(q, c));
    }

    auto result = static_cast<float>(T::reduce_add_i32(sum));
    uint64_t tail = dim - B * n;
    if (tail > 0 && fallback) {
        result += fallback(query + B * n, codes + B * n, tail);
    }
    return result;
}

template <typename T>
inline float
Int8ComputeL2SqrImpl(const int8_t* query,
                     const int8_t* codes,
                     uint64_t dim,
                     Int8Fallback fallback = nullptr) {
    using I32 = typename T::Int32Vec;
    constexpr int B = T::BatchSize;

    const uint64_t n = dim / B;
    if (n == 0) {
        if (fallback) {
            return fallback(query, codes, dim);
        }
        return 0.0f;
    }

    I32 sum = T::zero_i32();
    for (uint64_t i = 0; i < n; ++i) {
        auto q = T::cvt_i8_to_i16(T::load_i8(query + B * i));
        auto c = T::cvt_i8_to_i16(T::load_i8(codes + B * i));
        auto diff = T::sub_i16(q, c);
        sum = T::add_i32(sum, T::madd_i16(diff, diff));
    }

    auto result = static_cast<float>(T::reduce_add_i32(sum));
    uint64_t tail = dim - B * n;
    if (tail > 0 && fallback) {
        result += fallback(query + B * n, codes + B * n, tail);
    }
    return result;
}

}  // namespace vsag::simd
