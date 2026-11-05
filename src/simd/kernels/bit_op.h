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

// Bitwise operation kernels: AND, OR, XOR, NOT.
//
// Parameterized on BitTraits<ISA>, which must expose:
//   IntVec              - integer vector type (e.g. __m128i, __m256i, __m512i)
//   ByteWidth           - number of bytes per vector (16, 32, 64)
//   load(const uint8_t* p)     -> IntVec
//   store(uint8_t* p, IntVec v)
//   bit_and(IntVec a, IntVec b) -> IntVec
//   bit_or(IntVec a, IntVec b)  -> IntVec
//   bit_xor(IntVec a, IntVec b) -> IntVec
//   bit_not(IntVec a)            -> IntVec    (optional, only needed for BitNotImpl)

#include <cstdint>

namespace vsag::simd {

using BitOpFallback = void (*)(const uint8_t*, const uint8_t*, uint64_t, uint8_t*);
using BitNotFallback = void (*)(const uint8_t*, uint64_t, uint8_t*);

template <typename T>
inline void
BitAndImpl(const uint8_t* x,
           const uint8_t* y,
           uint64_t num_byte,
           uint8_t* result,
           BitOpFallback fallback = nullptr) {
    constexpr int W = T::ByteWidth;
    if (num_byte == 0)
        return;
    if (num_byte < static_cast<uint64_t>(W)) {
        return fallback(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + W <= static_cast<int64_t>(num_byte); i += W) {
        T::store(result + i, T::bit_and(T::load(x + i), T::load(y + i)));
    }
    if (i < static_cast<int64_t>(num_byte)) {
        fallback(x + i, y + i, num_byte - i, result + i);
    }
}

template <typename T>
inline void
BitOrImpl(const uint8_t* x,
          const uint8_t* y,
          uint64_t num_byte,
          uint8_t* result,
          BitOpFallback fallback = nullptr) {
    constexpr int W = T::ByteWidth;
    if (num_byte == 0)
        return;
    if (num_byte < static_cast<uint64_t>(W)) {
        return fallback(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + W <= static_cast<int64_t>(num_byte); i += W) {
        T::store(result + i, T::bit_or(T::load(x + i), T::load(y + i)));
    }
    if (i < static_cast<int64_t>(num_byte)) {
        fallback(x + i, y + i, num_byte - i, result + i);
    }
}

template <typename T>
inline void
BitXorImpl(const uint8_t* x,
           const uint8_t* y,
           uint64_t num_byte,
           uint8_t* result,
           BitOpFallback fallback = nullptr) {
    constexpr int W = T::ByteWidth;
    if (num_byte == 0)
        return;
    if (num_byte < static_cast<uint64_t>(W)) {
        return fallback(x, y, num_byte, result);
    }
    int64_t i = 0;
    for (; i + W <= static_cast<int64_t>(num_byte); i += W) {
        T::store(result + i, T::bit_xor(T::load(x + i), T::load(y + i)));
    }
    if (i < static_cast<int64_t>(num_byte)) {
        fallback(x + i, y + i, num_byte - i, result + i);
    }
}

template <typename T>
inline void
BitNotImpl(const uint8_t* x,
           uint64_t num_byte,
           uint8_t* result,
           BitNotFallback fallback = nullptr) {
    constexpr int W = T::ByteWidth;
    if (num_byte == 0)
        return;
    if (num_byte < static_cast<uint64_t>(W)) {
        return fallback(x, num_byte, result);
    }
    int64_t i = 0;
    for (; i + W <= static_cast<int64_t>(num_byte); i += W) {
        T::store(result + i, T::bit_not(T::load(x + i)));
    }
    if (i < static_cast<int64_t>(num_byte)) {
        fallback(x + i, num_byte - i, result + i);
    }
}

}  // namespace vsag::simd
