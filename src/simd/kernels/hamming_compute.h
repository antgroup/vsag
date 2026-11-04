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

// Shared scalar pieces of the Hamming distance kernels.
//
// A binary vector with `dim` bits occupies ceil(dim / 8) bytes packed MSB-first: logical bit i is
// (bytes[i / 8] >> (7 - i % 8)) & 1. When dim % 8 != 0 only the high dim % 8 bits of the last byte
// are part of the vector; every kernel masks the remaining padding bits so they never change a
// distance.
//
// The helpers are templates on an ISA tag declared inside each ISA translation unit. Every
// instantiation is therefore private to one translation unit and is compiled with that unit's
// target flags (for example, __builtin_popcountll becomes POPCNT only where POPCNT is enabled),
// which keeps code built for different ISAs from being merged by the linker.

#include <cstdint>
#include <cstring>

namespace vsag::simd {

/// Number of whole bytes of a `dim`-bit vector; the partial last byte, if any, is not included.
constexpr uint64_t
HammingFullBytes(uint64_t dim) {
    return dim / 8;
}

/// Mask of the valid bits of the partial last byte. Only meaningful when dim % 8 != 0.
constexpr uint32_t
HammingTailMask(uint64_t dim) {
    return (0xFF00U >> (dim % 8)) & 0xFFU;
}

/**
 * @brief Hamming distance of bytes [begin, ceil(dim / 8)), masking the partial last byte.
 *
 * SIMD kernels process whole vector registers first and finish the remainder with this helper.
 */
template <typename IsaTag>
inline uint64_t
HammingScalarRange(const uint8_t* x, const uint8_t* y, uint64_t begin, uint64_t dim) {
    const uint64_t full_bytes = HammingFullBytes(dim);
    uint64_t distance = 0;
    uint64_t i = begin;
    for (; i + sizeof(uint64_t) <= full_bytes; i += sizeof(uint64_t)) {
        uint64_t a = 0;
        uint64_t b = 0;
        std::memcpy(&a, x + i, sizeof(a));
        std::memcpy(&b, y + i, sizeof(b));
        distance += static_cast<uint64_t>(__builtin_popcountll(a ^ b));
    }
    for (; i < full_bytes; ++i) {
        distance += static_cast<uint64_t>(__builtin_popcount(static_cast<uint32_t>(x[i] ^ y[i])));
    }
    if (dim % 8 != 0) {
        const auto diff = static_cast<uint32_t>(x[full_bytes] ^ y[full_bytes]);
        distance += static_cast<uint64_t>(__builtin_popcount(diff & HammingTailMask(dim)));
    }
    return distance;
}

/**
 * @brief Four Hamming distances against one query over bytes [begin, ceil(dim / 8)).
 *
 * Adds the partial distances to results[0..3].
 */
template <typename IsaTag>
inline void
HammingScalarRangeBatch4(const uint8_t* query,
                         const uint8_t* codes1,
                         const uint8_t* codes2,
                         const uint8_t* codes3,
                         const uint8_t* codes4,
                         uint64_t begin,
                         uint64_t dim,
                         uint64_t* results) {
    const uint8_t* codes[4] = {codes1, codes2, codes3, codes4};
    const uint64_t full_bytes = HammingFullBytes(dim);
    uint64_t i = begin;
    for (; i + sizeof(uint64_t) <= full_bytes; i += sizeof(uint64_t)) {
        uint64_t q = 0;
        std::memcpy(&q, query + i, sizeof(q));
        for (uint64_t k = 0; k < 4; ++k) {
            uint64_t c = 0;
            std::memcpy(&c, codes[k] + i, sizeof(c));
            results[k] += static_cast<uint64_t>(__builtin_popcountll(q ^ c));
        }
    }
    for (; i < full_bytes; ++i) {
        for (uint64_t k = 0; k < 4; ++k) {
            results[k] += static_cast<uint64_t>(
                __builtin_popcount(static_cast<uint32_t>(query[i] ^ codes[k][i])));
        }
    }
    if (dim % 8 != 0) {
        const uint32_t mask = HammingTailMask(dim);
        for (uint64_t k = 0; k < 4; ++k) {
            const auto diff = static_cast<uint32_t>(query[full_bytes] ^ codes[k][full_bytes]);
            results[k] += static_cast<uint64_t>(__builtin_popcount(diff & mask));
        }
    }
}

/// A distance as a float. Distances never exceed dim < 2^63, so converting through int64_t gives
/// the same value as the unsigned conversion while compiling to a single instruction on x86-64.
inline float
HammingToFloat(uint64_t distance) {
    return static_cast<float>(static_cast<int64_t>(distance));
}

/**
 * @brief Distances from one query to `count` codes stored back to back, as floats.
 *
 * ISAs without a dedicated scan loop build it from their four-way and single-code kernels. Code i
 * starts at codes + i * ceil(dim / 8).
 */
template <typename IsaTag, typename Batch4Fn, typename SingleFn>
inline void
HammingScanByBatch4(const uint8_t* query,
                    const uint8_t* codes,
                    uint64_t dim,
                    uint64_t count,
                    float* dists,
                    Batch4Fn batch4,
                    SingleFn single) {
    const uint64_t code_size = (dim + 7) / 8;
    uint64_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const uint8_t* block = codes + i * code_size;
        uint64_t results[4];
        batch4(query,
               block,
               block + code_size,
               block + 2 * code_size,
               block + 3 * code_size,
               dim,
               results);
        dists[i] = HammingToFloat(results[0]);
        dists[i + 1] = HammingToFloat(results[1]);
        dists[i + 2] = HammingToFloat(results[2]);
        dists[i + 3] = HammingToFloat(results[3]);
    }
    for (; i < count; ++i) {
        dists[i] = HammingToFloat(single(query, codes + i * code_size, dim));
    }
}

}  // namespace vsag::simd
