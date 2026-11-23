
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

// Four-lane backends consume an entire code byte per iteration. The shared
// query loads in the batch path never require an overread of the packed codes.
template <typename T, uint32_t N>
inline void
RaBitQBinaryIP4Lane(const float* query,
                    const uint8_t* const (&bits)[N],
                    uint64_t dim,
                    float scale,
                    float* results) {
    static_assert(T::Width == 4);
    // The legacy codec cutoff is a double literal, as in the generic path.
    const float positive = scale > 1e-3 ? scale : 1.0F;
    const float negative = scale > 1e-3 ? -scale : 0.0F;
    const auto pos = T::set1(positive);
    const auto neg = T::set1(negative);
    typename T::FloatVec sums[N];
    for (auto& sum : sums) {
        sum = T::zero();
    }

    uint64_t d = 0;
    for (; dim - d >= 8; d += 8) {
        const auto q0 = T::load(query + d);
        const auto q1 = T::load(query + d + 4);
        for (uint32_t k = 0; k < N; ++k) {
            const auto byte = bits[k][d / 8];
            sums[k] = T::fmadd(T::select_nibble(byte, pos, neg), q0, sums[k]);
            sums[k] = T::fmadd(T::select_nibble(byte >> 4, pos, neg), q1, sums[k]);
        }
    }
    for (uint32_t k = 0; k < N; ++k) {
        results[k] = T::reduce_add(sums[k]);
    }
    for (; d < dim; ++d) {
        for (uint32_t k = 0; k < N; ++k) {
            const bool set = (bits[k][d / 8] & (1U << (d % 8))) != 0;
            results[k] += query[d] * (set ? positive : negative);
        }
    }
}

// Popcount each query bit plane masked by each candidate. Accumulators are
// at least 32 bits per lane and preserve the packed result's modulo-2^32 sums.
template <typename T, uint32_t N, bool WithBaseSum>
inline void
RaBitQSQ4UBinary(const uint8_t* codes,
                 const uint8_t* const (&bases)[N],
                 uint64_t dim,
                 uint64_t* results) {
    constexpr auto W = T::ByteWidth;
    const uint64_t bytes = (dim + 7) / 8;
    typename T::CountVec inner_acc[N][4];
    typename T::CountVec base_acc[N];
    for (uint32_t k = 0; k < N; ++k) {
        base_acc[k] = T::zero_counts();
        for (auto& acc : inner_acc[k]) {
            acc = T::zero_counts();
        }
    }
    uint64_t i = 0;
    for (; bytes - i >= W; i += W) {
        const typename T::ByteVec planes[4] = {T::load_bytes(codes + i),
                                               T::load_bytes(codes + bytes + i),
                                               T::load_bytes(codes + 2 * bytes + i),
                                               T::load_bytes(codes + 3 * bytes + i)};
        for (uint32_t k = 0; k < N; ++k) {
            const auto base = T::load_bytes(bases[k] + i);
            if constexpr (WithBaseSum) {
                base_acc[k] = T::accumulate(base_acc[k], base);
            }
            for (uint32_t bit = 0; bit < 4; ++bit) {
                inner_acc[k][bit] = T::accumulate(inner_acc[k][bit], T::bit_and(base, planes[bit]));
            }
        }
    }
    for (uint32_t k = 0; k < N; ++k) {
        uint32_t inner = 0;
        uint32_t base_sum = 0;
        if constexpr (WithBaseSum) {
            base_sum = T::reduce_counts(base_acc[k]);
        }
        for (uint32_t bit = 0; bit < 4; ++bit) {
            inner += T::reduce_counts(inner_acc[k][bit]) << bit;
        }
        for (uint64_t tail = i; tail < bytes; ++tail) {
            const auto base = bases[k][tail];
            if constexpr (WithBaseSum) {
                base_sum += static_cast<uint32_t>(__builtin_popcount(base));
            }
            for (uint32_t bit = 0; bit < 4; ++bit) {
                inner += static_cast<uint32_t>(__builtin_popcount(codes[bit * bytes + tail] & base))
                         << bit;
            }
        }
        results[k] = (static_cast<uint64_t>(base_sum) << 32U) | inner;
    }
}

}  // namespace vsag::simd
