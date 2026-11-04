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

namespace vsag {

// Hamming distance kernels for bit-packed binary vectors.
//
// `dim` counts bits. A vector occupies ceil(dim / 8) bytes packed MSB-first: logical bit i is
// (x[i / 8] >> (7 - i % 8)) & 1, the layout of numpy.packbits with its default bitorder="big".
// When dim % 8 != 0 the low 8 - dim % 8 bits of the last byte are padding and are ignored.
// Every implementation returns exactly popcount(x XOR y) over the first `dim` bits, so all of
// them agree with generic::HammingDistance bit for bit. Pointers need no particular alignment.
//
// HammingDistanceBatch4 compares one query with four codes and writes results[0..3].
//
// HammingDistanceScan compares one query with `count` codes stored back to back (code i starts at
// codes + i * ceil(dim / 8)) and writes the distances to dists[0..count). A float represents
// every distance exactly while dim <= 2^24, the largest dimension an index accepts.
#define DECLARE_HAMMING_FUNCTIONS(ns)                                                            \
    namespace ns {                                                                               \
    uint64_t                                                                                     \
    HammingDistance(const uint8_t* x, const uint8_t* y, uint64_t dim);                           \
    void                                                                                         \
    HammingDistanceBatch4(const uint8_t* query,                                                  \
                          const uint8_t* codes1,                                                 \
                          const uint8_t* codes2,                                                 \
                          const uint8_t* codes3,                                                 \
                          const uint8_t* codes4,                                                 \
                          uint64_t dim,                                                          \
                          uint64_t* results);                                                    \
    void                                                                                         \
    HammingDistanceScan(                                                                         \
        const uint8_t* query, const uint8_t* codes, uint64_t dim, uint64_t count, float* dists); \
    }  // namespace ns

DECLARE_HAMMING_FUNCTIONS(generic)
DECLARE_HAMMING_FUNCTIONS(sse)
DECLARE_HAMMING_FUNCTIONS(avx2)
DECLARE_HAMMING_FUNCTIONS(avx512)
DECLARE_HAMMING_FUNCTIONS(avx512vpopcntdq)
DECLARE_HAMMING_FUNCTIONS(neon)
DECLARE_HAMMING_FUNCTIONS(sve)

#undef DECLARE_HAMMING_FUNCTIONS

using HammingDistanceType = uint64_t (*)(const uint8_t* x, const uint8_t* y, uint64_t dim);
using HammingDistanceBatch4Type = void (*)(const uint8_t* query,
                                           const uint8_t* codes1,
                                           const uint8_t* codes2,
                                           const uint8_t* codes3,
                                           const uint8_t* codes4,
                                           uint64_t dim,
                                           uint64_t* results);
using HammingDistanceScanType = void (*)(
    const uint8_t* query, const uint8_t* codes, uint64_t dim, uint64_t count, float* dists);

// Runtime-dispatched entry points: AVX-512 VPOPCNTDQ (with AVX-512BW), AVX-512BW, AVX2, POPCNT
// (built with the SSE4.2 unit), SVE, NEON, then the generic scalar fallback.
extern HammingDistanceType HammingDistance;
extern HammingDistanceBatch4Type HammingDistanceBatch4;
extern HammingDistanceScanType HammingDistanceScan;

}  // namespace vsag
