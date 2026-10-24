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

/**
 * Kernels for the rotated local scalar quantizer with split x + y codes.
 *
 * Codes are packed little-endian, `x` (filter) and `y` (supplement) bits per dimension.
 * Each kernel processes eight dimensions at a time: eight x-bit codes occupy exactly x bytes,
 * so a group of eight starts at a byte-aligned offset and can be loaded as one 64-bit word.
 *
 * The per-vector metadata (v_min, delta, |d|^2, sum(c)) is not part of these kernels; callers
 * add the affine offsets. All kernels return the raw code-domain inner product:
 *   FloatPackedIP:  sum_i q_i * ((hi_i << y) | lo_i)
 *   BytePackedIP:   same, with 8-bit quantized query codes
 *   PackedPackedIP: sum_i ((hi1_i << y) | lo1_i) * ((hi2_i << y) | lo2_i)
 *   FloatByteIP:    sum_i q_i * code_i   (one byte per dimension, optimized build layout)
 */
using SQFloatPackedIPType =
    float (*)(const float*, const uint8_t*, const uint8_t*, uint64_t, uint64_t, uint64_t);
using SQBytePackedIPType =
    float (*)(const uint8_t*, const uint8_t*, const uint8_t*, uint64_t, uint64_t, uint64_t);
using SQPackedPackedIPType = float (*)(
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, uint64_t, uint64_t, uint64_t);
using SQFloatByteIPType = float (*)(const float*, const uint8_t*, uint64_t);
using SQByteByteIPType = float (*)(const uint8_t*, const uint8_t*, uint64_t);

extern SQFloatPackedIPType SQFloatPackedIP;
extern SQBytePackedIPType SQBytePackedIP;
extern SQPackedPackedIPType SQPackedPackedIP;
extern SQFloatByteIPType SQFloatByteIP;
extern SQByteByteIPType SQByteByteIP;

namespace generic {
float
FloatPackedIP(
    const float* q, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y);
float
BytePackedIP(
    const uint8_t* qb, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y);
float
PackedPackedIP(const uint8_t* hi1,
               const uint8_t* lo1,
               const uint8_t* hi2,
               const uint8_t* lo2,
               uint64_t dim,
               uint64_t x,
               uint64_t y);
float
FloatByteIP(const float* q, const uint8_t* codes, uint64_t dim);
float
ByteByteIP(const uint8_t* a, const uint8_t* b, uint64_t dim);
}  // namespace generic

namespace avx512 {
float
FloatPackedIP(
    const float* q, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y);
float
BytePackedIP(
    const uint8_t* qb, const uint8_t* hi, const uint8_t* lo, uint64_t dim, uint64_t x, uint64_t y);
float
PackedPackedIP(const uint8_t* hi1,
               const uint8_t* lo1,
               const uint8_t* hi2,
               const uint8_t* lo2,
               uint64_t dim,
               uint64_t x,
               uint64_t y);
float
FloatByteIP(const float* q, const uint8_t* codes, uint64_t dim);
float
ByteByteIP(const uint8_t* a, const uint8_t* b, uint64_t dim);
}  // namespace avx512

}  // namespace vsag
