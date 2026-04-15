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

/**
 * @file bf16_simd.h
 * @brief SIMD-accelerated BF16 (Brain Floating Point 16-bit) operations.
 *
 * This file provides distance computation functions for BF16 encoded vectors,
 * supporting multiple SIMD instruction sets including SSE, AVX, AVX2,
 * AVX512, NEON, and SVE. BF16 provides a good tradeoff between precision
 * and memory footprint for neural network computations.
 */

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

/**
 * @brief Macro to declare BF16 distance computation functions for a specific SIMD namespace.
 *
 * This macro expands to declare the following functions:
 * - BF16ComputeIP: Inner product between two BF16 vectors
 * - BF16ComputeL2Sqr: Squared L2 distance between two BF16 vectors
 *
 * @param ns The namespace name for the SIMD implementation.
 */
#define DECLARE_BF16_FUNCTIONS(ns)                                                                \
    namespace ns {                                                                                \
    float                                                                                         \
    BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim);    \
    float                                                                                         \
    BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim); \
    }  // namespace ns

/**
 * @brief Generic namespace for BF16 conversion functions.
 */
namespace generic {

/**
 * @brief Convert BF16 value to FP32 (float).
 *
 * @param bf16_value The BF16 value stored as uint16_t.
 * @return The converted FP32 (float) value.
 */
float
BF16ToFloat(const uint16_t bf16_value);

/**
 * @brief Convert FP32 (float) value to BF16.
 *
 * @param fp32_value The FP32 (float) value to convert.
 * @return The converted BF16 value stored as uint16_t.
 */
uint16_t
FloatToBF16(const float fp32_value);

}  // namespace generic

DECLARE_BF16_FUNCTIONS(generic)
DECLARE_BF16_FUNCTIONS(sse)
DECLARE_BF16_FUNCTIONS(avx)
DECLARE_BF16_FUNCTIONS(avx2)
DECLARE_BF16_FUNCTIONS(avx512)
DECLARE_BF16_FUNCTIONS(neon)
DECLARE_BF16_FUNCTIONS(sve)

#undef DECLARE_BF16_FUNCTIONS

/**
 * @brief Function pointer type for BF16 distance computation.
 *
 * @param query The query vector in BF16 format.
 * @param codes The codes vector in BF16 format.
 * @param dim The dimension of the vectors.
 * @return The computed distance (IP or L2 squared).
 */
using BF16ComputeType = float (*)(const uint8_t* RESTRICT query,
                                  const uint8_t* RESTRICT codes,
                                  uint64_t dim);

/**
 * @brief Function pointer for computing inner product between two BF16 vectors.
 * @see BF16ComputeType
 */
extern BF16ComputeType BF16ComputeIP;

/**
 * @brief Function pointer for computing squared L2 distance between two BF16 vectors.
 * @see BF16ComputeType
 */
extern BF16ComputeType BF16ComputeL2Sqr;

}  // namespace vsag