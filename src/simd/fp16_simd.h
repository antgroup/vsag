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
 * @file fp16_simd.h
 * @brief SIMD-accelerated FP16 (16-bit floating point) operations.
 *
 * This file provides distance computation functions for FP16 encoded vectors,
 * supporting multiple SIMD instruction sets including SSE, AVX, AVX2,
 * AVX512, NEON, and SVE. FP16 provides reduced precision with lower memory
 * footprint compared to FP32.
 */

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

/**
 * @brief Macro to declare FP16 distance computation functions for a specific SIMD namespace.
 *
 * This macro expands to declare the following functions:
 * - FP16ComputeIP: Inner product between two FP16 vectors
 * - FP16ComputeL2Sqr: Squared L2 distance between two FP16 vectors
 *
 * @param ns The namespace name for the SIMD implementation.
 */
#define DECLARE_FP16_FUNCTIONS(ns)                                                                \
    namespace ns {                                                                                \
    float                                                                                         \
    FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim);    \
    float                                                                                         \
    FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim); \
    }  // namespace ns

/**
 * @brief Generic namespace for FP16 conversion functions.
 */
namespace generic {

/**
 * @brief Convert FP16 value to FP32 (float).
 *
 * @param bf16_value The FP16 value stored as uint16_t.
 * @return The converted FP32 (float) value.
 */
float
FP16ToFloat(const uint16_t bf16_value);

/**
 * @brief Convert FP32 (float) value to FP16.
 *
 * @param fp32_value The FP32 (float) value to convert.
 * @return The converted FP16 value stored as uint16_t.
 */
uint16_t
FloatToFP16(const float fp32_value);

}  // namespace generic

DECLARE_FP16_FUNCTIONS(generic)
DECLARE_FP16_FUNCTIONS(sse)
DECLARE_FP16_FUNCTIONS(avx)
DECLARE_FP16_FUNCTIONS(avx2)
DECLARE_FP16_FUNCTIONS(avx512)
DECLARE_FP16_FUNCTIONS(neon)
DECLARE_FP16_FUNCTIONS(sve)

#undef DECLARE_FP16_FUNCTIONS

/**
 * @brief Function pointer type for FP16 distance computation.
 *
 * @param query The query vector in FP16 format.
 * @param codes The codes vector in FP16 format.
 * @param dim The dimension of the vectors.
 * @return The computed distance (IP or L2 squared).
 */
using FP16ComputeType = float (*)(const uint8_t* RESTRICT query,
                                  const uint8_t* RESTRICT codes,
                                  uint64_t dim);

/**
 * @brief Function pointer for computing inner product between two FP16 vectors.
 * @see FP16ComputeType
 */
extern FP16ComputeType FP16ComputeIP;

/**
 * @brief Function pointer for computing squared L2 distance between two FP16 vectors.
 * @see FP16ComputeType
 */
extern FP16ComputeType FP16ComputeL2Sqr;

}  // namespace vsag