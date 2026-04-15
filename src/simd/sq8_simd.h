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
 * @file sq8_simd.h
 * @brief SIMD-accelerated scalar quantization with 8-bit precision.
 *
 * This file provides distance computation functions for SQ8 (Scalar Quantization 8-bit)
 * encoded vectors, supporting multiple SIMD instruction sets including SSE, AVX, AVX2,
 * AVX512, NEON, and SVE.
 */

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

/**
 * @brief Macro to declare SQ8 distance computation functions for a specific SIMD namespace.
 *
 * This macro expands to declare the following functions:
 * - SQ8ComputeIP: Inner product between query and SQ8 codes
 * - SQ8ComputeL2Sqr: Squared L2 distance between query and SQ8 codes
 * - SQ8ComputeCodesIP: Inner product between two SQ8 codes
 * - SQ8ComputeCodesL2Sqr: Squared L2 distance between two SQ8 codes
 *
 * @param ns The namespace name for the SIMD implementation.
 */
#define DECLARE_SQ8_FUNCTIONS(ns)                           \
    namespace ns {                                          \
    float                                                   \
    SQ8ComputeIP(const float* RESTRICT query,               \
                 const uint8_t* RESTRICT codes,             \
                 const float* RESTRICT lower_bound,         \
                 const float* RESTRICT diff,                \
                 uint64_t dim);                             \
    float                                                   \
    SQ8ComputeL2Sqr(const float* RESTRICT query,            \
                    const uint8_t* RESTRICT codes,          \
                    const float* RESTRICT lower_bound,      \
                    const float* RESTRICT diff,             \
                    uint64_t dim);                          \
    float                                                   \
    SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,       \
                      const uint8_t* RESTRICT codes2,       \
                      const float* RESTRICT lower_bound,    \
                      const float* RESTRICT diff,           \
                      uint64_t dim);                        \
    float                                                   \
    SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,    \
                         const uint8_t* RESTRICT codes2,    \
                         const float* RESTRICT lower_bound, \
                         const float* RESTRICT diff,        \
                         uint64_t dim);                     \
    }  // namespace ns

DECLARE_SQ8_FUNCTIONS(generic)
DECLARE_SQ8_FUNCTIONS(sse)
DECLARE_SQ8_FUNCTIONS(avx)
DECLARE_SQ8_FUNCTIONS(avx2)
DECLARE_SQ8_FUNCTIONS(avx512)
DECLARE_SQ8_FUNCTIONS(neon)
DECLARE_SQ8_FUNCTIONS(sve)

#undef DECLARE_SQ8_FUNCTIONS

/**
 * @brief Function pointer type for SQ8 distance computation with a query vector.
 *
 * @param query The query vector (float32).
 * @param codes The SQ8 encoded codes.
 * @param lower_bound The lower bound values for dequantization.
 * @param diff The difference values for dequantization (upper_bound - lower_bound).
 * @param dim The dimension of the vectors.
 * @return The computed distance (IP or L2 squared).
 */
using SQ8ComputeType = float (*)(const float* RESTRICT query,
                                 const uint8_t* RESTRICT codes,
                                 const float* RESTRICT lower_bound,
                                 const float* RESTRICT diff,
                                 uint64_t dim);

/**
 * @brief Function pointer for computing inner product with SQ8 codes.
 * @see SQ8ComputeType
 */
extern SQ8ComputeType SQ8ComputeIP;

/**
 * @brief Function pointer for computing squared L2 distance with SQ8 codes.
 * @see SQ8ComputeType
 */
extern SQ8ComputeType SQ8ComputeL2Sqr;

/**
 * @brief Function pointer type for SQ8 distance computation between two code vectors.
 *
 * @param codes1 The first SQ8 encoded codes.
 * @param codes2 The second SQ8 encoded codes.
 * @param lower_bound The lower bound values for dequantization.
 * @param diff The difference values for dequantization.
 * @param dim The dimension of the vectors.
 * @return The computed distance (IP or L2 squared).
 */
using SQ8ComputeCodesType = float (*)(const uint8_t* RESTRICT codes1,
                                      const uint8_t* RESTRICT codes2,
                                      const float* RESTRICT lower_bound,
                                      const float* RESTRICT diff,
                                      uint64_t dim);

/**
 * @brief Function pointer for computing inner product between two SQ8 codes.
 * @see SQ8ComputeCodesType
 */
extern SQ8ComputeCodesType SQ8ComputeCodesIP;

/**
 * @brief Function pointer for computing squared L2 distance between two SQ8 codes.
 * @see SQ8ComputeCodesType
 */
extern SQ8ComputeCodesType SQ8ComputeCodesL2Sqr;

}  // namespace vsag