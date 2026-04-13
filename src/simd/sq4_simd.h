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
 * @file sq4_simd.h
 * @brief SIMD-accelerated scalar quantization with 4-bit precision.
 *
 * This file provides distance computation functions for SQ4 (Scalar Quantization 4-bit)
 * encoded vectors, supporting multiple SIMD instruction sets including SSE, AVX, AVX2,
 * AVX512, NEON, and SVE.
 */

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

/**
 * @brief Macro to declare SQ4 distance computation functions for a specific SIMD namespace.
 *
 * This macro expands to declare the following functions:
 * - SQ4ComputeIP: Inner product between query and SQ4 codes
 * - SQ4ComputeL2Sqr: Squared L2 distance between query and SQ4 codes
 * - SQ4ComputeCodesIP: Inner product between two SQ4 codes
 * - SQ4ComputeCodesL2Sqr: Squared L2 distance between two SQ4 codes
 *
 * @param ns The namespace name for the SIMD implementation.
 */
#define DECLARE_SQ4_FUNCTIONS(ns)                           \
    namespace ns {                                          \
    float                                                   \
    SQ4ComputeIP(const float* RESTRICT query,               \
                 const uint8_t* RESTRICT codes,             \
                 const float* RESTRICT lower_bound,         \
                 const float* RESTRICT diff,                \
                 uint64_t dim);                             \
    float                                                   \
    SQ4ComputeL2Sqr(const float* RESTRICT query,            \
                    const uint8_t* RESTRICT codes,          \
                    const float* RESTRICT lower_bound,      \
                    const float* RESTRICT diff,             \
                    uint64_t dim);                          \
    float                                                   \
    SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,       \
                      const uint8_t* RESTRICT codes2,       \
                      const float* RESTRICT lower_bound,    \
                      const float* RESTRICT diff,           \
                      uint64_t dim);                        \
    float                                                   \
    SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,    \
                         const uint8_t* RESTRICT codes2,    \
                         const float* RESTRICT lower_bound, \
                         const float* RESTRICT diff,        \
                         uint64_t dim);                     \
    }  // namespace ns

DECLARE_SQ4_FUNCTIONS(generic)
DECLARE_SQ4_FUNCTIONS(sse)
DECLARE_SQ4_FUNCTIONS(avx)
DECLARE_SQ4_FUNCTIONS(avx2)
DECLARE_SQ4_FUNCTIONS(avx512)
DECLARE_SQ4_FUNCTIONS(neon)
DECLARE_SQ4_FUNCTIONS(sve)

#undef DECLARE_SQ4_FUNCTIONS

/**
 * @brief Function pointer type for SQ4 distance computation with a query vector.
 *
 * @param query The query vector (float32).
 * @param codes The SQ4 encoded codes.
 * @param lower_bound The lower bound values for dequantization.
 * @param diff The difference values for dequantization (upper_bound - lower_bound).
 * @param dim The dimension of the vectors.
 * @return The computed distance (IP or L2 squared).
 */
using SQ4ComputeType = float (*)(const float* RESTRICT query,
                                 const uint8_t* RESTRICT codes,
                                 const float* RESTRICT lower_bound,
                                 const float* RESTRICT diff,
                                 uint64_t dim);

/**
 * @brief Function pointer for computing inner product with SQ4 codes.
 * @see SQ4ComputeType
 */
extern SQ4ComputeType SQ4ComputeIP;

/**
 * @brief Function pointer for computing squared L2 distance with SQ4 codes.
 * @see SQ4ComputeType
 */
extern SQ4ComputeType SQ4ComputeL2Sqr;

/**
 * @brief Function pointer type for SQ4 distance computation between two code vectors.
 *
 * @param codes1 The first SQ4 encoded codes.
 * @param codes2 The second SQ4 encoded codes.
 * @param lower_bound The lower bound values for dequantization.
 * @param diff The difference values for dequantization.
 * @param dim The dimension of the vectors.
 * @return The computed distance (IP or L2 squared).
 */
using SQ4ComputeCodesType = float (*)(const uint8_t* RESTRICT codes1,
                                      const uint8_t* RESTRICT codes2,
                                      const float* RESTRICT lower_bound,
                                      const float* RESTRICT diff,
                                      uint64_t dim);

/**
 * @brief Function pointer for computing inner product between two SQ4 codes.
 * @see SQ4ComputeCodesType
 */
extern SQ4ComputeCodesType SQ4ComputeCodesIP;

/**
 * @brief Function pointer for computing squared L2 distance between two SQ4 codes.
 * @see SQ4ComputeCodesType
 */
extern SQ4ComputeCodesType SQ4ComputeCodesL2Sqr;

}  // namespace vsag