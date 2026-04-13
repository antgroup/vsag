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
 * @file int8_simd.h
 * @brief SIMD-accelerated INT8 (8-bit integer) vector operations.
 *
 * This file provides distance computation functions for INT8 encoded vectors,
 * supporting multiple SIMD instruction sets including SSE, AVX, AVX2,
 * AVX512, NEON, and SVE. INT8 quantization is commonly used for efficient
 * storage and computation in vector similarity search.
 */

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

/**
 * @brief Macro to declare INT8 distance computation functions for a specific SIMD namespace.
 *
 * This macro expands to declare the following functions:
 * - INT8ComputeIP: Inner product between two INT8 vectors
 * - INT8ComputeL2Sqr: Squared L2 distance between two INT8 vectors
 *
 * @param ns The namespace name for the SIMD implementation.
 */
#define DECLARE_INT8_FUNCTIONS(ns)                                                              \
    namespace ns {                                                                              \
    float                                                                                       \
    INT8ComputeIP(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim);    \
    float                                                                                       \
    INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim); \
    }  // namespace ns

DECLARE_INT8_FUNCTIONS(generic)
DECLARE_INT8_FUNCTIONS(sse)
DECLARE_INT8_FUNCTIONS(avx)
DECLARE_INT8_FUNCTIONS(avx2)
DECLARE_INT8_FUNCTIONS(avx512)
DECLARE_INT8_FUNCTIONS(neon)
DECLARE_INT8_FUNCTIONS(sve)

#undef DECLARE_INT8_FUNCTIONS

/**
 * @brief Function pointer type for INT8 distance computation.
 *
 * @param query The query vector in INT8 format.
 * @param codes The codes vector in INT8 format.
 * @param dim The dimension of the vectors.
 * @return The computed distance (IP or L2 squared).
 */
using INT8ComputeType = float (*)(const int8_t* RESTRICT query,
                                  const int8_t* RESTRICT codes,
                                  uint64_t dim);

/**
 * @brief Function pointer for computing squared L2 distance between two INT8 vectors.
 * @see INT8ComputeType
 */
extern INT8ComputeType INT8ComputeL2Sqr;

/**
 * @brief Function pointer for computing inner product between two INT8 vectors.
 * @see INT8ComputeType
 */
extern INT8ComputeType INT8ComputeIP;

}  // namespace vsag