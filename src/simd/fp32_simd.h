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
 * @file fp32_simd.h
 * @brief 32-bit floating point SIMD operations for vector computations.
 *
 * This header declares SIMD-optimized functions for 32-bit floating point
 * vector operations including inner product, L2 distance, batch computations,
 * and arithmetic operations. Multiple SIMD implementations are provided
 * (generic, SSE, AVX, AVX2, AVX512, NEON, SVE) with runtime selection.
 */

#pragma once

#include <cstdint>

#include "simd_marco.h"
namespace vsag {

/**
 * @brief Macro to declare FP32 SIMD functions for a specific namespace.
 *
 * This macro declares the following functions in the specified namespace:
 * - FP32ComputeIP: Inner product between two vectors
 * - FP32ComputeL2Sqr: L2 squared distance between two vectors
 * - FP32ComputeIPBatch4: Batch inner product (4 vectors at once)
 * - FP32ComputeL2SqrBatch4: Batch L2 squared distance (4 vectors at once)
 * - FP32Sub: Vector subtraction
 * - FP32Add: Vector addition
 * - FP32Mul: Vector multiplication
 * - FP32Div: Vector division
 * - FP32ReduceAdd: Sum all elements
 *
 * @param ns Namespace name for the SIMD implementation (e.g., sse, avx, neon)
 */
#define DECLARE_FP32_FUNCTIONS(ns)                                                            \
    namespace ns {                                                                            \
    float                                                                                     \
    FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);    \
    float                                                                                     \
    FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim); \
    void                                                                                      \
    FP32ComputeIPBatch4(const float* RESTRICT query,                                          \
                        uint64_t dim,                                                         \
                        const float* RESTRICT codes1,                                         \
                        const float* RESTRICT codes2,                                         \
                        const float* RESTRICT codes3,                                         \
                        const float* RESTRICT codes4,                                         \
                        float& result1,                                                       \
                        float& result2,                                                       \
                        float& result3,                                                       \
                        float& result4);                                                      \
    void                                                                                      \
    FP32ComputeL2SqrBatch4(const float* RESTRICT query,                                       \
                           uint64_t dim,                                                      \
                           const float* RESTRICT codes1,                                      \
                           const float* RESTRICT codes2,                                      \
                           const float* RESTRICT codes3,                                      \
                           const float* RESTRICT codes4,                                      \
                           float& result1,                                                    \
                           float& result2,                                                    \
                           float& result3,                                                    \
                           float& result4);                                                   \
    void                                                                                      \
    FP32Sub(const float* x, const float* y, float* z, uint64_t dim);                          \
    void                                                                                      \
    FP32Add(const float* x, const float* y, float* z, uint64_t dim);                          \
    void                                                                                      \
    FP32Mul(const float* x, const float* y, float* z, uint64_t dim);                          \
    void                                                                                      \
    FP32Div(const float* x, const float* y, float* z, uint64_t dim);                          \
    float                                                                                     \
    FP32ReduceAdd(const float* x, uint64_t dim);                                              \
    }  // namespace ns

/** @brief Generic (scalar) implementation namespace */
DECLARE_FP32_FUNCTIONS(generic)
/** @brief SSE SIMD implementation namespace */
DECLARE_FP32_FUNCTIONS(sse)
/** @brief AVX SIMD implementation namespace */
DECLARE_FP32_FUNCTIONS(avx)
/** @brief AVX2 SIMD implementation namespace */
DECLARE_FP32_FUNCTIONS(avx2)
/** @brief AVX512 SIMD implementation namespace */
DECLARE_FP32_FUNCTIONS(avx512)
/** @brief ARM NEON SIMD implementation namespace */
DECLARE_FP32_FUNCTIONS(neon)
/** @brief ARM SVE SIMD implementation namespace */
DECLARE_FP32_FUNCTIONS(sve)
#undef DECLARE_FP32_FUNCTIONS

/**
 * @brief Function pointer type for FP32 distance computation.
 *
 * @param query First vector (query vector).
 * @param codes Second vector (code/database vector).
 * @param dim Dimension of the vectors.
 * @return Computed distance or similarity value.
 */
using FP32ComputeType = float (*)(const float* RESTRICT query,
                                  const float* RESTRICT codes,
                                  uint64_t dim);

/** @brief Function pointer for FP32 inner product computation */
extern FP32ComputeType FP32ComputeIP;

/** @brief Function pointer for FP32 L2 squared distance computation */
extern FP32ComputeType FP32ComputeL2Sqr;

/**
 * @brief Function pointer type for batch FP32 distance computation (4 vectors).
 *
 * Computes distance/similarity between a query vector and 4 code vectors
 * simultaneously for improved cache utilization.
 *
 * @param query Query vector.
 * @param dim Dimension of the vectors.
 * @param codes1 First code vector.
 * @param codes2 Second code vector.
 * @param codes3 Third code vector.
 * @param codes4 Fourth code vector.
 * @param result1 Output for first distance result.
 * @param result2 Output for second distance result.
 * @param result3 Output for third distance result.
 * @param result4 Output for fourth distance result.
 */
using FP32ComputeBatch4Type = void (*)(const float* RESTRICT query,
                                       uint64_t dim,
                                       const float* RESTRICT codes1,
                                       const float* RESTRICT codes2,
                                       const float* RESTRICT codes3,
                                       const float* RESTRICT codes4,
                                       float& result1,
                                       float& result2,
                                       float& result3,
                                       float& result4);

/** @brief Function pointer for batch FP32 inner product computation */
extern FP32ComputeBatch4Type FP32ComputeIPBatch4;

/** @brief Function pointer for batch FP32 L2 squared distance computation */
extern FP32ComputeBatch4Type FP32ComputeL2SqrBatch4;

/**
 * @brief Function pointer type for FP32 element-wise arithmetic operations.
 *
 * @param x First input vector.
 * @param y Second input vector.
 * @param z Output vector for result.
 * @param dim Dimension of the vectors.
 */
using FP32ArithmeticType = void (*)(const float* x, const float* y, float* z, uint64_t dim);

/** @brief Function pointer for FP32 vector subtraction (z = x - y) */
extern FP32ArithmeticType FP32Sub;

/** @brief Function pointer for FP32 vector addition (z = x + y) */
extern FP32ArithmeticType FP32Add;

/** @brief Function pointer for FP32 vector multiplication (z = x * y) */
extern FP32ArithmeticType FP32Mul;

/** @brief Function pointer for FP32 vector division (z = x / y) */
extern FP32ArithmeticType FP32Div;

/**
 * @brief Function pointer type for FP32 reduction (sum all elements).
 *
 * @param x Input vector.
 * @param dim Dimension of the vector.
 * @return Sum of all elements in the vector.
 */
using FP32ReduceType = float (*)(const float* x, uint64_t dim);

/** @brief Function pointer for FP32 reduce-add (sum all elements) */
extern FP32ReduceType FP32ReduceAdd;

}  // namespace vsag