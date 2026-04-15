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
 * @file basic_func.h
 * @brief Basic distance computation function declarations for SIMD implementations.
 *
 * This header declares basic distance functions (L2, Inner Product) with
 * multiple SIMD implementations (generic, SSE, AVX, AVX2, AVX512, NEON, SVE).
 * The actual implementation is selected at runtime based on CPU capabilities.
 */

#pragma once

#include <cstdint>

namespace vsag {

/**
 * @brief Macro to declare basic distance functions for a specific SIMD namespace.
 *
 * This macro declares the following functions in the specified namespace:
 * - L2Sqr: L2 (Euclidean) squared distance
 * - InnerProduct: Inner product similarity
 * - InnerProductDistance: Inner product converted to distance
 * - INT8L2Sqr: L2 squared distance for int8 vectors
 * - INT8InnerProduct: Inner product for int8 vectors
 * - INT8InnerProductDistance: Inner product distance for int8 vectors
 * - PQDistanceFloat256: Product quantization distance
 * - Prefetch: Data prefetch hint
 *
 * @param ns Namespace name for the SIMD implementation (e.g., sse, avx, neon)
 */
#define DECLARE_BASIC_FUNCTIONS(ns)                                                         \
    namespace ns {                                                                          \
    float                                                                                   \
    L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);                   \
    float                                                                                   \
    InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);              \
    float                                                                                   \
    InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr);      \
    float                                                                                   \
    INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);               \
    float                                                                                   \
    INT8InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);          \
    float                                                                                   \
    INT8InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr);  \
    void                                                                                    \
    PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result); \
    void                                                                                    \
    Prefetch(const void* data);                                                             \
    }  // namespace ns

/** @brief Generic (scalar) implementation namespace */
DECLARE_BASIC_FUNCTIONS(generic)
/** @brief SSE SIMD implementation namespace */
DECLARE_BASIC_FUNCTIONS(sse)
/** @brief AVX SIMD implementation namespace */
DECLARE_BASIC_FUNCTIONS(avx)
/** @brief AVX2 SIMD implementation namespace */
DECLARE_BASIC_FUNCTIONS(avx2)
/** @brief AVX512 SIMD implementation namespace */
DECLARE_BASIC_FUNCTIONS(avx512)
/** @brief ARM NEON SIMD implementation namespace */
DECLARE_BASIC_FUNCTIONS(neon)
/** @brief ARM SVE SIMD implementation namespace */
DECLARE_BASIC_FUNCTIONS(sve)

#undef DECLARE_BASIC_FUNCTIONS

/**
 * @brief Function pointer type for distance computation.
 *
 * @param query1 First vector pointer.
 * @param query2 Second vector pointer.
 * @param qty_ptr Pointer to dimension count.
 * @return Computed distance value.
 */
using DistanceFuncType = float (*)(const void* query1, const void* query2, const void* qty_ptr);

/** @brief Function pointer for L2 squared distance computation */
extern DistanceFuncType L2Sqr;

/** @brief Function pointer for inner product computation */
extern DistanceFuncType InnerProduct;

/** @brief Function pointer for inner product distance computation */
extern DistanceFuncType InnerProductDistance;

/** @brief Function pointer for int8 L2 squared distance computation */
extern DistanceFuncType INT8L2Sqr;

/** @brief Function pointer for int8 inner product computation */
extern DistanceFuncType INT8InnerProduct;

/** @brief Function pointer for int8 inner product distance computation */
extern DistanceFuncType INT8InnerProductDistance;

/**
 * @brief Function pointer type for product quantization distance.
 *
 * @param single_dim_centers Centers for a single dimension.
 * @param single_dim_val Value to compare against centers.
 * @param result Output buffer for distance results.
 */
using PQDistanceFuncType = void (*)(const void* single_dim_centers,
                                    float single_dim_val,
                                    void* result);

/** @brief Function pointer for product quantization float-256 distance */
extern PQDistanceFuncType PQDistanceFloat256;

/**
 * @brief Function pointer type for data prefetch.
 *
 * @param data Pointer to data to prefetch into cache.
 */
using PrefetchFuncType = void (*)(const void* data);

/** @brief Function pointer for data prefetch hint */
extern PrefetchFuncType Prefetch;

}  // namespace vsag