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
 * @file normalize.h
 * @brief Vector normalization function declarations for SIMD implementations.
 *
 * This header declares vector normalization and scaling functions with
 * multiple SIMD implementations (generic, SSE, AVX, AVX2, AVX512, NEON, SVE).
 * These operations are commonly used in vector similarity search preprocessing.
 */

#pragma once

#include <cstdint>

namespace vsag {

/**
 * @brief Macro to declare normalize functions for a specific SIMD namespace.
 *
 * This macro declares the following functions in the specified namespace:
 * - DivScalar: Divide each vector element by a scalar
 * - Normalize: Normalize a vector to unit length
 *
 * @param ns Namespace name for the SIMD implementation (e.g., sse, avx, neon)
 */
#define DECLARE_NORMALIZE_FUNCTIONS(ns)                                  \
    namespace ns {                                                       \
    void                                                                 \
    DivScalar(const float* from, float* to, uint64_t dim, float scalar); \
    float                                                                \
    Normalize(const float* from, float* to, uint64_t dim);               \
    }  // namespace ns

namespace generic {
/**
 * @brief Normalize a vector by subtracting a centroid and computing unit vector.
 *
 * @param from Input vector data.
 * @param centroid Centroid vector to subtract.
 * @param to Output buffer for normalized vector.
 * @param dim Dimension of the vectors.
 * @return The L2 norm before normalization.
 */
float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim);

/**
 * @brief Inverse normalization by adding centroid and scaling.
 *
 * @param from Input normalized vector data.
 * @param centroid Centroid vector to add back.
 * @param to Output buffer for denormalized vector.
 * @param dim Dimension of the vectors.
 * @param norm The original norm to scale by.
 */
void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm);
}  // namespace generic

/** @brief Generic (scalar) implementation namespace */
DECLARE_NORMALIZE_FUNCTIONS(generic)
/** @brief SSE SIMD implementation namespace */
DECLARE_NORMALIZE_FUNCTIONS(sse)
/** @brief AVX SIMD implementation namespace */
DECLARE_NORMALIZE_FUNCTIONS(avx)
/** @brief AVX2 SIMD implementation namespace */
DECLARE_NORMALIZE_FUNCTIONS(avx2)
/** @brief AVX512 SIMD implementation namespace */
DECLARE_NORMALIZE_FUNCTIONS(avx512)
/** @brief ARM NEON SIMD implementation namespace */
DECLARE_NORMALIZE_FUNCTIONS(neon)
/** @brief ARM SVE SIMD implementation namespace */
DECLARE_NORMALIZE_FUNCTIONS(sve)

#undef DECLARE_NORMALIZE_FUNCTIONS

/**
 * @brief Function pointer type for vector normalization.
 *
 * @param from Input vector data.
 * @param to Output buffer for normalized vector.
 * @param dim Dimension of the vectors.
 * @return The L2 norm before normalization.
 */
using NormalizeType = float (*)(const float* from, float* to, uint64_t dim);

/** @brief Function pointer for vector normalization */
extern NormalizeType Normalize;

/**
 * @brief Function pointer type for centroid-based normalization.
 *
 * @param from Input vector data.
 * @param centroid Centroid vector to subtract.
 * @param to Output buffer for normalized vector.
 * @param dim Dimension of the vectors.
 * @return The L2 norm before normalization.
 */
using NormalizeWithCentroidType = float (*)(const float* from,
                                            const float* centroid,
                                            float* to,
                                            uint64_t dim);

/** @brief Function pointer for centroid-based normalization */
extern NormalizeWithCentroidType NormalizeWithCentroid;

/**
 * @brief Function pointer type for inverse centroid-based normalization.
 *
 * @param from Input normalized vector data.
 * @param centroid Centroid vector to add back.
 * @param to Output buffer for denormalized vector.
 * @param dim Dimension of the vectors.
 * @param norm The original norm to scale by.
 */
using InverseNormalizeWithCentroidType =
    void (*)(const float* from, const float* centroid, float* to, uint64_t dim, float norm);

/** @brief Function pointer for inverse centroid-based normalization */
extern InverseNormalizeWithCentroidType InverseNormalizeWithCentroid;

/**
 * @brief Function pointer type for scalar division.
 *
 * @param from Input vector data.
 * @param to Output buffer for result vector.
 * @param dim Dimension of the vectors.
 * @param scalar Scalar value to divide by.
 */
using DivScalarType = void (*)(const float* from, float* to, uint64_t dim, float scalar);

/** @brief Function pointer for scalar division */
extern DivScalarType DivScalar;

}  // namespace vsag