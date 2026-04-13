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
 * @file sq8_uniform_simd.h
 * @brief SIMD-accelerated uniform scalar quantization with 8-bit precision.
 *
 * This file provides distance computation functions for SQ8 uniform quantization
 * where all dimensions share the same quantization parameters, supporting multiple
 * SIMD instruction sets including SSE, AVX, AVX2, AVX512, NEON, and SVE.
 */

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

/**
 * @brief Macro to declare SQ8 uniform distance computation functions for a specific SIMD namespace.
 *
 * This macro expands to declare the SQ8UniformComputeCodesIP function for computing
 * inner product between two uniformly quantized SQ8 codes.
 *
 * @param ns The namespace name for the SIMD implementation.
 */
#define DECLARE_SQ8_UNIFORM_FUNCTIONS(ns)                    \
    namespace ns {                                           \
    float                                                    \
    SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1, \
                             const uint8_t* RESTRICT codes2, \
                             uint64_t dim);                  \
    }  // namespace ns

DECLARE_SQ8_UNIFORM_FUNCTIONS(generic)
DECLARE_SQ8_UNIFORM_FUNCTIONS(sse)
DECLARE_SQ8_UNIFORM_FUNCTIONS(avx)
DECLARE_SQ8_UNIFORM_FUNCTIONS(avx2)
DECLARE_SQ8_UNIFORM_FUNCTIONS(avx512)
DECLARE_SQ8_UNIFORM_FUNCTIONS(neon)
DECLARE_SQ8_UNIFORM_FUNCTIONS(sve)

#undef DECLARE_SQ8_UNIFORM_FUNCTIONS

/**
 * @brief Function pointer type for SQ8 uniform code distance computation.
 *
 * @param codes1 The first SQ8 uniform encoded codes.
 * @param codes2 The second SQ8 uniform encoded codes.
 * @param dim The dimension of the vectors.
 * @return The computed inner product.
 */
using SQ8UniformComputeCodesType = float (*)(const uint8_t* RESTRICT codes1,
                                             const uint8_t* RESTRICT codes2,
                                             uint64_t dim);

/**
 * @brief Function pointer for computing inner product between two SQ8 uniform codes.
 * @see SQ8UniformComputeCodesType
 */
extern SQ8UniformComputeCodesType SQ8UniformComputeCodesIP;

}  // namespace vsag