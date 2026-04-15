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
 * @file pqfs_simd.h
 * @brief SIMD-accelerated Product Quantization Fast Scan operations.
 *
 * This file provides functions for Product Quantization Fast Scan (PQFS)
 * table lookup operations, supporting multiple SIMD instruction sets
 * including SSE, AVX, AVX2, AVX512, NEON, and SVE.
 */

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

/**
 * @brief Macro to declare PQ Fast Scan functions for a specific SIMD namespace.
 *
 * This macro expands to declare the PQFastScanLookUp32 function for
 * performing fast table lookup on 32 code vectors simultaneously.
 *
 * @param ns The namespace name for the SIMD implementation.
 */
#define DECLARE_PQFS_FUNCTIONS(ns)                           \
    namespace ns {                                           \
    void                                                     \
    PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table, \
                       const uint8_t* RESTRICT codes,        \
                       uint64_t pq_dim,                      \
                       int32_t* RESTRICT result);            \
    }  // namespace ns

DECLARE_PQFS_FUNCTIONS(generic)
DECLARE_PQFS_FUNCTIONS(sse)
DECLARE_PQFS_FUNCTIONS(avx)
DECLARE_PQFS_FUNCTIONS(avx2)
DECLARE_PQFS_FUNCTIONS(avx512)
DECLARE_PQFS_FUNCTIONS(neon)
DECLARE_PQFS_FUNCTIONS(sve)

#undef DECLARE_PQFS_FUNCTIONS

/**
 * @brief Function pointer type for PQ Fast Scan table lookup.
 *
 * Performs simultaneous distance table lookup for 32 PQ encoded vectors,
 * producing 32 distance results at once for efficient batch processing.
 *
 * @param lookup_table The precomputed distance lookup table.
 * @param codes The PQ encoded codes for 32 vectors.
 * @param pq_dim The number of PQ subspaces.
 * @param result Output array for 32 computed distances.
 */
using PQFastScanLookUp32Type = void (*)(const uint8_t* RESTRICT lookup_table,
                                        const uint8_t* RESTRICT codes,
                                        uint64_t pq_dim,
                                        int32_t* RESTRICT result);

/**
 * @brief Function pointer for PQ Fast Scan table lookup (32 vectors).
 * @see PQFastScanLookUp32Type
 */
extern PQFastScanLookUp32Type PQFastScanLookUp32;

}  // namespace vsag