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
 * @file bit_simd.h
 * @brief SIMD-accelerated bitwise operations for binary vectors.
 *
 * This file provides bitwise operations for binary vectors, supporting
 * multiple SIMD instruction sets including SSE, AVX, AVX2, AVX512,
 * NEON, and SVE. These operations are used for binary vector similarity
 * computations such as Hamming distance.
 */

#pragma once

#include <cstdint>

namespace vsag {

/**
 * @brief Macro to declare bitwise operation functions for a specific SIMD namespace.
 *
 * This macro expands to declare the following functions:
 * - BitAnd: Bitwise AND operation between two binary vectors
 * - BitOr: Bitwise OR operation between two binary vectors
 * - BitXor: Bitwise XOR operation between two binary vectors
 * - BitNot: Bitwise NOT operation on a binary vector
 *
 * @param ns The namespace name for the SIMD implementation.
 */
#define DECLARE_BIT_FUNCTIONS(ns)                                                         \
    namespace ns {                                                                        \
    void                                                                                  \
    BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result); \
    void                                                                                  \
    BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result);  \
    void                                                                                  \
    BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result); \
    void                                                                                  \
    BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result);                   \
    }  // namespace ns

DECLARE_BIT_FUNCTIONS(generic)
DECLARE_BIT_FUNCTIONS(sse)
DECLARE_BIT_FUNCTIONS(avx)
DECLARE_BIT_FUNCTIONS(avx2)
DECLARE_BIT_FUNCTIONS(avx512)
DECLARE_BIT_FUNCTIONS(neon)
DECLARE_BIT_FUNCTIONS(sve)

#undef DECLARE_BIT_FUNCTIONS

/**
 * @brief Function pointer type for binary bitwise operations (AND, OR, XOR).
 *
 * @param x The first binary vector.
 * @param y The second binary vector.
 * @param num_byte The number of bytes in the vectors.
 * @param result Output buffer for the result.
 */
using BitOperatorType = void (*)(const uint8_t* x,
                                 const uint8_t* y,
                                 const uint64_t num_byte,
                                 uint8_t* result);

/**
 * @brief Function pointer for bitwise AND operation.
 * @see BitOperatorType
 */
extern BitOperatorType BitAnd;

/**
 * @brief Function pointer for bitwise OR operation.
 * @see BitOperatorType
 */
extern BitOperatorType BitOr;

/**
 * @brief Function pointer for bitwise XOR operation.
 * @see BitOperatorType
 */
extern BitOperatorType BitXor;

/**
 * @brief Function pointer type for bitwise NOT operation.
 *
 * @param x The input binary vector.
 * @param num_byte The number of bytes in the vector.
 * @param result Output buffer for the result.
 */
using BitNotType = void (*)(const uint8_t* x, const uint64_t num_byte, uint8_t* result);

/**
 * @brief Function pointer for bitwise NOT operation.
 * @see BitNotType
 */
extern BitNotType BitNot;

}  // namespace vsag