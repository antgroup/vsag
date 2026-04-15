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
 * @file simd_marco.h
 * @brief Platform-independent SIMD helper macros.
 *
 * This header provides portable macros for compiler-specific attributes
 * used in SIMD code, ensuring compatibility across different compilers
 * (GCC, Clang, MSVC).
 */

#pragma once

#if defined(__cplusplus)
#if defined(__GNUC__) || defined(__clang__)
/** @brief Pointer aliasing hint for optimization (GCC/Clang C++) */
#define RESTRICT __restrict__
#else
/** @brief Pointer aliasing hint for optimization (other C++ compilers) */
#define RESTRICT
#endif
#else
#if defined(__GNUC__) || defined(__clang__)
/** @brief Pointer aliasing hint for optimization (GCC/Clang C) */
#define RESTRICT restrict
#else
/** @brief Pointer aliasing hint for optimization (other C compilers) */
#define RESTRICT
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
/** @brief 32-byte alignment attribute for SIMD data (GCC/Clang) */
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
/** @brief 64-byte alignment attribute for SIMD data (GCC/Clang) */
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
/** @brief 32-byte alignment attribute for SIMD data (other compilers) */
#define PORTABLE_ALIGN32
/** @brief 64-byte alignment attribute for SIMD data (other compilers) */
#define PORTABLE_ALIGN64
#endif