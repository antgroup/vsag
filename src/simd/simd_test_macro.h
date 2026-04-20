
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

#pragma once

// Conditional wrapper macros for SIMD test code.
// These macros allow architecture-specific test blocks to be compiled only when the
// corresponding SIMD instruction set is available, preventing linker errors on platforms
// that don't support specific instruction sets (e.g., x86 instructions on ARM).
//
// Usage inside macro definitions (where #ifdef cannot be used directly):
//   SIMD_TEST_SSE(
//       if (SimdStatus::SupportSSE()) {
//           auto result = sse::SomeFunc(...);
//           REQUIRE(result == expected);
//       }
//   )

#ifdef ENABLE_SSE
#define SIMD_TEST_SSE(...) __VA_ARGS__
#else
#define SIMD_TEST_SSE(...)
#endif

#ifdef ENABLE_AVX
#define SIMD_TEST_AVX(...) __VA_ARGS__
#else
#define SIMD_TEST_AVX(...)
#endif

#ifdef ENABLE_AVX2
#define SIMD_TEST_AVX2(...) __VA_ARGS__
#else
#define SIMD_TEST_AVX2(...)
#endif

#ifdef ENABLE_AVX512
#define SIMD_TEST_AVX512(...) __VA_ARGS__
#else
#define SIMD_TEST_AVX512(...)
#endif

#ifdef ENABLE_AVX512VPOPCNTDQ
#define SIMD_TEST_AVX512VPOPCNTDQ(...) __VA_ARGS__
#else
#define SIMD_TEST_AVX512VPOPCNTDQ(...)
#endif

#ifdef ENABLE_NEON
#define SIMD_TEST_NEON(...) __VA_ARGS__
#else
#define SIMD_TEST_NEON(...)
#endif

#ifdef ENABLE_SVE
#define SIMD_TEST_SVE(...) __VA_ARGS__
#else
#define SIMD_TEST_SVE(...)
#endif
