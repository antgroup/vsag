
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

// Scalar "pseudo-SIMD" traits for the generic backend.
//
// IMPORTANT (compile-strategy invariant):
//   - This header MUST only be included by ISA translation units whose
//     COMPILE_FLAGS are compatible with the intrinsics it exposes.
//   - The generic backend uses no intrinsics, so this header is safe in
//     any TU; the convention nevertheless holds for the AVX/NEON/SVE
//     traits headers that will be added in later phases.
//   - Cross-ISA inclusion of a higher-tier traits header from a lower-tier
//     TU (e.g. avx512 traits in sse.cpp) will trigger
//     "target specific option mismatch" errors. Do not break this rule.

namespace vsag::simd {

struct Generic_Tag {};

// Primary template; specializations live in sibling headers.
template <typename ISA>
struct SimdTraits;

template <>
struct SimdTraits<Generic_Tag> {
    using FloatVec = float;
    static constexpr int Width = 1;

    static inline __attribute__((always_inline)) FloatVec
    zero() {
        return 0.0f;
    }

    static inline __attribute__((always_inline)) FloatVec
    load(const float* p) {
        return *p;
    }

    static inline __attribute__((always_inline)) FloatVec
    mul(FloatVec a, FloatVec b) {
        return a * b;
    }

    static inline __attribute__((always_inline)) FloatVec
    add(FloatVec a, FloatVec b) {
        return a + b;
    }

    static inline __attribute__((always_inline)) FloatVec
    sub(FloatVec a, FloatVec b) {
        return a - b;
    }

    static inline __attribute__((always_inline)) FloatVec
    fmadd(FloatVec a, FloatVec b, FloatVec c) {
        return a * b + c;
    }

    static inline __attribute__((always_inline)) FloatVec
    div(FloatVec a, FloatVec b) {
        return a / b;
    }

    static inline __attribute__((always_inline)) void
    store(float* p, FloatVec v) {
        *p = v;
    }

    static inline __attribute__((always_inline)) float
    reduce_add(FloatVec v) {
        return v;
    }
};

}  // namespace vsag::simd
