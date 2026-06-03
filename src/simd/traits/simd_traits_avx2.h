
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

// AVX2 traits (Width = 8, __m256, with FMA). MUST only be included from
// a TU compiled with -mavx2 -mfma. Inherits SimdTraits<AVX_Tag> and
// overrides fmadd to use the real FMA instruction.

#include "simd_traits_avx.h"

namespace vsag::simd {

struct AVX2_Tag {};

template <>
struct SimdTraits<AVX2_Tag> : SimdTraits<AVX_Tag> {
    // Override: AVX2 + FMA gives us a single fused mul-add.
    static inline __attribute__((always_inline)) FloatVec
    fmadd(FloatVec a, FloatVec b, FloatVec c) {
        return _mm256_fmadd_ps(a, b, c);
    }
};

}  // namespace vsag::simd
