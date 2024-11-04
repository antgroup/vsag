
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

#include <cstdint>

namespace vsag {
namespace generic {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace generic

#if defined(ENABLE_SSE)
namespace sse {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace sse
#endif

#if defined(ENABLE_AVX2)
namespace avx2 {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace avx2
#endif

#if defined(ENABLE_AVX512)
namespace avx512 {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace avx512
#endif

inline void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_AVX512)
    avx512::DivScalar(from, to, dim, scalar);
#endif
#if defined(ENABLE_AVX2)
    avx2::DivScalar(from, to, dim, scalar);
#endif
#if defined(ENABLE_SSE)
    sse::DivScalar(from, to, dim, scalar);
#endif
    generic::DivScalar(from, to, dim, scalar);
}

inline float
Normalize(const float* from, float* to, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::Normalize(from, to, dim);
#endif
#if defined(ENABLE_AVX2)
    return avx2::Normalize(from, to, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::Normalize(from, to, dim);
#endif
    return generic::Normalize(from, to, dim);
}

}  // namespace vsag
