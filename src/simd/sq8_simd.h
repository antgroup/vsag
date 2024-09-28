
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
float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim);
float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lowerBound,
                const float* diff,
                uint64_t dim);
float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  uint64_t dim);
float
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lowerBound,
                     const float* diff,
                     uint64_t dim);
}  // namespace generic

#if defined(ENABLE_SSE)
namespace sse {
float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim);
float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lowerBound,
                const float* diff,
                uint64_t dim);
float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  uint64_t dim);
float
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lowerBound,
                     const float* diff,
                     uint64_t dim);
}  // namespace sse
#endif

#if defined(ENABLE_AVX2)
namespace avx2 {
float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim);
float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lowerBound,
                const float* diff,
                uint64_t dim);
float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  uint64_t dim);
float
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lowerBound,
                     const float* diff,
                     uint64_t dim);
}  // namespace avx2
#endif

#if defined(ENABLE_AVX512)
namespace avx512 {
float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim);
float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lowerBound,
                const float* diff,
                uint64_t dim);
float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  uint64_t dim);
float
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lowerBound,
                     const float* diff,
                     uint64_t dim);
}  // namespace avx512
#endif

inline float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
#endif
#if defined(ENABLE_AVX22)
    return AVX2::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
#endif
    return generic::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
}

inline float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lowerBound,
                const float* diff,
                uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
#endif
#if defined(ENABLE_AVX22)
    return AVX2::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
#endif
    return generic::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
}

inline float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
#endif
#if defined(ENABLE_AVX22)
    return AVX2::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
#endif
    return generic::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}

inline float
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lowerBound,
                     const float* diff,
                     uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::SQ8ComputeCodesL2Sqr(codes1, codes2, lowerBound, diff, dim);
#endif
#if defined(ENABLE_AVX22)
    return AVX2::SQ8ComputeCodesL2Sqr(codes1, codes2, lowerBound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::SQ8ComputeCodesL2Sqr(codes1, codes2, lowerBound, diff, dim);
#endif
    return generic::SQ8ComputeCodesL2Sqr(codes1, codes2, lowerBound, diff, dim);
}

}  // namespace vsag