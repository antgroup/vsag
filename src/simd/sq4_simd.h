
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
#include <cpuinfo.h>
#include <x86intrin.h>

#include <vector>

namespace vsag {
namespace generic {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                const uint64_t dim);
float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  const uint64_t dim);
float
SQ4ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     const uint64_t dim);
}  // namespace generic

#if defined(ENABLE_SSE)
namespace sse {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                const uint64_t dim);
float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  const uint64_t dim);
float
SQ4ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     const uint64_t dim);
}  // namespace sse
#endif

#if defined(ENABLE_AVX2)
namespace avx2 {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                const uint64_t dim);
float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  const uint64_t dim);
float
SQ4ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     const uint64_t dim);
}  // namespace avx2
#endif

#if defined(ENABLE_AVX512)
namespace avx512 {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             const uint64_t dim);
float
SQ4ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                const uint64_t dim);
float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  const uint64_t dim);
float
SQ4ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     const uint64_t dim);
}  // namespace avx512
#endif

inline float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
#if defined(ENABLE_AVX2)
    return avx2::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
    return generic::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
}

inline float
SQ4ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
#if defined(ENABLE_AVX2)
    return avx2::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
    return generic::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
}

inline float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
#if defined(ENABLE_AVX2)
    return avx2::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
    return generic::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
}

inline float
SQ4ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
#if defined(ENABLE_AVX2)
    return avx2::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
    return generic::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
}

}  // namespace vsag
