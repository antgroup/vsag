
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

#include "simd_marco.h"
namespace vsag {

#define DECLARE_SQ8_FUNCTIONS(ns)                           \
    namespace ns {                                          \
    float                                                   \
    SQ8ComputeIP(const float* RESTRICT query,               \
                 const uint8_t* RESTRICT codes,             \
                 const float* RESTRICT lower_bound,         \
                 const float* RESTRICT diff,                \
                 uint64_t dim);                             \
    float                                                   \
    SQ8ComputeL2Sqr(const float* RESTRICT query,            \
                    const uint8_t* RESTRICT codes,          \
                    const float* RESTRICT lower_bound,      \
                    const float* RESTRICT diff,             \
                    uint64_t dim);                          \
    float                                                   \
    SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,       \
                      const uint8_t* RESTRICT codes2,       \
                      const float* RESTRICT lower_bound,    \
                      const float* RESTRICT diff,           \
                      uint64_t dim);                        \
    float                                                   \
    SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,    \
                         const uint8_t* RESTRICT codes2,    \
                         const float* RESTRICT lower_bound, \
                         const float* RESTRICT diff,        \
                         uint64_t dim);                     \
    void                                                    \
    SQ8SparseAccumulate(float* RESTRICT dists,              \
                        const uint16_t* RESTRICT ids,       \
                        const uint8_t* RESTRICT vals,       \
                        float query_val,                    \
                        uint32_t num);                      \
    }  // namespace ns

DECLARE_SQ8_FUNCTIONS(generic)
DECLARE_SQ8_FUNCTIONS(sse)
DECLARE_SQ8_FUNCTIONS(avx)
DECLARE_SQ8_FUNCTIONS(avx2)
DECLARE_SQ8_FUNCTIONS(avx512)
DECLARE_SQ8_FUNCTIONS(neon)
DECLARE_SQ8_FUNCTIONS(sve)

#undef DECLARE_SQ8_FUNCTIONS

using SQ8ComputeType = float (*)(const float* RESTRICT query,
                                 const uint8_t* RESTRICT codes,
                                 const float* RESTRICT lower_bound,
                                 const float* RESTRICT diff,
                                 uint64_t dim);
extern SQ8ComputeType SQ8ComputeIP;
extern SQ8ComputeType SQ8ComputeL2Sqr;

using SQ8ComputeCodesType = float (*)(const uint8_t* RESTRICT codes1,
                                      const uint8_t* RESTRICT codes2,
                                      const float* RESTRICT lower_bound,
                                      const float* RESTRICT diff,
                                      uint64_t dim);

extern SQ8ComputeCodesType SQ8ComputeCodesIP;
extern SQ8ComputeCodesType SQ8ComputeCodesL2Sqr;

using SQ8SparseAccumulateType = void (*)(float* RESTRICT dists,
                                         const uint16_t* RESTRICT ids,
                                         const uint8_t* RESTRICT vals,
                                         float query_val,
                                         uint32_t num);
extern SQ8SparseAccumulateType SQ8SparseAccumulate;
}  // namespace vsag
