
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
#define DECLARE_SQ4_LOCAL(ns)                                                     \
    namespace ns {                                                                \
    float                                                                         \
    SQ4LocalComputeL2Sqr(const float* query, const uint8_t* codes, uint64_t dim); \
    float                                                                         \
    SQ4LocalComputeCodesL2Sqr(const uint8_t* a, const uint8_t* b, uint64_t dim);  \
    float                                                                         \
    SQ4LocalComputeIP(const float* query, const uint8_t* codes, uint64_t dim);    \
    float                                                                         \
    SQ4LocalComputeCodesIP(const uint8_t* a, const uint8_t* b, uint64_t dim);     \
    }
DECLARE_SQ4_LOCAL(generic)
DECLARE_SQ4_LOCAL(sse)
DECLARE_SQ4_LOCAL(avx)
DECLARE_SQ4_LOCAL(avx2)
DECLARE_SQ4_LOCAL(avx512)
DECLARE_SQ4_LOCAL(neon)
DECLARE_SQ4_LOCAL(sve)
#undef DECLARE_SQ4_LOCAL
using SQ4LocalComputeType = float (*)(const float*, const uint8_t*, uint64_t);
using SQ4LocalComputeCodesType = float (*)(const uint8_t*, const uint8_t*, uint64_t);
extern SQ4LocalComputeType SQ4LocalComputeL2Sqr;
extern SQ4LocalComputeCodesType SQ4LocalComputeCodesL2Sqr;
extern SQ4LocalComputeType SQ4LocalComputeIP;
extern SQ4LocalComputeCodesType SQ4LocalComputeCodesIP;
}  // namespace vsag
