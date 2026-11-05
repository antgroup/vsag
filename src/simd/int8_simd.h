
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

#define DECLARE_INT8_FUNCTIONS(ns)                                                              \
    namespace ns {                                                                              \
    float                                                                                       \
    INT8ComputeIP(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim);    \
    float                                                                                       \
    INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim); \
    }  // namespace ns

DECLARE_INT8_FUNCTIONS(generic)
DECLARE_INT8_FUNCTIONS(sse)
DECLARE_INT8_FUNCTIONS(avx)
DECLARE_INT8_FUNCTIONS(avx2)
DECLARE_INT8_FUNCTIONS(avx512)
DECLARE_INT8_FUNCTIONS(neon)
DECLARE_INT8_FUNCTIONS(sve)
#undef DECLARE_INT8_FUNCTIONS

using INT8ComputeType = float (*)(const int8_t* RESTRICT query,
                                  const int8_t* RESTRICT codes,
                                  uint64_t dim);
extern INT8ComputeType INT8ComputeL2Sqr;
extern INT8ComputeType INT8ComputeIP;
}  // namespace vsag
