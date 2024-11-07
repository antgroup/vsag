
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

#include <cstdlib>

#include "simd_status.h"
namespace vsag {

SimdStatus
setup_simd();

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);
float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr);
float
INT8InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);
float
INT8InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr);

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result);

#if defined(ENABLE_SSE)
float
L2SqrSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
L2SqrSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
L2SqrSIMD4ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
L2SqrSIMD16ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

float
InnerProductSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductDistanceSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductDistanceSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductDistanceSIMD4ExtResidualsSSE(const void* pVect1v,
                                         const void* pVect2v,
                                         const void* qty_ptr);
float
InnerProductDistanceSIMD16ExtResidualsSSE(const void* pVect1v,
                                          const void* pVect2v,
                                          const void* qty_ptr);
void
PQDistanceSSEFloat256(const void* single_dim_centers, float single_dim_val, void* result);
#endif

#if defined(ENABLE_AVX)
float
L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductSIMD4ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
void
PQDistanceAVXFloat256(const void* single_dim_centers, float single_dim_val, void* result);
#endif

#if defined(ENABLE_AVX512)
float
L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
INT8InnerProduct256ResidualsAVX512Distance(const void* pVect1v,
                                           const void* pVect2v,
                                           const void* qty_ptr);
float
INT8InnerProduct512ResidualsAVX512Distance(const void* pVect1v,
                                           const void* pVect2v,
                                           const void* qty_ptr);
#endif

typedef float (*DistanceFunc)(const void* pVect1, const void* pVect2, const void* qty_ptr);
DistanceFunc
GetL2DistanceFunc(size_t dim);
DistanceFunc
GetInnerProductDistanceFunc(size_t dim);

DistanceFunc
GetINT8InnerProductDistanceFunc(size_t dim);

typedef void (*PQDistanceFunc)(const void* single_dim_centers, float single_dim_val, void* result);

PQDistanceFunc
GetPQDistanceFunc();

}  // namespace vsag
