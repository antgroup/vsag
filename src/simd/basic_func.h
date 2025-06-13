
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
void
Prefetch(const void* data);
}  // namespace generic

namespace sse {
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
void
Prefetch(const void* data);
}  // namespace sse

namespace avx {
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
}  // namespace avx

namespace avx2 {
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
}  // namespace avx2

namespace avx512 {
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
}  // namespace avx512

namespace neon {
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
void
Prefetch(const void* data);
}  // namespace neon

using DistanceFuncType = float (*)(const void* query1, const void* query2, const void* qty_ptr);
extern DistanceFuncType L2Sqr;
extern DistanceFuncType InnerProduct;
extern DistanceFuncType InnerProductDistance;
extern DistanceFuncType INT8InnerProduct;
extern DistanceFuncType INT8InnerProductDistance;

using PQDistanceFuncType = void (*)(const void* single_dim_centers,
                                    float single_dim_val,
                                    void* result);
extern PQDistanceFuncType PQDistanceFloat256;

using PrefetchFuncType = void (*)(const void* data);
extern PrefetchFuncType Prefetch;
}  // namespace vsag
