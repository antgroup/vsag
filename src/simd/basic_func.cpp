
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

#include "basic_func.h"

#include "simd_dispatch.h"

namespace vsag {

VSAG_DEFINE_SIMD_DISPATCH(L2Sqr, DistanceFuncType);
VSAG_DEFINE_SIMD_DISPATCH(InnerProduct, DistanceFuncType);
VSAG_DEFINE_SIMD_DISPATCH(InnerProductDistance, DistanceFuncType);
VSAG_DEFINE_SIMD_DISPATCH(INT8InnerProduct, DistanceFuncType);
VSAG_DEFINE_SIMD_DISPATCH(INT8L2Sqr, DistanceFuncType);
VSAG_DEFINE_SIMD_DISPATCH(INT8InnerProductDistance, DistanceFuncType);
VSAG_DEFINE_SIMD_DISPATCH(PQDistanceFloat256, PQDistanceFuncType);
VSAG_DEFINE_SIMD_DISPATCH_PREFETCH(Prefetch, PrefetchFuncType);

}  // namespace vsag
