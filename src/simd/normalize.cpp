
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

#include "normalize.h"

#include "simd_dispatch.h"

namespace vsag {

VSAG_DEFINE_SIMD_DISPATCH(Normalize, NormalizeType);
VSAG_DEFINE_SIMD_DISPATCH(DivScalar, DivScalarType);

// NormalizeWithCentroid and InverseNormalizeWithCentroid currently only
// have generic implementations. Keep them as explicit one-liners so the
// lack of SIMD variants is obvious at the call site.
NormalizeWithCentroidType NormalizeWithCentroid = generic::NormalizeWithCentroid;
InverseNormalizeWithCentroidType InverseNormalizeWithCentroid =
    generic::InverseNormalizeWithCentroid;

}  // namespace vsag
