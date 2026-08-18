
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
#include "pqfs_simd.h"

#include "simd_dispatch.h"

namespace vsag {

VSAG_DEFINE_SIMD_DISPATCH(PQFastScanLookUp32, PQFastScanLookUp32Type);

static PQFastScanLookUp32FloatType
GetPQFastScanLookUp32Float() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::PQFastScanLookUp32Float;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::PQFastScanLookUp32Float;
#endif
    }
    return generic::PQFastScanLookUp32Float;
}

PQFastScanLookUp32FloatType PQFastScanLookUp32Float = GetPQFastScanLookUp32Float();
}  // namespace vsag
