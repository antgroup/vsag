
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

#include "sq4_local_simd.h"

#include "simd_dispatch.h"
#include "sq4_local_shared_simd.h"
namespace vsag {
VSAG_DEFINE_SIMD_DISPATCH(SQ4LocalComputeL2Sqr, SQ4LocalComputeType);
VSAG_DEFINE_SIMD_DISPATCH(SQ4LocalComputeCodesL2Sqr, SQ4LocalComputeCodesType);
VSAG_DEFINE_SIMD_DISPATCH(SQ4LocalComputeIP, SQ4LocalComputeType);
VSAG_DEFINE_SIMD_DISPATCH(SQ4LocalComputeCodesIP, SQ4LocalComputeCodesType);

SQ4LocalSharedComputeType
GetSQ4LocalSharedCompute() {
    static const SQ4LocalSharedComputeType compute = []() -> SQ4LocalSharedComputeType {
#if defined(ENABLE_AVX512)
        if (SimdStatus::SupportAVX512()) {
            return avx512::SQ4LocalSharedCompute;
        }
#endif
#if defined(ENABLE_AVX2)
        if (SimdStatus::SupportAVX2()) {
            return avx2::SQ4LocalSharedCompute;
        }
#endif
        return nullptr;
    }();
    return compute;
}

SQ4LocalSharedComputeBatch4Type
GetSQ4LocalSharedComputeBatch4() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::SQ4LocalSharedComputeBatch4;
    }
#endif
    return nullptr;
}
}  // namespace vsag
