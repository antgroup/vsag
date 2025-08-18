
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

#include "fp16_simd.h"

#include "simd_status.h"

namespace vsag {

static FP16ComputeType
GetFP16ComputeIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP16ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP16ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP16ComputeIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP16ComputeIP;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP16ComputeIP;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP16ComputeIP;
#endif
    }
    return generic::FP16ComputeIP;
}
FP16ComputeType FP16ComputeIP = GetFP16ComputeIP();

static FP16ComputeType
GetFP16ComputeL2Sqr() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP16ComputeL2Sqr;
#endif
    }
    return generic::FP16ComputeL2Sqr;
}
FP16ComputeType FP16ComputeL2Sqr = GetFP16ComputeL2Sqr();
}  // namespace vsag
