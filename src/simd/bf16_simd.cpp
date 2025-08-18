
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

#include "bf16_simd.h"

#include "simd_status.h"

namespace vsag {

static BF16ComputeType
GetBF16ComputeIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::BF16ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::BF16ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::BF16ComputeIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::BF16ComputeIP;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::BF16ComputeIP;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::BF16ComputeIP;
#endif
    }
    return generic::BF16ComputeIP;
}
BF16ComputeType BF16ComputeIP = GetBF16ComputeIP();

static BF16ComputeType
GetBF16ComputeL2Sqr() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::BF16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::BF16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::BF16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::BF16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::BF16ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::BF16ComputeL2Sqr;
#endif
    }
    return generic::BF16ComputeL2Sqr;
}
BF16ComputeType BF16ComputeL2Sqr = GetBF16ComputeL2Sqr();
}  // namespace vsag
