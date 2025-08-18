
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

#include "sq8_simd.h"

#include "simd_status.h"

namespace vsag {

static SQ8ComputeType
GetSQ8ComputeIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ8ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ8ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::SQ8ComputeIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ8ComputeIP;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::SQ8ComputeIP;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::SQ8ComputeIP;
#endif
    }
    return generic::SQ8ComputeIP;
}
SQ8ComputeType SQ8ComputeIP = GetSQ8ComputeIP();

static SQ8ComputeType
GetSQ8ComputeL2Sqr() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ8ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ8ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::SQ8ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ8ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::SQ8ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::SQ8ComputeL2Sqr;
#endif
    }
    return generic::SQ8ComputeL2Sqr;
}
SQ8ComputeType SQ8ComputeL2Sqr = GetSQ8ComputeL2Sqr();

static SQ8ComputeCodesType
GetSQ8ComputeCodesIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ8ComputeCodesIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ8ComputeCodesIP;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::SQ8ComputeCodesIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ8ComputeCodesIP;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::SQ8ComputeCodesIP;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::SQ8ComputeCodesIP;
#endif
    }
    return generic::SQ8ComputeCodesIP;
}
SQ8ComputeCodesType SQ8ComputeCodesIP = GetSQ8ComputeCodesIP();

static SQ8ComputeCodesType
GetSQ8ComputeCodesL2Sqr() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ8ComputeCodesL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ8ComputeCodesL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::SQ8ComputeCodesL2Sqr;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ8ComputeCodesL2Sqr;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::SQ8ComputeCodesL2Sqr;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::SQ8ComputeCodesL2Sqr;
#endif
    }
    return generic::SQ8ComputeCodesL2Sqr;
}
SQ8ComputeCodesType SQ8ComputeCodesL2Sqr = GetSQ8ComputeCodesL2Sqr();
}  // namespace vsag
