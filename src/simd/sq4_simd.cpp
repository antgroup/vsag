
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

#include "sq4_simd.h"

#include "simd_status.h"

namespace vsag {

static SQ4ComputeType
GetSQ4ComputeIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ4ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ4ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::SQ4ComputeIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ4ComputeIP;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::SQ4ComputeIP;
#endif
    }
    return generic::SQ4ComputeIP;
}
SQ4ComputeType SQ4ComputeIP = GetSQ4ComputeIP();

static SQ4ComputeType
GetSQ4ComputeL2Sqr() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ4ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ4ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::SQ4ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ4ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::SQ4ComputeL2Sqr;
#endif
    }
    return generic::SQ4ComputeL2Sqr;
}
SQ4ComputeType SQ4ComputeL2Sqr = GetSQ4ComputeL2Sqr();

static SQ4ComputeCodesType
GetSQ4ComputeCodesIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ4ComputeCodesIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ4ComputeCodesIP;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::SQ4ComputeCodesIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ4ComputeCodesIP;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::SQ4ComputeCodesIP;
#endif
    }
    return generic::SQ4ComputeCodesIP;
}
SQ4ComputeCodesType SQ4ComputeCodesIP = GetSQ4ComputeCodesIP();

static SQ4ComputeCodesType
GetSQ4ComputeCodesL2Sqr() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ4ComputeCodesL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ4ComputeCodesL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::SQ4ComputeCodesL2Sqr;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ4ComputeCodesL2Sqr;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::SQ4ComputeCodesL2Sqr;
#endif
    }
    return generic::SQ4ComputeCodesL2Sqr;
}
SQ4ComputeCodesType SQ4ComputeCodesL2Sqr = GetSQ4ComputeCodesL2Sqr();
}  // namespace vsag
