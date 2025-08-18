
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

#include "bit_simd.h"

#include "simd_status.h"

namespace vsag {

static BitOperatorType
GetBitAnd() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::BitAnd;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::BitAnd;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::BitAnd;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::BitAnd;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::BitAnd;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::BitAnd;
#endif
    }
    return generic::BitAnd;
}
BitOperatorType BitAnd = GetBitAnd();

static BitOperatorType
GetBitOr() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::BitOr;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::BitOr;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::BitOr;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::BitOr;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::BitOr;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::BitOr;
#endif
    }
    return generic::BitOr;
}
BitOperatorType BitOr = GetBitOr();

static BitOperatorType
GetBitXor() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::BitXor;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::BitXor;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::BitXor;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::BitXor;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::BitXor;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::BitXor;
#endif
    }
    return generic::BitXor;
}
BitOperatorType BitXor = GetBitXor();

static BitNotType
GetBitNot() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::BitNot;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::BitNot;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::BitNot;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::BitNot;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::BitNot;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::BitNot;
#endif
    }
    return generic::BitNot;
}
BitNotType BitNot = GetBitNot();

}  // namespace vsag
