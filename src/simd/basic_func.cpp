
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

#include "simd_status.h"

namespace vsag {

static DistanceFuncType
GetL2Sqr() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::L2Sqr;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::L2Sqr;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::L2Sqr;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::L2Sqr;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::L2Sqr;
#endif
    }
    return generic::L2Sqr;
}
DistanceFuncType L2Sqr = GetL2Sqr();

static DistanceFuncType
GetInnerProduct() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::InnerProduct;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::InnerProduct;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::InnerProduct;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::InnerProduct;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::InnerProduct;
#endif
    }
    return generic::InnerProduct;
}
DistanceFuncType InnerProduct = GetInnerProduct();

static DistanceFuncType
GetInnerProductDistance() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::InnerProductDistance;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::InnerProductDistance;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::InnerProductDistance;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::InnerProductDistance;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::InnerProductDistance;
#endif
    }
    return generic::InnerProductDistance;
}
DistanceFuncType InnerProductDistance = GetInnerProductDistance();

static DistanceFuncType
GetINT8InnerProduct() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::INT8InnerProduct;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::INT8InnerProduct;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::INT8InnerProduct;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::INT8InnerProduct;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::INT8InnerProduct;
#endif
    }
    return generic::INT8InnerProduct;
}
DistanceFuncType INT8InnerProduct = GetINT8InnerProduct();

static DistanceFuncType
GetINT8InnerProductDistance() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::INT8InnerProductDistance;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::INT8InnerProductDistance;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::INT8InnerProductDistance;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::INT8InnerProductDistance;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::INT8InnerProductDistance;
#endif
    }
    return generic::INT8InnerProductDistance;
}
DistanceFuncType INT8InnerProductDistance = GetINT8InnerProductDistance();

static PQDistanceFuncType
GetPQDistanceFloat256() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::PQDistanceFloat256;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::PQDistanceFloat256;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::PQDistanceFloat256;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::PQDistanceFloat256;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::PQDistanceFloat256;
#endif
    }
    return generic::PQDistanceFloat256;
}
PQDistanceFuncType PQDistanceFloat256 = GetPQDistanceFloat256();

static PrefetchFuncType
GetPrefetch() {
    if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::Prefetch;
#endif
    }
    return generic::Prefetch;
}
PrefetchFuncType Prefetch = GetPrefetch();
}  // namespace vsag
