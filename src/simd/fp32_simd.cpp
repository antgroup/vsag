
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

#include "fp32_simd.h"

#include "simd_status.h"

namespace vsag {

static FP32ComputeType
GetFP32ComputeIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP32ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP32ComputeIP;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP32ComputeIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP32ComputeIP;
#endif
    }
    return generic::FP32ComputeIP;
}
FP32ComputeType FP32ComputeIP = GetFP32ComputeIP();

static FP32ComputeBatch4Type
GetFP32ComputeIPBatch4() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP32ComputeIPBatch4;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP32ComputeIPBatch4;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP32ComputeIPBatch4;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP32ComputeIPBatch4;
#endif
    }
    return generic::FP32ComputeIPBatch4;
}
FP32ComputeBatch4Type FP32ComputeIPBatch4 = GetFP32ComputeIPBatch4();

static FP32ComputeType
GetFP32ComputeL2Sqr() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP32ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP32ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP32ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP32ComputeL2Sqr;
#endif
    }
    return generic::FP32ComputeL2Sqr;
}
FP32ComputeType FP32ComputeL2Sqr = GetFP32ComputeL2Sqr();

static FP32ComputeBatch4Type
GetFP32ComputeL2SqrBatch4() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP32ComputeL2SqrBatch4;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP32ComputeL2SqrBatch4;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP32ComputeL2SqrBatch4;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP32ComputeL2SqrBatch4;
#endif
    }
    return generic::FP32ComputeL2SqrBatch4;
}
FP32ComputeBatch4Type FP32ComputeL2SqrBatch4 = GetFP32ComputeL2SqrBatch4();
}  // namespace vsag
