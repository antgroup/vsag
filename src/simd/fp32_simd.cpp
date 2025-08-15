
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
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP32ComputeIP;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP32ComputeIP;
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
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP32ComputeIPBatch4;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP32ComputeIPBatch4;
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
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP32ComputeL2Sqr;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP32ComputeL2Sqr;
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
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP32ComputeL2SqrBatch4;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP32ComputeL2SqrBatch4;
#endif
    }
    return generic::FP32ComputeL2SqrBatch4;
}
FP32ComputeBatch4Type FP32ComputeL2SqrBatch4 = GetFP32ComputeL2SqrBatch4();

static FP32ArithmeticType
GetFP32Sub() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP32Sub;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP32Sub;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP32Sub;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP32Sub;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP32Sub;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP32Sub;
#endif
    }
    return generic::FP32Sub;
}
FP32ArithmeticType FP32Sub = GetFP32Sub();

static FP32ArithmeticType
GetFP32Add() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP32Add;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP32Add;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP32Add;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP32Add;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP32Add;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP32Add;
#endif
    }
    return generic::FP32Add;
}
FP32ArithmeticType FP32Add = GetFP32Add();

static FP32ArithmeticType
GetFP32Mul() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP32Mul;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP32Mul;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP32Mul;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP32Mul;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP32Mul;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP32Mul;
#endif
    }
    return generic::FP32Mul;
}
FP32ArithmeticType FP32Mul = GetFP32Mul();

static FP32ArithmeticType
GetFP32Div() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP32Div;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP32Div;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP32Div;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP32Div;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP32Div;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP32Div;
#endif
    }
    return generic::FP32Div;
}
FP32ArithmeticType FP32Div = GetFP32Div();

static FP32ReduceType
GetFP32ReduceAdd() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FP32ReduceAdd;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::FP32ReduceAdd;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::FP32ReduceAdd;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::FP32ReduceAdd;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FP32ReduceAdd;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FP32ReduceAdd;
#endif
    }
    return generic::FP32ReduceAdd;
}
FP32ReduceType FP32ReduceAdd = GetFP32ReduceAdd();

}  // namespace vsag
