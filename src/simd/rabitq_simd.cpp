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

#include "rabitq_simd.h"

#include "simd_dispatch.h"

namespace vsag {

VSAG_DEFINE_SIMD_DISPATCH(RaBitQFloatBinaryIP, RaBitQFloatBinaryType);
VSAG_DEFINE_SIMD_DISPATCH(RaBitQFloatBinaryIPBatch4, RaBitQFloatBinaryBatch4Type);
VSAG_DEFINE_SIMD_DISPATCH(RaBitQFloatThreeBitIPBatch4, RaBitQFloatThreeBitBatch4Type);
VSAG_DEFINE_SIMD_DISPATCH(RaBitQFloatSplitCodeIP, RaBitQFloatSplitCodeType);
VSAG_DEFINE_SIMD_DISPATCH(RaBitQFloatSupplementCodeIP, RaBitQFloatSupplementCodeType);

static RaBitQFloatPackedSupplementCodeType
GetRaBitQFloatPackedSupplementCodeIP() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatPackedSupplementCodeIP;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatPackedSupplementCodeIP;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatPackedSupplementCodeIP;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatPackedSupplementCodeIP;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatPackedSupplementCodeIP;
    }
#endif
    return generic::RaBitQFloatPackedSupplementCodeIP;
}

RaBitQFloatPackedSupplementCodeType RaBitQFloatPackedSupplementCodeIP =
    GetRaBitQFloatPackedSupplementCodeIP();

static RaBitQFloatExCode7Type
GetRaBitQFloatExCode7IP() {
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatExCode7IP;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatExCode7IP;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatExCode7IP;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatExCode7IP;
    }
#endif
    return generic::RaBitQFloatExCode7IP;
}
RaBitQFloatExCode7Type RaBitQFloatExCode7IP = GetRaBitQFloatExCode7IP();
static RaBitQExCode7ToVectorType
GetRaBitQExCode7ToVector() {
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQExCode7ToVector;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQExCode7ToVector;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQExCode7ToVector;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQExCode7ToVector;
    }
#endif
    return generic::RaBitQExCode7ToVector;
}
RaBitQExCode7ToVectorType RaBitQExCode7ToVector = GetRaBitQExCode7ToVector();
static RaBitQExCode7ToBytesType
GetRaBitQExCode7ToBytes() {
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQExCode7ToBytes;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQExCode7ToBytes;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQExCode7ToBytes;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQExCode7ToBytes;
    }
#endif
    return generic::RaBitQExCode7ToBytes;
}
RaBitQExCode7ToBytesType RaBitQExCode7ToBytes = GetRaBitQExCode7ToBytes();
static RaBitQSQ4UBinaryType
GetRaBitQSQ4UBinaryIP() {
#if defined(ENABLE_AVX512VPOPCNTDQ)
    if (SimdStatus::SupportAVX512VPOPCNTDQ()) {
        return avx512vpopcntdq::RaBitQSQ4UBinaryIP;
    }
#endif
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQSQ4UBinaryIP;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQSQ4UBinaryIP;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQSQ4UBinaryIP;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQSQ4UBinaryIP;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQSQ4UBinaryIP;
    }
#endif
    return generic::RaBitQSQ4UBinaryIP;
}
RaBitQSQ4UBinaryType RaBitQSQ4UBinaryIP = GetRaBitQSQ4UBinaryIP();

static RaBitQSQ4UBinaryWithBaseSumType
GetRaBitQSQ4UBinaryIPWithBaseSum() {
#if defined(ENABLE_AVX512VPOPCNTDQ)
    if (SimdStatus::SupportAVX512VPOPCNTDQ()) {
        return avx512vpopcntdq::RaBitQSQ4UBinaryIPWithBaseSum;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQSQ4UBinaryIPWithBaseSum;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQSQ4UBinaryIPWithBaseSum;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQSQ4UBinaryIPWithBaseSum;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQSQ4UBinaryIPWithBaseSum;
    }
#endif
    return generic::RaBitQSQ4UBinaryIPWithBaseSum;
}
RaBitQSQ4UBinaryWithBaseSumType RaBitQSQ4UBinaryIPWithBaseSum = GetRaBitQSQ4UBinaryIPWithBaseSum();
static RaBitQSQ4UBinaryWithBaseSumBatch4Type
GetRaBitQSQ4UBinaryIPWithBaseSumBatch4() {
#if defined(ENABLE_AVX512VPOPCNTDQ)
    if (SimdStatus::SupportAVX512VPOPCNTDQ()) {
        return avx512vpopcntdq::RaBitQSQ4UBinaryIPWithBaseSumBatch4;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQSQ4UBinaryIPWithBaseSumBatch4;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQSQ4UBinaryIPWithBaseSumBatch4;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQSQ4UBinaryIPWithBaseSumBatch4;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQSQ4UBinaryIPWithBaseSumBatch4;
    }
#endif
    return generic::RaBitQSQ4UBinaryIPWithBaseSumBatch4;
}
RaBitQSQ4UBinaryWithBaseSumBatch4Type RaBitQSQ4UBinaryIPWithBaseSumBatch4 =
    GetRaBitQSQ4UBinaryIPWithBaseSumBatch4();

static RaBitQCodeCodeType
GetRaBitQCodeCodeIP() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQCodeCodeIP;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQCodeCodeIP;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQCodeCodeIP;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQCodeCodeIP;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQCodeCodeIP;
    }
#endif
    return generic::RaBitQCodeCodeIP;
}
RaBitQCodeCodeType RaBitQCodeCodeIP = GetRaBitQCodeCodeIP();
static RaBitQPackScalarToSplitPlanesType
GetRaBitQPackScalarToSplitPlanes() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQPackScalarToSplitPlanes;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQPackScalarToSplitPlanes;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQPackScalarToSplitPlanes;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQPackScalarToSplitPlanes;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQPackScalarToSplitPlanes;
    }
#endif
    return generic::RaBitQPackScalarToSplitPlanes;
}

RaBitQPackScalarToSplitPlanesType RaBitQPackScalarToSplitPlanes =
    GetRaBitQPackScalarToSplitPlanes();

static RaBitQPackScalarToSplitCodeType
GetRaBitQPackScalarToSplitCode() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQPackScalarToSplitCode;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQPackScalarToSplitCode;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQPackScalarToSplitCode;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQPackScalarToSplitCode;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQPackScalarToSplitCode;
    }
#endif
    return generic::RaBitQPackScalarToSplitCode;
}

RaBitQPackScalarToSplitCodeType RaBitQPackScalarToSplitCode = GetRaBitQPackScalarToSplitCode();

VSAG_DEFINE_SIMD_DISPATCH(FHTRotate, FHTRotateType);
VSAG_DEFINE_SIMD_DISPATCH(KacsWalk, KacsWalkType);
VSAG_DEFINE_SIMD_DISPATCH(VecRescale, VecRescaleType);
VSAG_DEFINE_SIMD_DISPATCH(RotateOp, RotateOpType);

static RaBitQFloatTwoBitCenteredType
GetRaBitQFloatTwoBitCenteredIP() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatTwoBitCenteredIP;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatTwoBitCenteredIP;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatTwoBitCenteredIP;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatTwoBitCenteredIP;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatTwoBitCenteredIP;
    }
#endif
    return generic::RaBitQFloatTwoBitCenteredIP;
}
RaBitQFloatTwoBitCenteredType RaBitQFloatTwoBitCenteredIP = GetRaBitQFloatTwoBitCenteredIP();

static RaBitQFloatTwoBitCenteredBatch4Type
GetRaBitQFloatTwoBitCenteredIPBatch4() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatTwoBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatTwoBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatTwoBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatTwoBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatTwoBitCenteredIPBatch4;
    }
#endif
    return generic::RaBitQFloatTwoBitCenteredIPBatch4;
}
RaBitQFloatTwoBitCenteredBatch4Type RaBitQFloatTwoBitCenteredIPBatch4 =
    GetRaBitQFloatTwoBitCenteredIPBatch4();

static RaBitQFloatThreeBitCenteredType
GetRaBitQFloatThreeBitCenteredIP() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatThreeBitCenteredIP;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatThreeBitCenteredIP;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatThreeBitCenteredIP;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatThreeBitCenteredIP;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatThreeBitCenteredIP;
    }
#endif
    return generic::RaBitQFloatThreeBitCenteredIP;
}
RaBitQFloatThreeBitCenteredType RaBitQFloatThreeBitCenteredIP = GetRaBitQFloatThreeBitCenteredIP();

static RaBitQFloatThreeBitCenteredBatch4Type
GetRaBitQFloatThreeBitCenteredIPBatch4() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatThreeBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatThreeBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatThreeBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatThreeBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatThreeBitCenteredIPBatch4;
    }
#endif
    return generic::RaBitQFloatThreeBitCenteredIPBatch4;
}
RaBitQFloatThreeBitCenteredBatch4Type RaBitQFloatThreeBitCenteredIPBatch4 =
    GetRaBitQFloatThreeBitCenteredIPBatch4();

static RaBitQFloatFourBitCenteredType
GetRaBitQFloatFourBitCenteredIP() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatFourBitCenteredIP;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatFourBitCenteredIP;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatFourBitCenteredIP;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatFourBitCenteredIP;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatFourBitCenteredIP;
    }
#endif
    return generic::RaBitQFloatFourBitCenteredIP;
}
RaBitQFloatFourBitCenteredType RaBitQFloatFourBitCenteredIP = GetRaBitQFloatFourBitCenteredIP();

static RaBitQFloatFourBitCenteredBatch4Type
GetRaBitQFloatFourBitCenteredIPBatch4() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatFourBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatFourBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatFourBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatFourBitCenteredIPBatch4;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatFourBitCenteredIPBatch4;
    }
#endif
    return generic::RaBitQFloatFourBitCenteredIPBatch4;
}
RaBitQFloatFourBitCenteredBatch4Type RaBitQFloatFourBitCenteredIPBatch4 =
    GetRaBitQFloatFourBitCenteredIPBatch4();

static RaBitQFloatThreeBitByLookupType
GetRaBitQFloatThreeBitIPByLookup() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatThreeBitIPByLookup;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatThreeBitIPByLookup;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatThreeBitIPByLookup;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatThreeBitIPByLookup;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatThreeBitIPByLookup;
    }
#endif
    return generic::RaBitQFloatThreeBitIPByLookup;
}
RaBitQFloatThreeBitByLookupType RaBitQFloatThreeBitIPByLookup = GetRaBitQFloatThreeBitIPByLookup();

static RaBitQFloatMultiBitByLookupType
GetRaBitQFloatMultiBitIPByLookup() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatMultiBitIPByLookup;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatMultiBitIPByLookup;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatMultiBitIPByLookup;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatMultiBitIPByLookup;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatMultiBitIPByLookup;
    }
#endif
    return generic::RaBitQFloatMultiBitIPByLookup;
}
RaBitQFloatMultiBitByLookupType RaBitQFloatMultiBitIPByLookup = GetRaBitQFloatMultiBitIPByLookup();

static RaBitQFloatThreeBitBatch4ByLookupType
GetRaBitQFloatThreeBitIPBatch4ByLookup() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatThreeBitIPBatch4ByLookup;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatThreeBitIPBatch4ByLookup;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatThreeBitIPBatch4ByLookup;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatThreeBitIPBatch4ByLookup;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatThreeBitIPBatch4ByLookup;
    }
#endif
    return generic::RaBitQFloatThreeBitIPBatch4ByLookup;
}
RaBitQFloatThreeBitBatch4ByLookupType RaBitQFloatThreeBitIPBatch4ByLookup =
    GetRaBitQFloatThreeBitIPBatch4ByLookup();

static RaBitQFloatMultiBitBatch4ByLookupType
GetRaBitQFloatMultiBitIPBatch4ByLookup() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatMultiBitIPBatch4ByLookup;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatMultiBitIPBatch4ByLookup;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatMultiBitIPBatch4ByLookup;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatMultiBitIPBatch4ByLookup;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatMultiBitIPBatch4ByLookup;
    }
#endif
    return generic::RaBitQFloatMultiBitIPBatch4ByLookup;
}
RaBitQFloatMultiBitBatch4ByLookupType RaBitQFloatMultiBitIPBatch4ByLookup =
    GetRaBitQFloatMultiBitIPBatch4ByLookup();

static RaBitQFloatSQType
GetRaBitQFloatSQIP() {
#if defined(ENABLE_AVX512)
    if (SimdStatus::SupportAVX512()) {
        return avx512::RaBitQFloatSQIP;
    }
#endif
#if defined(ENABLE_AVX2)
    if (SimdStatus::SupportAVX2()) {
        return avx2::RaBitQFloatSQIP;
    }
#endif
#if defined(ENABLE_SVE)
    if (SimdStatus::SupportSVE()) {
        return sve::RaBitQFloatSQIP;
    }
#endif
#if defined(ENABLE_NEON)
    if (SimdStatus::SupportNEON()) {
        return neon::RaBitQFloatSQIP;
    }
#endif
#if defined(ENABLE_SSE)
    if (SimdStatus::SupportSSE()) {
        return sse::RaBitQFloatSQIP;
    }
#endif
    return generic::RaBitQFloatSQIP;
}
RaBitQFloatSQType RaBitQFloatSQIP = GetRaBitQFloatSQIP();

VSAG_DEFINE_SIMD_DISPATCH(FlipSign, FlipSignType);

}  // namespace vsag
