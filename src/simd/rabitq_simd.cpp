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
static RaBitQFloatExCode7Type
GetRaBitQFloatExCode7IP() {
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatExCode7IP;
#endif
    }
    return generic::RaBitQFloatExCode7IP;
}
RaBitQFloatExCode7Type RaBitQFloatExCode7IP = GetRaBitQFloatExCode7IP();
VSAG_DEFINE_SIMD_DISPATCH_VPOPCNTDQ(RaBitQSQ4UBinaryIP, RaBitQSQ4UBinaryType);
static RaBitQSQ4UBinaryWithBaseSumType
GetRaBitQSQ4UBinaryIPWithBaseSum() {
    if (SimdStatus::SupportAVX512VPOPCNTDQ()) {
#if defined(ENABLE_AVX512VPOPCNTDQ)
        return avx512vpopcntdq::RaBitQSQ4UBinaryIPWithBaseSum;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQSQ4UBinaryIPWithBaseSum;
#endif
    }
    return generic::RaBitQSQ4UBinaryIPWithBaseSum;
}
RaBitQSQ4UBinaryWithBaseSumType RaBitQSQ4UBinaryIPWithBaseSum = GetRaBitQSQ4UBinaryIPWithBaseSum();
static RaBitQSQ4UBinaryWithBaseSumBatch4Type
GetRaBitQSQ4UBinaryIPWithBaseSumBatch4() {
    if (SimdStatus::SupportAVX512VPOPCNTDQ()) {
#if defined(ENABLE_AVX512VPOPCNTDQ)
        return avx512vpopcntdq::RaBitQSQ4UBinaryIPWithBaseSumBatch4;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQSQ4UBinaryIPWithBaseSumBatch4;
#endif
    }
    return generic::RaBitQSQ4UBinaryIPWithBaseSumBatch4;
}
RaBitQSQ4UBinaryWithBaseSumBatch4Type RaBitQSQ4UBinaryIPWithBaseSumBatch4 =
    GetRaBitQSQ4UBinaryIPWithBaseSumBatch4();

static RaBitQCodeCodeType
GetRaBitQCodeCodeIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQCodeCodeIP;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQCodeCodeIP;
#endif
    }
    if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::RaBitQCodeCodeIP;
#endif
    }
    if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::RaBitQCodeCodeIP;
#endif
    }
    return generic::RaBitQCodeCodeIP;
}
RaBitQCodeCodeType RaBitQCodeCodeIP = GetRaBitQCodeCodeIP();
static RaBitQPackScalarToSplitPlanesType
GetRaBitQPackScalarToSplitPlanes() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQPackScalarToSplitPlanes;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQPackScalarToSplitPlanes;
#endif
    }
    if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::RaBitQPackScalarToSplitPlanes;
#endif
    }
    if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::RaBitQPackScalarToSplitPlanes;
#endif
    }
    return generic::RaBitQPackScalarToSplitPlanes;
}

RaBitQPackScalarToSplitPlanesType RaBitQPackScalarToSplitPlanes =
    GetRaBitQPackScalarToSplitPlanes();

VSAG_DEFINE_SIMD_DISPATCH(FHTRotate, FHTRotateType);
VSAG_DEFINE_SIMD_DISPATCH(KacsWalk, KacsWalkType);
VSAG_DEFINE_SIMD_DISPATCH(VecRescale, VecRescaleType);
VSAG_DEFINE_SIMD_DISPATCH(RotateOp, RotateOpType);

static RaBitQFloatTwoBitCenteredType
GetRaBitQFloatTwoBitCenteredIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatTwoBitCenteredIP;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatTwoBitCenteredIP;
#endif
    }
    return generic::RaBitQFloatTwoBitCenteredIP;
}
RaBitQFloatTwoBitCenteredType RaBitQFloatTwoBitCenteredIP = GetRaBitQFloatTwoBitCenteredIP();

static RaBitQFloatTwoBitCenteredBatch4Type
GetRaBitQFloatTwoBitCenteredIPBatch4() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatTwoBitCenteredIPBatch4;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatTwoBitCenteredIPBatch4;
#endif
    }
    return generic::RaBitQFloatTwoBitCenteredIPBatch4;
}
RaBitQFloatTwoBitCenteredBatch4Type RaBitQFloatTwoBitCenteredIPBatch4 =
    GetRaBitQFloatTwoBitCenteredIPBatch4();

static RaBitQFloatThreeBitCenteredType
GetRaBitQFloatThreeBitCenteredIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatThreeBitCenteredIP;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatThreeBitCenteredIP;
#endif
    }
    return generic::RaBitQFloatThreeBitCenteredIP;
}
RaBitQFloatThreeBitCenteredType RaBitQFloatThreeBitCenteredIP = GetRaBitQFloatThreeBitCenteredIP();

static RaBitQFloatThreeBitCenteredBatch4Type
GetRaBitQFloatThreeBitCenteredIPBatch4() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatThreeBitCenteredIPBatch4;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatThreeBitCenteredIPBatch4;
#endif
    }
    return generic::RaBitQFloatThreeBitCenteredIPBatch4;
}
RaBitQFloatThreeBitCenteredBatch4Type RaBitQFloatThreeBitCenteredIPBatch4 =
    GetRaBitQFloatThreeBitCenteredIPBatch4();

static RaBitQFloatFourBitCenteredType
GetRaBitQFloatFourBitCenteredIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatFourBitCenteredIP;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatFourBitCenteredIP;
#endif
    }
    return generic::RaBitQFloatFourBitCenteredIP;
}
RaBitQFloatFourBitCenteredType RaBitQFloatFourBitCenteredIP = GetRaBitQFloatFourBitCenteredIP();

static RaBitQFloatFourBitCenteredBatch4Type
GetRaBitQFloatFourBitCenteredIPBatch4() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatFourBitCenteredIPBatch4;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatFourBitCenteredIPBatch4;
#endif
    }
    return generic::RaBitQFloatFourBitCenteredIPBatch4;
}
RaBitQFloatFourBitCenteredBatch4Type RaBitQFloatFourBitCenteredIPBatch4 =
    GetRaBitQFloatFourBitCenteredIPBatch4();

static RaBitQFloatThreeBitByLookupType
GetRaBitQFloatThreeBitIPByLookup() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatThreeBitIPByLookup;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatThreeBitIPByLookup;
#endif
    }
    return generic::RaBitQFloatThreeBitIPByLookup;
}
RaBitQFloatThreeBitByLookupType RaBitQFloatThreeBitIPByLookup = GetRaBitQFloatThreeBitIPByLookup();

static RaBitQFloatMultiBitByLookupType
GetRaBitQFloatMultiBitIPByLookup() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatMultiBitIPByLookup;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatMultiBitIPByLookup;
#endif
    }
    return generic::RaBitQFloatMultiBitIPByLookup;
}
RaBitQFloatMultiBitByLookupType RaBitQFloatMultiBitIPByLookup = GetRaBitQFloatMultiBitIPByLookup();

static RaBitQFloatThreeBitBatch4ByLookupType
GetRaBitQFloatThreeBitIPBatch4ByLookup() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatThreeBitIPBatch4ByLookup;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatThreeBitIPBatch4ByLookup;
#endif
    }
    return generic::RaBitQFloatThreeBitIPBatch4ByLookup;
}
RaBitQFloatThreeBitBatch4ByLookupType RaBitQFloatThreeBitIPBatch4ByLookup =
    GetRaBitQFloatThreeBitIPBatch4ByLookup();

static RaBitQFloatMultiBitBatch4ByLookupType
GetRaBitQFloatMultiBitIPBatch4ByLookup() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatMultiBitIPBatch4ByLookup;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatMultiBitIPBatch4ByLookup;
#endif
    }
    return generic::RaBitQFloatMultiBitIPBatch4ByLookup;
}
RaBitQFloatMultiBitBatch4ByLookupType RaBitQFloatMultiBitIPBatch4ByLookup =
    GetRaBitQFloatMultiBitIPBatch4ByLookup();

static RaBitQFloatSQType
GetRaBitQFloatSQIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatSQIP;
#endif
    }
    if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatSQIP;
#endif
    }
    if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::RaBitQFloatSQIP;
#endif
    }
    if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::RaBitQFloatSQIP;
#endif
    }
    return generic::RaBitQFloatSQIP;
}
RaBitQFloatSQType RaBitQFloatSQIP = GetRaBitQFloatSQIP();

// FlipSign only has AVX512 / SVE / NEON implementations (no AVX2, AVX
// or SSE variants). Kept as an explicit cascade rather than introducing
// a one-off macro for this unique combination.
static FlipSignType
GetFlipSign() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::FlipSign;
#endif
    }
    if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::FlipSign;
#endif
    }
    if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::FlipSign;
#endif
    }
    return generic::FlipSign;
}
FlipSignType FlipSign = GetFlipSign();

}  // namespace vsag
