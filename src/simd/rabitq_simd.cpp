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
VSAG_DEFINE_SIMD_DISPATCH_VPOPCNTDQ(RaBitQSQ4UBinaryIP, RaBitQSQ4UBinaryType);
VSAG_DEFINE_SIMD_DISPATCH(FHTRotate, FHTRotateType);
VSAG_DEFINE_SIMD_DISPATCH(KacsWalk, KacsWalkType);
VSAG_DEFINE_SIMD_DISPATCH(VecRescale, VecRescaleType);
VSAG_DEFINE_SIMD_DISPATCH(RotateOp, RotateOpType);

// RaBitQFloatSQIP currently only has a generic implementation. Kept as
// an explicit one-liner so the absence of SIMD variants is obvious.
RaBitQFloatSQType RaBitQFloatSQIP = generic::RaBitQFloatSQIP;

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
