
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
#include "sq8_uniform_simd.h"

#include "simd_dispatch.h"

namespace vsag {

VSAG_DEFINE_SIMD_DISPATCH(SQ8UniformComputeCodesIP, SQ8UniformComputeCodesType);

// Batch IP dispatch. Prefer AMX when available; otherwise fall back to
// the same cascade as the scalar entry point. This keeps the macro
// system simple: VSAG_DEFINE_SIMD_DISPATCH always emits an AVX512
// branch, and the AMX branch is layered on top by hand because no
// other dispatch in the codebase uses AMX yet.
static SQ8UniformComputeCodesIPBatchType
GetSQ8UniformComputeCodesIPBatch() {
    if (SimdStatus::SupportAMX()) {
        VSAG_SIMD_DISPATCH_BODY_AMX(SQ8UniformComputeCodesIPBatch)
    }
    if (SimdStatus::SupportAVX512()) {
        VSAG_SIMD_DISPATCH_BODY_AVX512(SQ8UniformComputeCodesIPBatch)
    }
    if (SimdStatus::SupportAVX2()) {
        VSAG_SIMD_DISPATCH_BODY_AVX2(SQ8UniformComputeCodesIPBatch)
    }
    if (SimdStatus::SupportAVX()) {
        VSAG_SIMD_DISPATCH_BODY_AVX(SQ8UniformComputeCodesIPBatch)
    }
    if (SimdStatus::SupportSSE()) {
        VSAG_SIMD_DISPATCH_BODY_SSE(SQ8UniformComputeCodesIPBatch)
    }
    if (SimdStatus::SupportSVE()) {
        VSAG_SIMD_DISPATCH_BODY_SVE(SQ8UniformComputeCodesIPBatch)
    }
    if (SimdStatus::SupportNEON()) {
        VSAG_SIMD_DISPATCH_BODY_NEON(SQ8UniformComputeCodesIPBatch)
    }
    return generic::SQ8UniformComputeCodesIPBatch;
}
SQ8UniformComputeCodesIPBatchType SQ8UniformComputeCodesIPBatch =
    GetSQ8UniformComputeCodesIPBatch();
}  // namespace vsag
