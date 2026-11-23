
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
#include "fp16_simd.h"

#include "simd_dispatch.h"

namespace vsag {

// The x86 half kernels and their tails use F16C in addition to their
// nominal ISA. Keep the SSE software converter available without F16C.
#define VSAG_DEFINE_FP16_DISPATCH(name, type)    \
    static type Get##name() {                    \
        if (SimdStatus::SupportAVX512F16C()) {   \
            VSAG_SIMD_DISPATCH_BODY_AVX512(name) \
        }                                        \
        if (SimdStatus::SupportAVX2F16C()) {     \
            VSAG_SIMD_DISPATCH_BODY_AVX2(name)   \
        }                                        \
        if (SimdStatus::SupportAVXF16C()) {      \
            VSAG_SIMD_DISPATCH_BODY_AVX(name)    \
        }                                        \
        if (SimdStatus::SupportSSE()) {          \
            VSAG_SIMD_DISPATCH_BODY_SSE(name)    \
        }                                        \
        if (SimdStatus::SupportSVE()) {          \
            VSAG_SIMD_DISPATCH_BODY_SVE(name)    \
        }                                        \
        if (SimdStatus::SupportNEON()) {         \
            VSAG_SIMD_DISPATCH_BODY_NEON(name)   \
        }                                        \
        return generic::name;                    \
    }                                            \
    type name = Get##name()

VSAG_DEFINE_FP16_DISPATCH(FP16ComputeIP, FP16ComputeType);
VSAG_DEFINE_FP16_DISPATCH(FP16ComputeL2Sqr, FP16ComputeType);
VSAG_DEFINE_FP16_DISPATCH(FP16SparseAccumulate, FP16SparseAccumulateType);
#undef VSAG_DEFINE_FP16_DISPATCH
}  // namespace vsag
