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

#include "pipnn_half_scalar.h"
#include "simd_status.h"

namespace vsag::pipnn_half {
Dispatch
ResolveDispatch(ImplementationSelection selection) {
    Dispatch result{detail::DecodeRowScalar<Format::FP16>,
                    detail::DecodeRowScalar<Format::BF16>,
                    detail::PreparationScalar};
#if defined(ENABLE_AVX2)
    if (selection.decode_bf16_avx2) {
        result.decode_bf16 = detail::DecodeRowBF16;
    }
#if defined(ENABLE_PIPNN_F16C)
    if (selection.decode_fp16_avx2_f16c) {
        result.decode_fp16 = detail::DecodeRowFP16;
    }
    if (selection.preparation_avx2_f16c) {
        result.preparation = detail::PreparationAVX2;
    }
#endif
#else
    (void)selection;
#endif
    return result;
}

const Dispatch&
GetDispatch() {
    static const Dispatch dispatch = [] {
        Capabilities caps{};
#if defined(ENABLE_AVX2)
        caps.compiled_avx2 = true;
#if defined(ENABLE_PIPNN_F16C)
        caps.compiled_f16c = true;
#endif
        caps.cpuinfo_initialized = SimdStatus::Init();
        // The selector requires both AVX and AVX2.
        caps.cpu_avx = caps.cpu_avx2 = SimdStatus::SupportAVXAndAVX2();
#if defined(ENABLE_PIPNN_F16C)
        caps.cpu_f16c = SimdStatus::SupportAVX2F16C();
#endif
#endif
        return ResolveDispatch(SelectImplementations(caps));
    }();
    return dispatch;
}
}  // namespace vsag::pipnn_half
