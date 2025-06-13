
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

#include "simd_status.h"

namespace vsag {

static SQ8UniformComputeCodesType
GetSQ8UniformComputeCodesIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::SQ8UniformComputeCodesIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::SQ8UniformComputeCodesIP;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::SQ8UniformComputeCodesIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::SQ8UniformComputeCodesIP;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::SQ8UniformComputeCodesIP;
#endif
    }
    return generic::SQ8UniformComputeCodesIP;
}
SQ8UniformComputeCodesType SQ8UniformComputeCodesIP = GetSQ8UniformComputeCodesIP();
}  // namespace vsag
