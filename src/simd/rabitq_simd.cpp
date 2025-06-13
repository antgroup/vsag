
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

namespace vsag {

static RaBitQFloatBinaryType
GetRaBitQFloatBinaryIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQFloatBinaryIP;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::RaBitQFloatBinaryIP;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::RaBitQFloatBinaryIP;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::RaBitQFloatBinaryIP;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::RaBitQFloatBinaryIP;
#endif
    }
    return generic::RaBitQFloatBinaryIP;
}

static RaBitQSQ4UBinaryType
GetRaBitQSQ4UBinaryIP() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::RaBitQSQ4UBinaryIP;
#endif
    }
    return generic::RaBitQSQ4UBinaryIP;
}

RaBitQFloatBinaryType RaBitQFloatBinaryIP = GetRaBitQFloatBinaryIP();
RaBitQSQ4UBinaryType RaBitQSQ4UBinaryIP = GetRaBitQSQ4UBinaryIP();

}  // namespace vsag
