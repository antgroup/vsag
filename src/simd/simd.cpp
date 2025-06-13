
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

#include "simd.h"

#include <cpuinfo.h>

namespace vsag {

SimdStatus
setup_simd() {
    SimdStatus ret;

    if (cpuinfo_has_x86_sse()) {
        ret.runtime_has_sse = true;
#ifndef ENABLE_SSE
    }
#else
    }
    ret.dist_support_sse = true;
#endif

    if (cpuinfo_has_x86_avx()) {
        ret.runtime_has_avx = true;
#ifndef ENABLE_AVX
    }
#else
    }
    ret.dist_support_avx = true;
#endif

    if (cpuinfo_has_x86_avx2()) {
        ret.runtime_has_avx2 = true;
#ifndef ENABLE_AVX2
    }
#else
    }
    ret.dist_support_avx2 = true;
#endif

    if (cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512bw() &&
        cpuinfo_has_x86_avx512vl()) {
        ret.runtime_has_avx512f = true;
        ret.runtime_has_avx512dq = true;
        ret.runtime_has_avx512bw = true;
        ret.runtime_has_avx512vl = true;
#ifndef ENABLE_AVX512
    }
#else
    }
    ret.dist_support_avx512f = true;
    ret.dist_support_avx512dq = true;
    ret.dist_support_avx512bw = true;
    ret.dist_support_avx512vl = true;
#endif

    if (cpuinfo_has_arm_neon()) {
        ret.runtime_has_neon = true;
#ifndef ENABLE_NEON
    }
#else
    }
    ret.dist_support_neon = true;
#endif

    return ret;
}

}  // namespace vsag
