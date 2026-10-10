// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "rabitq_filter_ip.h"

namespace vsag::lite::experiment {

float
rabitq_filter_ip_generic(const float* query, const uint8_t* filter, uint64_t dim) {
    const uint64_t plane_bytes = (dim + 7) / 8;
    const uint8_t* plane0 = filter;
    const uint8_t* plane1 = filter + plane_bytes;
    const uint8_t* plane2 = filter + 2 * plane_bytes;
    float result = 0.0F;
    for (uint64_t d = 0; d < dim; ++d) {
        const uint64_t byte = d >> 3U;
        const auto mask = static_cast<uint8_t>(1U << (d & 7U));
        float weight = (plane0[byte] & mask) != 0U ? 2.0F : -2.0F;
        weight += (plane1[byte] & mask) != 0U ? 1.0F : -1.0F;
        weight += (plane2[byte] & mask) != 0U ? 0.5F : -0.5F;
        result += query[d] * weight;
    }
    return result;
}

RaBitQFilterIP
select_rabitq_filter_ip() {
    static const RaBitQFilterIP selected = []() -> RaBitQFilterIP {
#ifdef VSAG_LITE_RABITQ_X86_SIMD
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512f") and __builtin_cpu_supports("avx512dq") and
            __builtin_cpu_supports("avx512bw") and __builtin_cpu_supports("avx512vl")) {
            return rabitq_filter_ip_avx512;
        }
        if (__builtin_cpu_supports("avx2") and __builtin_cpu_supports("fma")) {
            return rabitq_filter_ip_avx2;
        }
#endif
        return rabitq_filter_ip_generic;
    }();
    return selected;
}

}  // namespace vsag::lite::experiment
