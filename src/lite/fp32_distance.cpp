// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "fp32_distance.h"

#include "simd/kernels/compute_l2.h"
#include "simd/traits/simd_traits_generic.h"

namespace vsag::lite::detail {

float
generic_fp32_distance(const float* query, const float* vector, uint64_t dim) {
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::GenericTag>>(query, vector, dim);
}

FP32Distance
select_fp32_distance() {
    static const FP32Distance selected = []() -> FP32Distance {
#ifdef VSAG_LITE_HAS_X86_SIMD
        // GCC/Clang check both CPU features and OS-enabled vector register state.
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512f") and __builtin_cpu_supports("avx512dq") and
            __builtin_cpu_supports("avx512bw") and __builtin_cpu_supports("avx512vl") and
            __builtin_cpu_supports("avx2")) {
            return avx512_fp32_distance;
        }
        if (__builtin_cpu_supports("avx2") and __builtin_cpu_supports("fma")) {
            return avx2_fp32_distance;
        }
        if (__builtin_cpu_supports("sse4.1")) {
            return sse_fp32_distance;
        }
#endif
        return generic_fp32_distance;
    }();
    return selected;
}

}  // namespace vsag::lite::detail
