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

#ifdef VSAG_LITE_HAS_X86_SIMD
FP32Distance
select_fp32_distance_for(bool sse, bool avx2, bool avx512, bool fma) {
    // AVX512 delegates short rows and tails to the FMA-based AVX2 kernel.
    if (avx512 and avx2 and fma) {
        return avx512_fp32_distance;
    }
    if (avx2 and fma) {
        return avx2_fp32_distance;
    }
    if (sse) {
        return sse_fp32_distance;
    }
    return generic_fp32_distance;
}
#endif

FP32Distance
select_fp32_distance() {
    static const FP32Distance selected = []() -> FP32Distance {
#ifdef VSAG_LITE_HAS_X86_SIMD
        // GCC/Clang check both CPU features and OS-enabled vector register state.
        __builtin_cpu_init();
        const bool avx512 =
            __builtin_cpu_supports("avx512f") and __builtin_cpu_supports("avx512dq") and
            __builtin_cpu_supports("avx512bw") and __builtin_cpu_supports("avx512vl");
        return select_fp32_distance_for(__builtin_cpu_supports("sse4.1"),
                                        __builtin_cpu_supports("avx2"),
                                        avx512,
                                        __builtin_cpu_supports("fma"));
#endif
        return generic_fp32_distance;
    }();
    return selected;
}

}  // namespace vsag::lite::detail
