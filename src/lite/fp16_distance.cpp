// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "fp16_distance.h"

#ifdef VSAG_LITE_HAS_X86_SIMD
#include <cpuid.h>
#endif

#include "simd/kernels/half_compute.h"
#include "simd/traits/simd_traits_generic.h"

namespace vsag::lite::detail {
namespace {

#ifdef VSAG_LITE_HAS_X86_SIMD
bool
supports_f16c() {
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    constexpr unsigned int k_f16c_bit = 1U << 29U;
    return __get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0 and (ecx & k_f16c_bit) != 0;
}
#endif

}  // namespace

float
generic_fp16_distance(const uint16_t* query, const uint16_t* vector, uint64_t dim) {
    return simd::HalfComputeL2SqrImpl<simd::FP16Traits<simd::GenericFP16Tag>>(
        reinterpret_cast<const uint8_t*>(query), reinterpret_cast<const uint8_t*>(vector), dim);
}

#ifdef VSAG_LITE_HAS_X86_SIMD
FP16Distance
select_fp16_distance_for(bool avx, bool avx2, bool avx512, bool fma, bool f16c) {
    if (avx512 and avx2 and f16c) {
        return avx512_fp16_distance;
    }
    if (avx2 and fma and f16c) {
        return avx2_fp16_distance;
    }
    if (avx and f16c) {
        return avx_fp16_distance;
    }
    return generic_fp16_distance;
}
#endif

FP16Distance
select_fp16_distance() {
    static const FP16Distance selected = []() -> FP16Distance {
#ifdef VSAG_LITE_HAS_X86_SIMD
        __builtin_cpu_init();
        const bool avx512 =
            __builtin_cpu_supports("avx512f") and __builtin_cpu_supports("avx512dq") and
            __builtin_cpu_supports("avx512bw") and __builtin_cpu_supports("avx512vl");
        return select_fp16_distance_for(__builtin_cpu_supports("avx"),
                                        __builtin_cpu_supports("avx2"),
                                        avx512,
                                        __builtin_cpu_supports("fma"),
                                        supports_f16c());
#else
        return generic_fp16_distance;
#endif
    }();
    return selected;
}

}  // namespace vsag::lite::detail
