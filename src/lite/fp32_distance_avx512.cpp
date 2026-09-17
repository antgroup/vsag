// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "fp32_distance.h"
#include "simd/kernels/compute_l2.h"
#include "simd/traits/simd_traits_avx512.h"

namespace vsag::lite::detail {

float
Avx512FP32Distance(const float* query, const float* vector, uint64_t dim) {
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::Avx512Tag>, /*Unroll=*/4>(
        query, vector, dim, Avx2FP32Distance);
}

}  // namespace vsag::lite::detail
