// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "fp32_distance.h"
#include "simd/kernels/compute_l2.h"
#include "simd/traits/simd_traits_avx2.h"

namespace vsag::lite::detail {

float
avx2_fp32_distance(const float* query, const float* vector, uint64_t dim) {
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::Avx2Tag>>(
        query, vector, dim, sse_fp32_distance);
}

}  // namespace vsag::lite::detail
