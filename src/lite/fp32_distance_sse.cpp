// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "fp32_distance.h"
#include "simd/kernels/compute_l2.h"
#include "simd/traits/simd_traits_sse.h"

namespace vsag::lite::detail {

float
SseFP32Distance(const float* query, const float* vector, uint64_t dim) {
    return simd::ComputeL2SqrImpl<simd::SimdTraits<simd::SseTag>>(
        query, vector, dim, GenericFP32Distance);
}

}  // namespace vsag::lite::detail
