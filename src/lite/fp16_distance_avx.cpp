// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "fp16_distance.h"
#include "simd/kernels/half_compute.h"
#include "simd/traits/simd_traits_avx.h"

namespace vsag::lite::detail {

float
avx_fp16_distance(const uint16_t* query, const uint16_t* vector, uint64_t dim) {
    return simd::HalfComputeL2SqrImpl<simd::FP16Traits<simd::AvxFP16Tag>>(
        reinterpret_cast<const uint8_t*>(query),
        reinterpret_cast<const uint8_t*>(vector),
        dim,
        [](const uint8_t* left, const uint8_t* right, uint64_t tail) {
            return generic_fp16_distance(reinterpret_cast<const uint16_t*>(left),
                                         reinterpret_cast<const uint16_t*>(right),
                                         tail);
        });
}

}  // namespace vsag::lite::detail
