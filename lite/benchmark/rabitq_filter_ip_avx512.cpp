// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "rabitq_filter_ip.h"
#include "simd/kernels/rabitq_compute.h"
#include "simd/traits/simd_traits_avx512.h"

namespace vsag::lite::experiment {

float
rabitq_filter_ip_avx512(const float* query, const uint8_t* filter, uint64_t dim) {
    return simd::RaBitQFloatThreeBitCenteredIPImpl<simd::RaBitQTraits<simd::Avx512RaBitQTag>>(
        query, filter, dim, rabitq_filter_ip_generic);
}

}  // namespace vsag::lite::experiment
