// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "rabitq_constants.h"
#include "rabitq_filter_ip.h"
#include "simd/kernels/rabitq_compute.h"
#include "simd/traits/simd_traits_avx2.h"

namespace vsag::lite::detail::rabitq {

float
rabitq_filter_ip_avx2(const float* query, const uint8_t* filter, uint64_t dim) {
    return simd::RaBitQFloatThreeBitCenteredIPImpl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        query, filter, dim, rabitq_filter_ip_generic);
}

void
rabitq_filter_ip_batch4_avx2(const float* query,
                             const uint8_t* filter0,
                             const uint8_t* filter1,
                             const uint8_t* filter2,
                             const uint8_t* filter3,
                             uint64_t dim,
                             float* results) {
    simd::RaBitQFloatThreeBitCenteredIPBatch4Impl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        query, filter0, filter1, filter2, filter3, dim, results, rabitq_filter_ip_batch4_generic);
}

float
rabitq_supplement_ip_avx2(const float* query, const uint8_t* supplement, uint64_t dim) {
    return simd::RaBitQFloatSupplementCodeIPImpl<simd::RaBitQTraits<simd::Avx2RaBitQTag>>(
        query, supplement, dim, K_SUPPLEMENT_BITS);
}

}  // namespace vsag::lite::detail::rabitq
