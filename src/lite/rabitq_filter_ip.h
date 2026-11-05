// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vsag::lite::detail::rabitq {

using RaBitQFilterIP = float (*)(const float*, const uint8_t*, uint64_t);
using RaBitQFilterIPBatch4 = void (*)(
    const float*, const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, uint64_t, float*);

float
rabitq_filter_ip_generic(const float* query, const uint8_t* filter, uint64_t dim);

float
rabitq_filter_ip_avx2(const float* query, const uint8_t* filter, uint64_t dim);

float
rabitq_filter_ip_avx512(const float* query, const uint8_t* filter, uint64_t dim);

void
rabitq_filter_ip_batch4_generic(const float* query,
                                const uint8_t* filter0,
                                const uint8_t* filter1,
                                const uint8_t* filter2,
                                const uint8_t* filter3,
                                uint64_t dim,
                                float* results);

void
rabitq_filter_ip_batch4_avx2(const float* query,
                             const uint8_t* filter0,
                             const uint8_t* filter1,
                             const uint8_t* filter2,
                             const uint8_t* filter3,
                             uint64_t dim,
                             float* results);

void
rabitq_filter_ip_batch4_avx512(const float* query,
                               const uint8_t* filter0,
                               const uint8_t* filter1,
                               const uint8_t* filter2,
                               const uint8_t* filter3,
                               uint64_t dim,
                               float* results);

float
rabitq_supplement_ip_generic(const float* query, const uint8_t* supplement, uint64_t dim);

float
rabitq_supplement_ip_avx2(const float* query, const uint8_t* supplement, uint64_t dim);

float
rabitq_supplement_ip_avx512(const float* query, const uint8_t* supplement, uint64_t dim);

RaBitQFilterIP
select_rabitq_supplement_ip();

RaBitQFilterIP
select_rabitq_filter_ip();

RaBitQFilterIPBatch4
select_rabitq_filter_ip_batch4();

}  // namespace vsag::lite::detail::rabitq
