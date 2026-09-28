// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vsag::lite::experiment {

using RaBitQFilterIP = float (*)(const float*, const uint8_t*, uint64_t);

float
rabitq_filter_ip_generic(const float* query, const uint8_t* filter, uint64_t dim);

float
rabitq_filter_ip_avx2(const float* query, const uint8_t* filter, uint64_t dim);

float
rabitq_filter_ip_avx512(const float* query, const uint8_t* filter, uint64_t dim);

RaBitQFilterIP
select_rabitq_filter_ip();

}  // namespace vsag::lite::experiment
