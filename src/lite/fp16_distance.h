// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vsag::lite::detail {

using FP16Distance = float (*)(const uint16_t*, const uint16_t*, uint64_t);

float
generic_fp16_distance(const uint16_t* query, const uint16_t* vector, uint64_t dim);

#ifdef VSAG_LITE_HAS_X86_SIMD
float
avx_fp16_distance(const uint16_t* query, const uint16_t* vector, uint64_t dim);
float
avx2_fp16_distance(const uint16_t* query, const uint16_t* vector, uint64_t dim);
float
avx512_fp16_distance(const uint16_t* query, const uint16_t* vector, uint64_t dim);
FP16Distance
select_fp16_distance_for(bool avx, bool avx2, bool avx512, bool fma, bool f16c);
#endif

/** Select the fastest compiled FP16 kernel supported by the running CPU and OS. */
FP16Distance
select_fp16_distance();

}  // namespace vsag::lite::detail
