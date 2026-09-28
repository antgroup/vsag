// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vsag::lite::detail {

using FP32Distance = float (*)(const float*, const float*, uint64_t);

float
generic_fp32_distance(const float* query, const float* vector, uint64_t dim);

#ifdef VSAG_LITE_HAS_X86_SIMD
float
sse_fp32_distance(const float* query, const float* vector, uint64_t dim);
float
avx2_fp32_distance(const float* query, const float* vector, uint64_t dim);
float
avx512_fp32_distance(const float* query, const float* vector, uint64_t dim);
#endif

/** Select the fastest compiled kernel supported by the running CPU and OS. */
FP32Distance
select_fp32_distance();

}  // namespace vsag::lite::detail
