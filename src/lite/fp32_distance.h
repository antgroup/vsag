// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vsag::lite::detail {

using FP32Distance = float (*)(const float*, const float*, uint64_t);

float
GenericFP32Distance(const float* query, const float* vector, uint64_t dim);

#ifdef VSAG_LITE_HAS_X86_SIMD
float
SseFP32Distance(const float* query, const float* vector, uint64_t dim);
float
Avx2FP32Distance(const float* query, const float* vector, uint64_t dim);
float
Avx512FP32Distance(const float* query, const float* vector, uint64_t dim);
#endif

/** Select the fastest compiled kernel supported by the running CPU and OS. */
FP32Distance
SelectFP32Distance();

}  // namespace vsag::lite::detail
