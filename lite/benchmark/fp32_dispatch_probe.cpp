// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#ifdef VSAG_BENCH_FULL
#include "simd/fp32_simd.h"
#else
#include "lite/fp32_distance.h"
#endif

int main() {
#ifdef VSAG_BENCH_FULL
  const bool scalar = vsag::FP32ComputeL2Sqr == vsag::avx512::FP32ComputeL2Sqr;
  const bool batch =
      vsag::FP32ComputeL2SqrBatch4 == vsag::avx512::FP32ComputeL2SqrBatch4;
  std::cout << "{\"side\":\"native_full\",\"scalar_avx512\":" << scalar
            << ",\"batch4_avx512\":" << batch << "}\n";
  return scalar and batch ? 0 : 1;
#else
  const bool selected = vsag::lite::detail::select_fp32_distance() ==
                        vsag::lite::detail::avx512_fp32_distance;
  std::cout << "{\"side\":\"lite\",\"scalar_avx512\":" << selected << "}\n";
  return selected ? 0 : 1;
#endif
}
