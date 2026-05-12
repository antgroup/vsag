
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag::amx {

// AMX BF16 GEMM specialised for KMeans-style centroid assignment.
//
// Computes:
//   C[m + n * ldc] = alpha * < A[m, :], B[n, :] >   for m in [0, M), n in [0, N)
//
// where the inner product is over the K dimension. Inputs A and B are FP32
// (row-major, contiguous per row); they are internally packed into BF16 tiles
// and multiplied with `_tile_dpbf16ps` (BF16 x BF16 -> FP32 accumulation).
// Output C is column-major with leading dimension `ldc` (>= M) so that it
// drops into the existing column-major SGEMM call in `find_nearest_one_with_blas`.
//
// Constraints (caller is responsible):
//   - M >= 1, N >= 1, K >= 1
//   - lda == K, ldb == K  (i.e. A is row-major M x K, B is row-major N x K)
//   - ldc >= M
//   - The host must support AMX-BF16 (check `SimdStatus::SupportAMXBF16()`).
//
// Returns true on success. Returns false when AMX-BF16 is unavailable at
// runtime so callers can transparently fall back to SGEMM.
//
// Note: This is intentionally NOT a general-purpose BLAS replacement; it is
// shaped to the one call site that benefits (KMeans, dense FP32, modest M,
// large N, K up to a few thousand).
bool
SgemmBF16IPColMajorOut(int64_t m,
                       int64_t n,
                       int64_t k,
                       float alpha,
                       const float* RESTRICT a_row_major,
                       const float* RESTRICT b_row_major,
                       float* RESTRICT c_col_major,
                       int64_t ldc);

}  // namespace vsag::amx
