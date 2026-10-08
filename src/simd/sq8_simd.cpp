
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
#include "sq8_simd.h"

#include "simd_dispatch.h"

namespace vsag {

VSAG_DEFINE_SIMD_DISPATCH(SQ8ComputeIP, SQ8ComputeType);
VSAG_DEFINE_SIMD_DISPATCH(SQ8ComputeL2Sqr, SQ8ComputeType);

static void
SQ8ComputeL2SqrBatch4Fallback(const float* query,
                              const uint8_t* codes1,
                              const uint8_t* codes2,
                              const uint8_t* codes3,
                              const uint8_t* codes4,
                              const float* lower_bound,
                              const float* diff,
                              uint64_t dim,
                              float& dist1,
                              float& dist2,
                              float& dist3,
                              float& dist4) {
    dist1 = SQ8ComputeL2Sqr(query, codes1, lower_bound, diff, dim);
    dist2 = SQ8ComputeL2Sqr(query, codes2, lower_bound, diff, dim);
    dist3 = SQ8ComputeL2Sqr(query, codes3, lower_bound, diff, dim);
    dist4 = SQ8ComputeL2Sqr(query, codes4, lower_bound, diff, dim);
}

static SQ8ComputeBatch4Type
GetSQ8ComputeL2SqrBatch4() {
    if (SimdStatus::SupportAVX512()) {
        VSAG_SIMD_DISPATCH_BODY_AVX512(SQ8ComputeL2SqrBatch4)
    }
    return SQ8ComputeL2SqrBatch4Fallback;
}
SQ8ComputeBatch4Type SQ8ComputeL2SqrBatch4 = GetSQ8ComputeL2SqrBatch4();
VSAG_DEFINE_SIMD_DISPATCH(SQ8ComputeCodesIP, SQ8ComputeCodesType);
VSAG_DEFINE_SIMD_DISPATCH(SQ8ComputeCodesL2Sqr, SQ8ComputeCodesType);
VSAG_DEFINE_SIMD_DISPATCH(SQ8SparseAccumulate, SQ8SparseAccumulateType);
}  // namespace vsag
