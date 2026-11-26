
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

#include "fp32_simd.h"

#include "pipnn_filter.h"
#include "pipnn_ip_tile.h"
#include "simd_dispatch.h"

namespace vsag {

VSAG_DEFINE_SIMD_DISPATCH(FP32ComputeIP, FP32ComputeType);
VSAG_DEFINE_SIMD_DISPATCH(FP32ComputeL2Sqr, FP32ComputeType);
VSAG_DEFINE_SIMD_DISPATCH(FP32SparseAccumulate, FP32SparseAccumulateType);
VSAG_DEFINE_SIMD_DISPATCH(FP32ComputeIPBatch4, FP32ComputeBatch4Type);
VSAG_DEFINE_SIMD_DISPATCH(FP32ComputeL2SqrBatch4, FP32ComputeBatch4Type);
VSAG_DEFINE_SIMD_DISPATCH(FP32Sub, FP32ArithmeticType);
VSAG_DEFINE_SIMD_DISPATCH(FP32Add, FP32ArithmeticType);
VSAG_DEFINE_SIMD_DISPATCH(FP32Mul, FP32ArithmeticType);
VSAG_DEFINE_SIMD_DISPATCH(FP32Div, FP32ArithmeticType);
VSAG_DEFINE_SIMD_DISPATCH(FP32ReduceAdd, FP32ReduceType);

uint64_t
ComputePiPNNOldRowTiles(const float* const* rows,
                        const uint32_t* members,
                        uint64_t old_count,
                        uint64_t new_count,
                        uint64_t dim,
                        float* dots) {
#if defined(ENABLE_AVX512)
    if (dim == avx512::PIPNN_TILE_DIM && FP32ComputeIP == avx512::FP32ComputeIP &&
        FP32ComputeIPBatch4 == avx512::FP32ComputeIPBatch4) {
        return avx512::ComputePiPNNOldRowTiles960(rows, members, old_count, new_count, dots);
    }
#endif
    return 0;
}

namespace pipnn_filter {
RelativeHashes
ScalarHash(const float* source, const float* target, uint64_t planes) {
    RelativeHashes hash{};
    bool coincident = true;
    for (uint64_t i = 0; i < planes; ++i) {
        const uint16_t bit = static_cast<uint16_t>(1U << i);
        if (target[i] >= source[i])
            hash.forward |= bit;
        if (source[i] >= target[i])
            hash.reverse |= bit;
        coincident = coincident && source[i] == target[i];
    }
    if (coincident) {
        hash.forward |= COINCIDENT_HASH_FLAG;
        hash.reverse |= COINCIDENT_HASH_FLAG;
    }
    return hash;
}

const Kernels&
GetKernels() {
    static const Kernels kernels = []() -> Kernels {
#if defined(ENABLE_AVX512)
        if (SimdStatus::SupportAVX512()) {
            return {avx512::Hash, avx512::SelectKth};
        }
#endif
        return {ScalarHash, nullptr};
    }();
    return kernels;
}
}  // namespace pipnn_filter
}  // namespace vsag
