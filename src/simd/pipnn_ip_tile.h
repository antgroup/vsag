// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vsag {

// Write new-by-old scores for complete four-row blocks. The caller handles old rows
// after the returned prefix and the new-by-new triangle with its usual kernels.
uint64_t
ComputePiPNNOldRowTiles(const float* const* rows,
                        const uint32_t* members,
                        uint64_t old_count,
                        uint64_t new_count,
                        uint64_t dim,
                        float* dots);

namespace avx512 {
inline constexpr uint64_t PIPNN_TILE_DIM = 960;
uint64_t
ComputePiPNNOldRowTiles960(const float* const* rows,
                           const uint32_t* members,
                           uint64_t old_count,
                           uint64_t new_count,
                           float* dots);
}  // namespace avx512
}  // namespace vsag
