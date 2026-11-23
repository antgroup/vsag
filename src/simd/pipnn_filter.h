// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vsag::pipnn_filter {

inline constexpr uint16_t COINCIDENT_HASH_FLAG = 1U << 15;

struct RelativeHashes {
    uint16_t forward;
    uint16_t reverse;
};

using HashFunction = RelativeHashes (*)(const float* source, const float* target, uint64_t planes);
// Distances and thresholds are in leaf row order. Include ties for the caller's ID tie-break.
using KthFilter = uint64_t (*)(const float* distances,
                               const float* thresholds,
                               uint64_t count,
                               uint32_t* selected);

struct Kernels {
    HashFunction hashes;
    KthFilter kth;  // Null selects the existing row-wise top-k loop.
};

const Kernels&
GetKernels();

namespace avx512 {
RelativeHashes
Hash(const float* source, const float* target, uint64_t planes);
uint64_t
SelectKth(const float* distances, const float* thresholds, uint64_t count, uint32_t* selected);
}  // namespace avx512
}  // namespace vsag::pipnn_filter
