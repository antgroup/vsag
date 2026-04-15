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
#include <algorithm>

#include "simd/simd.h"

namespace vsag {

/**
 * @file prefetch.h
 * @brief Cache prefetching utilities for performance optimization.
 */

/**
 * @brief Template function to prefetch multiple cache lines.
 * @tparam N Number of cache lines to prefetch (max 24).
 * @param data Pointer to the data to prefetch.
 *
 * Uses __builtin_prefetch to load cache lines into L1 cache,
 * improving performance for subsequent memory accesses.
 */
template <int N>
__inline void __attribute__((__always_inline__)) PrefetchImpl(const void* data) {
    if constexpr (N > 24) {
        return PrefetchImpl<24>(data);
    }
    for (int i = 0; i < N; ++i) {
        __builtin_prefetch(static_cast<const char*>(data) + i * 64, 0, 3);
    }
}

/**
 * @brief Prefetches memory region for improved cache performance.
 * @param data Pointer to the data to prefetch.
 * @param size Size of the memory region in bytes.
 *
 * Automatically calculates the number of cache lines to prefetch
 * based on the specified size.
 */
void
PrefetchLines(const void* data, uint64_t size);

}  // namespace vsag