// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>

namespace vsag {

/**
 * @brief Options controlling Build Cache behavior.
 *
 * Build Cache uses the neighbor relationships from a previous index build
 * to provide a warm start for the next build, reducing iteration rounds
 * needed for graph convergence.
 */
struct BuildCacheOptions {
    /** Whether to enable Warm Start. When false, BuildWithCache falls back to normal Build. */
    bool enable_warm_start = true;

    /** Number of Refine iterations for cache-hit nodes. Hit nodes already have
     *  high-quality neighbors from the cache and converge quickly (1-2 rounds typically). */
    uint32_t hit_refine_rounds = 2;

    /** Number of Refine iterations for cache-miss nodes. Miss nodes start from
     *  empty neighbors and need more iterations to integrate into the graph. */
    uint32_t missed_refine_rounds = 4;

    /** Whether to drop neighbors that map to deleted nodes (mapped to UINT32_MAX).
     *  When true, invalid neighbors are removed from the neighbor list before injection. */
    bool drop_invalid_neighbors = true;
};

/**
 * @brief Statistics from the most recent BuildWithCache operation.
 *
 * These metrics are for observability only and do not influence build decisions.
 */
struct BuildCacheStats {
    /** Total number of nodes in the new index. */
    uint64_t total_nodes = 0;

    /** Number of nodes present in the cache (old index size). */
    uint64_t cached_nodes = 0;

    /** Number of nodes that hit the cache (FeatureID exists in both old and new index). */
    uint64_t hit_nodes = 0;

    /** Number of nodes that missed the cache (new nodes or no record in cache). */
    uint64_t missed_nodes = 0;

    /** Total number of neighbors dropped because they mapped to deleted nodes. */
    uint64_t dropped_neighbors = 0;

    /** Total number of neighbors that mapped to deleted nodes (before drop filter). */
    uint64_t invalid_neighbors = 0;

    /** Actual Refine rounds executed for hit nodes. */
    uint64_t hit_refine_rounds = 0;

    /** Actual Refine rounds executed for missed nodes. */
    uint64_t missed_refine_rounds = 0;

    /** Cache loading time in microseconds. */
    uint64_t cache_load_us = 0;

    /** Warm Start neighbor injection time in microseconds. */
    uint64_t warm_start_apply_us = 0;

    /** Refine time for hit nodes in microseconds. */
    uint64_t hit_refine_us = 0;

    /** Refine time for missed nodes in microseconds. */
    uint64_t missed_refine_us = 0;

    /** Actual cache hit rate (hit_nodes / total_nodes). */
    float cache_hit_rate = 0.0F;
};

}  // namespace vsag
