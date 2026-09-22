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
#include <vector>

#include "partitioners.h"

namespace vsag::rpt_bench {

struct UniformityMetrics {
    uint64_t num_partitions{0};
    uint64_t expected_partitions{0};  // ceil(count / bucket_size)
    uint64_t max_size{0};
    uint64_t min_size{0};
    double max_min_ratio{0.0};
    double mean_size{0.0};
    double stddev_size{0.0};
    double gini{0.0};

    /// True when the task's acceptance criteria hold: the partition count matches
    /// ceil(count / bucket_size), no partition exceeds the bucket size, and (unless
    /// count < bucket_size) no partition is below half of it and max/min stays within 2.
    bool satisfies_bounds{false};
};

/// Locality measured against the dataset's own ground truth. For every query, its
/// ground-truth top-k neighbours are looked up in the partitioning; a partitioning that
/// preserves locality keeps them together.
struct LocalityMetrics {
    uint64_t queries_evaluated{0};
    uint64_t k{0};

    /// Mean number of distinct partitions the top-k neighbours of a query fall into.
    /// Lower is better; 1.0 means every query's neighbourhood sits in a single partition.
    double mean_partitions_touched{0.0};

    /// Fraction of top-k neighbours sharing the partition of the nearest neighbour.
    double top1_same_partition_rate{0.0};

    /// Mean number of partitions that must be scanned to cover 90% of the top-k, when
    /// partitions are visited in the best possible order. Lower is better.
    double mean_partitions_for_90_recall{0.0};
};

UniformityMetrics
ComputeUniformity(const Partitions& partitions, uint64_t count, uint64_t bucket_size);

/// `neighbors` is the row-major ground-truth id matrix of `num_queries` x `ground_truth_k`.
/// At most `max_queries` queries are sampled (0 means all) and at most `topk` neighbours
/// per query are used (0 means all available).
LocalityMetrics
ComputeLocality(const Partitions& partitions,
                uint64_t count,
                const int64_t* neighbors,
                uint64_t num_queries,
                uint64_t ground_truth_k,
                uint64_t topk,
                uint64_t max_queries);

}  // namespace vsag::rpt_bench
