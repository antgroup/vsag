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

#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace vsag::rpt_bench {

namespace {

uint64_t
CeilDiv(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

/// Gini coefficient of the partition sizes: 0 means all partitions are equal.
double
ComputeGini(std::vector<uint64_t> sizes) {
    if (sizes.empty()) {
        return 0.0;
    }
    std::sort(sizes.begin(), sizes.end());
    double n = static_cast<double>(sizes.size());
    double total = 0.0;
    double weighted = 0.0;
    for (uint64_t i = 0; i < sizes.size(); ++i) {
        total += static_cast<double>(sizes[i]);
        weighted += static_cast<double>(i + 1) * static_cast<double>(sizes[i]);
    }
    if (total <= 0.0) {
        return 0.0;
    }
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n;
}

}  // namespace

UniformityMetrics
ComputeUniformity(const Partitions& partitions, uint64_t count, uint64_t bucket_size) {
    UniformityMetrics metrics;
    metrics.num_partitions = partitions.size();
    metrics.expected_partitions = bucket_size == 0 ? 0 : CeilDiv(count, bucket_size);
    if (partitions.empty()) {
        return metrics;
    }

    std::vector<uint64_t> sizes;
    sizes.reserve(partitions.size());
    uint64_t total = 0;
    for (const auto& partition : partitions) {
        sizes.push_back(partition.size());
        total += partition.size();
    }
    metrics.max_size = *std::max_element(sizes.begin(), sizes.end());
    metrics.min_size = *std::min_element(sizes.begin(), sizes.end());
    metrics.max_min_ratio = metrics.min_size == 0 ? std::numeric_limits<double>::infinity()
                                                  : static_cast<double>(metrics.max_size) /
                                                        static_cast<double>(metrics.min_size);
    metrics.mean_size = static_cast<double>(total) / static_cast<double>(sizes.size());

    double variance = 0.0;
    for (uint64_t size : sizes) {
        double delta = static_cast<double>(size) - metrics.mean_size;
        variance += delta * delta;
    }
    metrics.stddev_size = std::sqrt(variance / static_cast<double>(sizes.size()));
    metrics.gini = ComputeGini(sizes);

    bool bounds =
        metrics.num_partitions == metrics.expected_partitions && metrics.max_size <= bucket_size;
    if (count >= bucket_size) {
        bounds = bounds && metrics.min_size >= bucket_size / 2 && metrics.max_min_ratio <= 2.0;
    }
    metrics.satisfies_bounds = bounds;
    return metrics;
}

LocalityMetrics
ComputeLocality(const Partitions& partitions,
                uint64_t count,
                const int64_t* neighbors,
                uint64_t num_queries,
                uint64_t ground_truth_k,
                uint64_t topk,
                uint64_t max_queries) {
    LocalityMetrics metrics;
    if (neighbors == nullptr || num_queries == 0 || ground_truth_k == 0 || partitions.empty()) {
        return metrics;
    }

    uint64_t k = topk == 0 ? ground_truth_k : std::min(topk, ground_truth_k);
    uint64_t queries = max_queries == 0 ? num_queries : std::min(max_queries, num_queries);
    metrics.k = k;

    std::vector<uint64_t> partition_of(count, 0);
    for (uint64_t p = 0; p < partitions.size(); ++p) {
        for (int64_t id : partitions[p]) {
            if (id >= 0 && static_cast<uint64_t>(id) < count) {
                partition_of[static_cast<uint64_t>(id)] = p;
            }
        }
    }

    double touched_total = 0.0;
    double same_partition_total = 0.0;
    double scans_total = 0.0;
    uint64_t evaluated = 0;

    std::unordered_map<uint64_t, uint64_t> counts;
    std::vector<uint64_t> ordered_counts;
    for (uint64_t q = 0; q < queries; ++q) {
        const int64_t* row = neighbors + q * ground_truth_k;
        counts.clear();
        uint64_t valid = 0;
        uint64_t top1_partition = 0;
        bool has_top1 = false;
        for (uint64_t i = 0; i < k; ++i) {
            int64_t id = row[i];
            if (id < 0 || static_cast<uint64_t>(id) >= count) {
                continue;
            }
            uint64_t partition = partition_of[static_cast<uint64_t>(id)];
            if (not has_top1) {
                top1_partition = partition;
                has_top1 = true;
            }
            ++counts[partition];
            ++valid;
        }
        if (valid == 0) {
            continue;
        }

        touched_total += static_cast<double>(counts.size());
        same_partition_total +=
            static_cast<double>(counts[top1_partition]) / static_cast<double>(valid);

        ordered_counts.clear();
        ordered_counts.reserve(counts.size());
        for (const auto& [partition, hits] : counts) {
            ordered_counts.push_back(hits);
        }
        std::sort(ordered_counts.begin(), ordered_counts.end(), std::greater<>());
        uint64_t target = CeilDiv(valid * 9, 10);
        uint64_t covered = 0;
        uint64_t scans = 0;
        for (uint64_t hits : ordered_counts) {
            covered += hits;
            ++scans;
            if (covered >= target) {
                break;
            }
        }
        scans_total += static_cast<double>(scans);
        ++evaluated;
    }

    if (evaluated == 0) {
        return metrics;
    }
    metrics.queries_evaluated = evaluated;
    metrics.mean_partitions_touched = touched_total / static_cast<double>(evaluated);
    metrics.top1_same_partition_rate = same_partition_total / static_cast<double>(evaluated);
    metrics.mean_partitions_for_90_recall = scans_total / static_cast<double>(evaluated);
    return metrics;
}

}  // namespace vsag::rpt_bench
