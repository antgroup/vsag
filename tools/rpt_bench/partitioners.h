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
#include <string>
#include <vector>

namespace vsag::rpt_bench {

using Partitions = std::vector<std::vector<int64_t>>;

struct PartitionResult {
    Partitions partitions;
    double build_time_ms{0.0};

    /// Growth of the process high-water RSS across the call, in KiB. Because the kernel
    /// only tracks a high-water mark, this is meaningful only for the first strategy that
    /// reaches a given peak; run one strategy per process for exact per-strategy numbers.
    int64_t peak_rss_delta_kb{0};

    std::string error;
};

struct PartitionRequest {
    uint64_t dim{0};
    uint64_t count{0};
    const float* data{nullptr};
    uint64_t bucket_size{0};
    uint64_t seed{0};
};

/// Random projection tree partitioning through the public vsag::rpt_partition API.
PartitionResult
PartitionByRpt(const PartitionRequest& request);

/// Baseline: shuffle the ids, then cut into equal-sized chunks.
PartitionResult
PartitionByRandom(const PartitionRequest& request);

/// Baseline: sort the ids by the highest-variance dimension, then cut into equal-sized chunks.
PartitionResult
PartitionBySingleDim(const PartitionRequest& request);

/// Baseline: KMeans with ceil(count / bucket_size) clusters. Partition sizes are unbounded.
PartitionResult
PartitionByKmeans(const PartitionRequest& request);

/// Runs the strategy registered under `name`, or reports an error in the result.
PartitionResult
RunStrategy(const std::string& name, const PartitionRequest& request);

/// Names of every strategy `RunStrategy` accepts.
std::vector<std::string>
StrategyNames();

}  // namespace vsag::rpt_bench
