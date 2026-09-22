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

#include "partitioners.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <string>

#include "impl/allocator/safe_allocator.h"
#include "impl/cluster/kmeans_cluster.h"
#include "vsag/vsag.h"

namespace vsag::rpt_bench {

namespace {

/// Current high-water resident set size of the process, in KiB, or 0 when unavailable.
int64_t
ReadPeakRssKb() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmHWM:") {
            int64_t value = 0;
            status >> value;
            return value;
        }
        status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
}

uint64_t
CeilDiv(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

/// Cuts an ordered id list into `leaves` chunks whose sizes differ by at most one.
Partitions
CutIntoChunks(const std::vector<int64_t>& ids, uint64_t leaves) {
    Partitions partitions;
    partitions.reserve(leaves);
    uint64_t count = ids.size();
    uint64_t quotient = count / leaves;
    uint64_t remainder = count % leaves;
    uint64_t cursor = 0;
    for (uint64_t i = 0; i < leaves; ++i) {
        uint64_t chunk = quotient + (i < remainder ? 1 : 0);
        partitions.emplace_back(ids.begin() + static_cast<int64_t>(cursor),
                                ids.begin() + static_cast<int64_t>(cursor + chunk));
        cursor += chunk;
    }
    return partitions;
}

/// Runs `body` while measuring wall time and the growth of the peak RSS.
template <typename Body>
PartitionResult
Measure(Body&& body) {
    PartitionResult result;
    int64_t rss_before = ReadPeakRssKb();
    auto start = std::chrono::steady_clock::now();
    body(result);
    auto end = std::chrono::steady_clock::now();
    result.build_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.peak_rss_delta_kb = ReadPeakRssKb() - rss_before;
    return result;
}

}  // namespace

PartitionResult
PartitionByRpt(const PartitionRequest& request) {
    return Measure([&](PartitionResult& result) {
        RPTPartitionParams params;
        params.bucket_size = request.bucket_size;
        params.seed = request.seed;
        auto partitions = rpt_partition(request.dim, request.count, request.data, params);
        if (not partitions.has_value()) {
            result.error = partitions.error().message;
            return;
        }
        result.partitions = std::move(partitions.value());
    });
}

PartitionResult
PartitionByRandom(const PartitionRequest& request) {
    return Measure([&](PartitionResult& result) {
        std::vector<int64_t> ids(request.count);
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937_64 generator(request.seed);
        std::shuffle(ids.begin(), ids.end(), generator);
        result.partitions = CutIntoChunks(ids, CeilDiv(request.count, request.bucket_size));
    });
}

PartitionResult
PartitionBySingleDim(const PartitionRequest& request) {
    return Measure([&](PartitionResult& result) {
        // Pick the dimension with the largest variance, then sort the ids along it.
        std::vector<double> sums(request.dim, 0.0);
        std::vector<double> square_sums(request.dim, 0.0);
        for (uint64_t i = 0; i < request.count; ++i) {
            const float* vector = request.data + i * request.dim;
            for (uint64_t j = 0; j < request.dim; ++j) {
                sums[j] += vector[j];
                square_sums[j] += static_cast<double>(vector[j]) * vector[j];
            }
        }
        uint64_t best_dim = 0;
        double best_variance = -1.0;
        for (uint64_t j = 0; j < request.dim; ++j) {
            double mean = sums[j] / static_cast<double>(request.count);
            double variance = square_sums[j] / static_cast<double>(request.count) - mean * mean;
            if (variance > best_variance) {
                best_variance = variance;
                best_dim = j;
            }
        }

        std::vector<int64_t> ids(request.count);
        std::iota(ids.begin(), ids.end(), 0);
        const float* data = request.data;
        uint64_t dim = request.dim;
        std::sort(ids.begin(), ids.end(), [data, dim, best_dim](int64_t a, int64_t b) {
            float left = data[static_cast<uint64_t>(a) * dim + best_dim];
            float right = data[static_cast<uint64_t>(b) * dim + best_dim];
            if (left != right) {
                return left < right;
            }
            return a < b;
        });
        result.partitions = CutIntoChunks(ids, CeilDiv(request.count, request.bucket_size));
    });
}

PartitionResult
PartitionByKmeans(const PartitionRequest& request) {
    return Measure([&](PartitionResult& result) {
        uint64_t clusters = CeilDiv(request.count, request.bucket_size);
        if (clusters > request.count) {
            result.error = "bucket_size yields more clusters than vectors";
            return;
        }
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        KMeansCluster cluster(static_cast<int32_t>(request.dim), allocator.get());
        auto labels =
            cluster.Run(static_cast<uint32_t>(clusters), request.data, request.count, /*iter=*/25);

        Partitions partitions(clusters);
        for (uint64_t i = 0; i < request.count; ++i) {
            int label = labels[i];
            if (label >= 0 && static_cast<uint64_t>(label) < clusters) {
                partitions[static_cast<uint64_t>(label)].push_back(static_cast<int64_t>(i));
            }
        }
        // KMeans may leave clusters empty; those are not partitions.
        partitions.erase(std::remove_if(partitions.begin(),
                                        partitions.end(),
                                        [](const std::vector<int64_t>& p) { return p.empty(); }),
                         partitions.end());
        result.partitions = std::move(partitions);
    });
}

PartitionResult
RunStrategy(const std::string& name, const PartitionRequest& request) {
    if (name == "rpt") {
        return PartitionByRpt(request);
    }
    if (name == "random") {
        return PartitionByRandom(request);
    }
    if (name == "single_dim") {
        return PartitionBySingleDim(request);
    }
    if (name == "kmeans") {
        return PartitionByKmeans(request);
    }
    PartitionResult result;
    result.error = "unknown strategy: " + name;
    return result;
}

std::vector<std::string>
StrategyNames() {
    return {"rpt", "random", "single_dim", "kmeans"};
}

}  // namespace vsag::rpt_bench
