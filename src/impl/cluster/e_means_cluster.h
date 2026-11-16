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

#include <optional>

#include "impl/thread_pool/safe_thread_pool.h"
#include "typing.h"

namespace vsag {
class Allocator;

/// Mini-batch k-means with exponential-moving-average (EMA) centroid updates.
///
/// Implements e-means from Bessa et al., "e-means: outperforming k-means (in
/// one-tenth the epochs)", arXiv:2611.09501: EMA updates of per-cluster counts
/// and sums, a cosine-annealed step size, dead-cluster revival by splitting,
/// last-epoch-mean finalization, and the paper's scaling laws that derive the
/// batch size and peak step from (n, k, epochs) when they are not given.
///
/// It reaches the squared-L2 loss of Lloyd k-means with far fewer passes over
/// the data. Seeding is uniformly random; the result depends on `random_seed`
/// and on the (random) batch order, but is deterministic for a fixed seed.
class EMeansCluster {
public:
    explicit EMeansCluster(int32_t dim,
                           Allocator* allocator,
                           SafeThreadPoolPtr thread_pool = nullptr);

    ~EMeansCluster();

    EMeansCluster(const EMeansCluster&) = delete;
    EMeansCluster&
    operator=(const EMeansCluster&) = delete;

    /// Clusters `datas` (`count` row-major vectors of `dim`) into `k` clusters.
    ///
    /// @param epochs number of passes over the data (one epoch = count / batch
    ///        updates); 3 to 5 usually matches Lloyd k-means at 25 iterations.
    /// @param err if not null, receives the mean squared L2 error per vector
    ///        of the returned assignment (the same quantity as KMeansCluster).
    /// @param batch_size mini-batch size; 0 picks it by the paper's batch law.
    /// @param peak_step peak EMA step alpha_0 in (0, 1]; <= 0 picks it by the
    ///        paper's step law.
    /// @return the nearest-centroid label of every vector; centroids are left
    ///         in `k_centroids_`.
    Vector<int>
    Run(uint32_t k,
        const float* datas,
        uint64_t count,
        int epochs = 3,
        double* err = nullptr,
        std::optional<uint32_t> random_seed = std::nullopt,
        uint64_t batch_size = 0,
        double peak_step = 0.0);

    /// Batch size from the paper's batch law, rounded to a power of two and
    /// capped at `count`.
    static uint64_t
    DefaultBatchSize(uint64_t count, uint32_t k, int epochs);

    /// Peak step alpha_0 from the paper's step law, capped at 1.
    static double
    DefaultPeakStep(uint64_t count, uint32_t k, int epochs, uint64_t batch_size);

public:
    float* k_centroids_{nullptr};

private:
    Allocator* const allocator_{nullptr};

    SafeThreadPoolPtr thread_pool_{nullptr};

    const int32_t dim_{0};
};

}  // namespace vsag
