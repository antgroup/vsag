
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

#include <random>

#include "impl/thread_pool/safe_thread_pool.h"
#include "kmeans_cluster.h"
#include "typing.h"

namespace vsag {

class Allocator;

// K-medoids clustering: same alternating scheme as `KMeansCluster`, but every
// center is constrained to be one of the input data points (a medoid).
//
// The medoid update relies on the identity
//     sum_j ||x_i - x_j||^2 = n_c * ||x_i - mean||^2 + const
// so the exact squared-L2 medoid of a cluster is its member closest to the
// cluster mean, which keeps every update exact and linear in the cluster size.
//
// Note that `seed == 0` selects a non-deterministic seed from
// `std::random_device` (matching the default behavior of `KMeansCluster`);
// pass a non-zero value to obtain reproducible results.
class KMedoidsCluster {
public:
    explicit KMedoidsCluster(int32_t dim,
                             Allocator* allocator,
                             SafeThreadPoolPtr thread_pool = nullptr);

    ~KMedoidsCluster();

    // Clusters `count` vectors (`datas`, row-major, `dim_` floats per row) into
    // `k` groups and returns the label of every input row. After the call the
    // selected medoids are available through `k_centroids_` (k x dim, row-major,
    // drop-in compatible with `KMeansCluster::k_centroids_`) and their row
    // indices in `datas` through `medoid_ids_`.
    //
    // The loop stops early either when the medoid set is stable or when the
    // absolute change of the error between two iterations falls below
    // `threshold`.
    Vector<int>
    Run(uint32_t k,
        const float* datas,
        uint64_t count,
        int iter = 10,
        double* err = nullptr,
        float threshold = 1e-6F,
        KMeansInitMethod init_method = KMeansInitMethod::KMEANS_PLUS_PLUS,
        uint64_t seed = 0);

public:
    float* k_centroids_{nullptr};

    Vector<int64_t> medoid_ids_;

private:
    void
    select_initial_medoids_random(const float* datas,
                                  uint64_t count,
                                  uint32_t k,
                                  std::mt19937& gen);

    void
    select_initial_medoids_kmeans_plus_plus(const float* datas,
                                            uint64_t count,
                                            uint32_t k,
                                            std::mt19937& gen);

    // Rebuilds every medoid from its cluster and returns true when at least
    // one medoid changed.
    bool
    update_medoids(const float* datas, uint64_t count, const Vector<int32_t>& labels);

private:
    Allocator* const allocator_{nullptr};

    SafeThreadPoolPtr thread_pool_{nullptr};

    const int32_t dim_{0};

    static constexpr uint64_t CLUSTER_BATCH = 64ULL;
};

}  // namespace vsag
