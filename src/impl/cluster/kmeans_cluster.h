
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

#include "impl/thread_pool/safe_thread_pool.h"
#include "typing.h"

namespace vsag {
class Allocator;
class KMeansCluster {
public:
    explicit KMeansCluster(int32_t dim,
                           Allocator* allocator,
                           SafeThreadPoolPtr thread_pool = nullptr);

    ~KMeansCluster();

    Vector<int>
    Run(uint32_t k,
        const float* datas,
        uint64_t count,
        int iter = 25,
        double* err = nullptr,
        bool use_mse_for_convergence = false,
        float threshold = 1e-6F);

public:
    float* k_centroids_{nullptr};

private:
    double
    find_nearest_one_with_blas(const float* query,
                               const uint64_t query_count,
                               const uint64_t k,
                               float* y_sqr,
                               float* distances,
                               Vector<int32_t>& labels);

    double
    find_nearest_one_with_hgraph(const float* query,
                                 const uint64_t query_count,
                                 const uint64_t k,
                                 Vector<int32_t>& labels);

private:
    Allocator* const allocator_{nullptr};

    SafeThreadPoolPtr thread_pool_{nullptr};

    const int32_t dim_{0};

    static constexpr uint64_t THRESHOLD_FOR_HGRAPH = 10000ULL;

    static constexpr uint64_t QUERY_BS = 65536ULL;
};

}  // namespace vsag
