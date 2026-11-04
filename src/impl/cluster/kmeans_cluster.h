
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
#include <random>

#include "impl/thread_pool/safe_thread_pool.h"
#include "typing.h"

namespace vsag {
class Allocator;

enum class KMeansInitMethod {
    RANDOM,
    KMEANS_PLUS_PLUS,
};

/// Opt-in settings for the CUDA build backend; the defaults leave it off.
///
/// IVFPartitionStrategyParameters carries the same four under their JSON names.
/// They are two structs because this layer cannot depend on the IVF parameters,
/// and ivf_nearest_partition.cpp copies them across in four adjacent lines so
/// that there is one place to change rather than several.
struct KMeansGpuConfig {
    /// Allow the CUDA backend during training. Off unless asked for.
    bool enabled{false};
    /// CUDA device ordinal. One the machine does not have keeps the run on the CPU
    /// rather than falling through to another device.
    int32_t device_id{0};
    /// Ceiling in bytes on what any pass asks for; 0 derives it from what the
    /// device has free and leaves the chunked passes their own default ceiling.
    uint64_t memory_budget{0};
    /// Smallest `count * k * dim` worth offloading; 0 uses the calibrated default.
    uint64_t min_work_threshold{0};
};

class KMeansCluster {
public:
    explicit KMeansCluster(int32_t dim,
                           Allocator* allocator,
                           SafeThreadPoolPtr thread_pool = nullptr,
                           KMeansGpuConfig gpu_config = {});

    ~KMeansCluster();

    Vector<int>
    Run(uint32_t k,
        const float* datas,
        uint64_t count,
        int iter = 25,
        double* err = nullptr,
        bool use_mse_for_convergence = false,
        float threshold = 1e-6F,
        KMeansInitMethod init_method = KMeansInitMethod::KMEANS_PLUS_PLUS,
        std::optional<uint32_t> random_seed = std::nullopt,
        bool deterministic_reduction = false);

public:
    float* k_centroids_{nullptr};

private:
    double
    find_nearest_one_with_hgraph(const float* query,
                                 const uint64_t query_count,
                                 const uint64_t k,
                                 Vector<int32_t>& labels);

    void
    select_initial_centroids_random(const float* datas,
                                    uint64_t count,
                                    uint32_t k,
                                    std::mt19937& gen);

    void
    select_initial_centroids_kmeans_plus_plus(const float* datas,
                                              uint64_t count,
                                              uint32_t k,
                                              std::mt19937& gen);

private:
    Allocator* const allocator_{nullptr};

    SafeThreadPoolPtr thread_pool_{nullptr};

    const int32_t dim_{0};

    const KMeansGpuConfig gpu_config_{};

    static constexpr uint64_t THRESHOLD_FOR_HGRAPH = 10000ULL;
};

}  // namespace vsag
