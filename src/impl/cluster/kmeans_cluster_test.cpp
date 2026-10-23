
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

#include "kmeans_cluster.h"

#include <cstdio>
#include <string>
#include <utility>

#include "gpu/cuda_backend.h"
#include "gpu/gpu_plan.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/logger/capturing_logger_test.h"
#include "unittest.h"
#include "vsag/logger.h"
#include "vsag/options.h"
std::vector<float>
GenerateDataset(int32_t k, int32_t dim, uint64_t count, std::vector<int>& labels) {
    std::vector<float> result(dim * count);
    labels.clear();
    labels.resize(k, 0);

    auto centroids = fixtures::generate_vectors(k, dim, /*normalize=*/true, /*seed=*/315);

    for (int64_t i = 0; i < count; ++i) {
        auto label = random() % k;
        for (int64_t j = 0; j < dim; ++j) {
            result[i * dim + j] = centroids[label * dim + j] + /*bias*/ 0.00001F;
        }
        labels[label]++;
    }
    std::sort(labels.begin(), labels.end());
    return result;
}

TEST_CASE("Kmeans Basic Test", "[ut][KMeansCluster]") {
    std::vector<int> labels;
    int32_t k = 10;
    int32_t dim = 3;
    uint64_t count = 2000;
    auto datas = GenerateDataset(k, dim, count, labels);

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    std::vector<int> new_labels(k);
    vsag::KMeansCluster cluster(dim, allocator.get());
    int iter = 0;
    while (iter < 500) {
        iter += 25;
        std::fill(new_labels.begin(), new_labels.end(), 0);
        auto pos = cluster.Run(k, datas.data(), count, iter, nullptr, false);
        for (int i = 0; i < count; ++i) {
            new_labels[pos[i]]++;
        }
        std::sort(new_labels.begin(), new_labels.end());
        if (new_labels[0] != 0) {
            for (int i = 0; i < k; ++i) {
                REQUIRE(new_labels[i] == labels[i]);
            }
            break;
        }
    }
}

// Exercises the centroid-assignment path with shape parameters that meet the
// AMX-BF16 fast-path thresholds in `find_nearest_one_with_blas` (k >= 16,
// dim >= 32, query batches >= 16).  On hosts without AMX-BF16 support, the
// kernel returns false and the SGEMM path is used; either way the test
// verifies KMeans converges to the cluster sizes implied by the synthetic
// dataset.
TEST_CASE("Kmeans Larger Dim (AMX BF16 path)", "[ut][KMeansCluster]") {
    std::vector<int> labels;
    int32_t k = 32;
    int32_t dim = 128;
    uint64_t count = 3000;
    auto datas = GenerateDataset(k, dim, count, labels);

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    std::vector<int> new_labels(k);
    vsag::KMeansCluster cluster(dim, allocator.get());
    int iter = 0;
    bool converged = false;
    while (iter < 500) {
        iter += 25;
        std::fill(new_labels.begin(), new_labels.end(), 0);
        auto pos = cluster.Run(k, datas.data(), count, iter, nullptr, false);
        for (uint64_t i = 0; i < count; ++i) {
            new_labels[pos[i]]++;
        }
        std::sort(new_labels.begin(), new_labels.end());
        if (new_labels[0] != 0) {
            for (int i = 0; i < k; ++i) {
                REQUIRE(new_labels[i] == labels[i]);
            }
            converged = true;
            break;
        }
    }
    REQUIRE(converged);
}

TEST_CASE("Kmeans seeded fixed-order reduction is reproducible", "[ut][KMeansCluster]") {
    constexpr uint32_t k = 8;
    constexpr int32_t dim = 17;
    constexpr uint64_t count = 4097;
    std::vector<float> data(count * dim);
    for (uint64_t i = 0; i < count; ++i) {
        for (int32_t d = 0; d < dim; ++d) {
            data[i * dim + d] =
                static_cast<float>(i % k) * 5.0F +
                static_cast<float>((i * 31 + static_cast<uint64_t>(d) * 17) % 97) * 0.0001F;
        }
    }

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto single_thread_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    single_thread_pool->SetPoolSize(1);
    auto multi_thread_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    multi_thread_pool->SetPoolSize(4);
    vsag::KMeansCluster single_thread(dim, allocator.get(), single_thread_pool);
    vsag::KMeansCluster multi_thread(dim, allocator.get(), multi_thread_pool);

    const auto single_thread_labels = single_thread.Run(k,
                                                        data.data(),
                                                        count,
                                                        6,
                                                        nullptr,
                                                        false,
                                                        1e-6F,
                                                        vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                                        0x52425131U,
                                                        true);
    const auto multi_thread_labels = multi_thread.Run(k,
                                                      data.data(),
                                                      count,
                                                      6,
                                                      nullptr,
                                                      false,
                                                      1e-6F,
                                                      vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                                      0x52425131U,
                                                      true);
    REQUIRE(single_thread_labels == multi_thread_labels);
    const uint64_t centroid_values = static_cast<uint64_t>(k) * dim;
    REQUIRE(std::equal(single_thread.k_centroids_,
                       single_thread.k_centroids_ + centroid_values,
                       multi_thread.k_centroids_));
}

TEST_CASE("Kmeans says why it stayed on the host", "[ut][KMeansCluster]") {
    // A run that asked for a device and did not get one is otherwise
    // indistinguishable from one that never asked, so the reason is traced. The
    // two it reports are the two a caller can act on: lower the threshold, or fix
    // the ordinal. Holds in both build configurations, since neither case here
    // needs a device.
    vsag::testing::CapturingLogger capture;
    const vsag::testing::ScopedLogger installed(&capture);

    constexpr uint32_t k = 8;
    constexpr int32_t dim = 16;
    constexpr uint64_t count = 2048;
    const std::vector<float> data(count * dim, 0.5F);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto reason_for = [&](vsag::KMeansGpuConfig config) {
        capture.lines.clear();
        vsag::KMeansCluster cluster(dim, allocator.get(), nullptr, config);
        cluster.Run(k,
                    data.data(),
                    count,
                    1,
                    nullptr,
                    false,
                    1e-6F,
                    vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                    1U,
                    true);
        for (const auto& line : capture.lines) {
            if (line.find("stayed on the host") != std::string::npos) {
                return line;
            }
        }
        return std::string{};
    };

    vsag::KMeansGpuConfig below_threshold;
    below_threshold.enabled = true;
    const auto below = reason_for(below_threshold);
    REQUIRE(below.find("below the offload threshold") != std::string::npos);

    vsag::KMeansGpuConfig absent_device;
    absent_device.enabled = true;
    absent_device.min_work_threshold = 1;
    absent_device.device_id = 4096;
    const auto absent = reason_for(absent_device);
    REQUIRE(absent.find("could not be bound") != std::string::npos);
    REQUIRE(absent.find("4096") != std::string::npos);

    // Nothing to report when the backend was never asked for.
    REQUIRE(reason_for(vsag::KMeansGpuConfig{}).empty());
}

TEST_CASE("Kmeans bounds the resident pass by a named budget only", "[ut][KMeansCluster]") {
    // A budget the caller names has to reach the seeding pass, which holds its
    // working set resident; the default ceiling must not, or a large training set
    // would be sent back to the host for no reason. Both budgets are traced, which
    // is the only place they are observable, since what Run returns is the same
    // either way.
    if (not vsag::gpu::CudaAvailable()) {
        SKIP("no CUDA device: nothing binds, and both budgets stay 0");
    }
    vsag::testing::CapturingLogger capture;
    const vsag::testing::ScopedLogger installed(&capture);

    constexpr uint32_t k = 8;
    constexpr int32_t dim = 16;
    constexpr uint64_t count = 2048;
    const std::vector<float> data(count * dim, 0.25F);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    // Returns {resident MiB, chunked MiB} as the session reported them.
    auto budgets_for = [&](uint64_t memory_budget) {
        capture.lines.clear();
        vsag::KMeansGpuConfig config;
        config.enabled = true;
        config.min_work_threshold = 1;
        config.memory_budget = memory_budget;
        vsag::KMeansCluster cluster(dim, allocator.get(), nullptr, config);
        cluster.Run(k,
                    data.data(),
                    count,
                    1,
                    nullptr,
                    false,
                    1e-6F,
                    vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                    1U,
                    true);
        std::pair<uint64_t, uint64_t> seen{0, 0};
        for (const auto& line : capture.lines) {
            if (line.find("resident budget") != std::string::npos) {
                unsigned long long resident = 0;
                unsigned long long chunked = 0;
                if (std::sscanf(line.c_str() + line.find("resident budget"),
                                "resident budget %llu MiB, chunked budget %llu MiB",
                                &resident,
                                &chunked) == 2) {
                    seen = {resident, chunked};
                }
            }
        }
        return seen;
    };

    constexpr uint64_t kNamed = 64ULL << 20;
    const auto named = budgets_for(kNamed);
    REQUIRE(named.first == 64);
    REQUIRE(named.second == 64);

    // With nothing named the chunked passes take the default ceiling while the
    // resident one takes what the card has free.
    const auto derived = budgets_for(0);
    const uint64_t cap_mib = vsag::gpu::kChunkedBudgetCap >> 20;
    if (derived.first <= cap_mib) {
        // The card has no more free than the default ceiling, so both budgets are
        // its share and there is nothing to tell apart. Checked rather than
        // assumed, because a card shared with other work reaches this state and a
        // test of the ceiling must not read as a failure there.
        REQUIRE(derived.second == derived.first);
        SKIP("the device has " + std::to_string(derived.first) +
             " MiB for the resident pass, at or below the " + std::to_string(cap_mib) +
             " MiB default ceiling, so the two budgets cannot differ");
    }
    REQUIRE(derived.second == cap_mib);
    REQUIRE(derived.first > derived.second);
}

TEST_CASE("Kmeans keeps small runs on the CPU whatever the GPU config says",
          "[ut][KMeansCluster]") {
    // Turning the backend on must not change what a run returns. Below the
    // threshold nothing binds and every pass declines, so the run is the host's,
    // in both build configurations. That is what makes the setting safe to leave
    // on in a configuration file that may travel to a machine without a card.
    //
    // Two things it does not show. The threshold itself is not checked here, and
    // is pinned in gpu_plan_test.cpp instead. Nor is the random source: with
    // nothing bound both arms take the identical path, so a pass that drew from
    // the caller's generator before declining would advance both equally and go
    // unseen.
    constexpr uint32_t k = 8;
    constexpr int32_t dim = 17;
    constexpr uint64_t count = 4097;
    std::vector<float> data(count * dim);
    for (uint64_t i = 0; i < count; ++i) {
        for (int32_t d = 0; d < dim; ++d) {
            data[i * dim + d] =
                static_cast<float>(i % k) * 5.0F +
                static_cast<float>((i * 31 + static_cast<uint64_t>(d) * 17) % 97) * 0.0001F;
        }
    }
    // count * k * dim is 557192, far below the calibrated default.
    REQUIRE(count * k * dim < (1ULL << 32));

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::KMeansGpuConfig gpu_config;
    gpu_config.enabled = true;

    vsag::KMeansCluster off(dim, allocator.get());
    vsag::KMeansCluster on(dim, allocator.get(), nullptr, gpu_config);

    const auto without = off.Run(k,
                                 data.data(),
                                 count,
                                 6,
                                 nullptr,
                                 false,
                                 1e-6F,
                                 vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                 0x52425131U,
                                 true);
    const auto with = on.Run(k,
                             data.data(),
                             count,
                             6,
                             nullptr,
                             false,
                             1e-6F,
                             vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                             0x52425131U,
                             true);
    REQUIRE(with == without);
}

TEST_CASE("Kmeans with the backend on still clusters", "[ut][KMeansCluster]") {
    // Lowering the threshold lets the device take the run where there is one.
    //
    // The assertions are about the clustering, not about equality with a host
    // run. Measured on this tree, the device seeding reaches the same picks as
    // the host routine from the same generator, so equality would hold without
    // showing that the device did anything; and it is not equality the pass
    // promises, since the accumulation sums with atomics in no fixed order.
    // What has to hold is that the run finishes, separates the clusters it was
    // given, and reports an error consistent with having done so.
    constexpr uint32_t k = 8;
    constexpr int32_t dim = 16;
    constexpr uint64_t per_cluster = 512;
    constexpr uint64_t count = k * per_cluster;

    // Well separated clusters, so which one a point belongs to is not in doubt
    // and the run has a single sensible answer to be checked against.
    std::vector<float> data(count * dim);
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t cluster_id = i / per_cluster;
        for (int32_t d = 0; d < dim; ++d) {
            data[i * dim + d] =
                (static_cast<uint64_t>(d) == cluster_id ? 100.0F : 0.0F) +
                static_cast<float>((i * 31 + static_cast<uint64_t>(d)) % 13) * 0.01F;
        }
    }

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::KMeansGpuConfig gpu_config;
    gpu_config.enabled = true;
    gpu_config.min_work_threshold = 1;
    vsag::KMeansCluster cluster(dim, allocator.get(), nullptr, gpu_config);

    double error = 0.0;
    const auto labels = cluster.Run(k, data.data(), count, 25, &error);
    REQUIRE(labels.size() == count);
    for (const auto label : labels) {
        REQUIRE(label >= 0);
        REQUIRE(label < static_cast<int>(k));
    }
    // Points of one cluster all land in the same bucket, and distinct clusters
    // land in distinct ones.
    std::vector<int> per_cluster_label;
    for (uint32_t c = 0; c < k; ++c) {
        const int first = labels[c * per_cluster];
        for (uint64_t i = 0; i < per_cluster; ++i) {
            REQUIRE(labels[c * per_cluster + i] == first);
        }
        per_cluster_label.push_back(first);
    }
    std::sort(per_cluster_label.begin(), per_cluster_label.end());
    REQUIRE(std::unique(per_cluster_label.begin(), per_cluster_label.end()) ==
            per_cluster_label.end());

    // The error is the mean squared distance to the assigned centroid, computed
    // by expansion, so on points that sit near their centroid it can land a
    // rounding step either side of the small value it should have. The scatter
    // within a cluster is under 0.16^2 per dimension.
    REQUIRE(error < 1.0);
    REQUIRE(error > -1e-3);
}
