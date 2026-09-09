
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

#include <cstring>
#include <stdexcept>

#include "impl/allocator/safe_allocator.h"
#include "simd/fp32_simd.h"
#include "unittest.h"

namespace {

// Reference implementation of the original serial initialization, independent of the pool.
std::vector<float>
SerialKMeansPlusPlus(const float* data, uint64_t count, uint64_t dim, uint32_t k, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint64_t> row_dis(0, count - 1);
    std::vector<float> centers(uint64_t{k} * dim);
    std::copy_n(data + row_dis(gen) * dim, dim, centers.data());
    std::vector<float> weights(count, std::numeric_limits<float>::max());
    for (uint32_t c = 1; c < k; ++c) {
        for (uint64_t i = 0; i < count; ++i) {
            const auto distance =
                vsag::FP32ComputeL2Sqr(data + i * dim, centers.data() + (c - 1) * dim, dim);
            weights[i] = std::min(weights[i], distance);
        }
        double total = 0.0;
        for (const auto weight : weights) {
            total += weight;
        }
        uint64_t selected = count - 1;
        if (total <= 0.0) {
            selected = row_dis(gen);
        } else {
            std::uniform_real_distribution<double> prob_dis(0.0, total);
            const auto threshold = prob_dis(gen);
            double cumulative = 0.0;
            for (uint64_t i = 0; i < count; ++i) {
                cumulative += weights[i];
                if (cumulative >= threshold) {
                    selected = i;
                    break;
                }
            }
        }
        std::copy_n(data + selected * dim, dim, centers.data() + uint64_t{c} * dim);
    }
    return centers;
}

class CountingKMeansPool : public vsag::DefaultThreadPool {
public:
    explicit CountingKMeansPool(uint64_t threads) : vsag::DefaultThreadPool(threads) {
    }

    std::future<void>
    Enqueue(std::function<void()> task) override {
        ++submitted;
        if (submitted == fail_on) {
            if (drop_task) {
                // Destroying the unexecuted packaged task makes its result a broken promise.
                std::promise<void> done;
                done.set_value();
                return done.get_future();
            }
            throw std::runtime_error("injected KMeans enqueue failure");
        }
        return vsag::DefaultThreadPool::Enqueue(std::move(task));
    }

    uint64_t submitted{0};
    uint64_t fail_on{0};
    bool drop_task{false};
};

void
RunInitialization(
    vsag::KMeansCluster& cluster, uint32_t k, const float* data, uint64_t count, uint32_t seed) {
    // Zero Lloyd iterations exposes just the KMeans++ centers without adding a test-only API.
    cluster.Run(
        k, data, count, 0, nullptr, false, 1e-6F, vsag::KMeansInitMethod::KMEANS_PLUS_PLUS, seed);
}

}  // namespace

TEST_CASE("Parallel KMeans++ matches serial initialization bit for bit",
          "[ut][KMeansCluster][kmeans_init]") {
    const int32_t dim = GENERATE(1, 17, 64);
    const uint64_t count = GENERATE(1, 4096, 4097, 16387);
    const uint64_t threads = GENERATE(1, 4);
    const uint32_t seed = GENERATE(7U, 0x52425131U);
    const auto k = static_cast<uint32_t>(std::min(uint64_t{17}, count));
    const auto data = fixtures::generate_vectors(count, dim, false, 42);
    const auto expected = SerialKMeansPlusPlus(data.data(), count, dim, k, seed);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<CountingKMeansPool>(threads);
    auto safe_pool = std::make_shared<vsag::SafeThreadPool>(pool);
    vsag::KMeansCluster cluster(dim, allocator.get(), safe_pool);
    RunInitialization(cluster, k, data.data(), count, seed);
    REQUIRE(std::memcmp(cluster.k_centroids_, expected.data(), expected.size() * sizeof(float)) ==
            0);
    const auto block_size = std::max(uint64_t{4096}, uint64_t{65536} / dim);
    const auto expected_tasks = count > block_size ? (1 + (count - 1) / block_size) * (k - 1) : 0;
    REQUIRE(pool->submitted == expected_tasks);
}

TEST_CASE("Parallel KMeans++ preserves zero-weight fallback and drains failed tasks",
          "[ut][KMeansCluster][kmeans_init]") {
    constexpr int32_t dim = 17;
    constexpr uint64_t count = 16387;
    constexpr uint32_t k = 9;
    constexpr uint32_t seed = 123;
    const uint32_t unique = GENERATE(1, 3);
    std::vector<float> data(count * dim);
    for (uint64_t row = 0; row < count; ++row) {
        std::fill_n(data.data() + row * dim, dim, static_cast<float>(row % unique));
    }
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<CountingKMeansPool>(4);
    auto safe_pool = std::make_shared<vsag::SafeThreadPool>(pool);
    vsag::KMeansCluster cluster(dim, allocator.get(), safe_pool);
    SECTION("normal initialization") {
    }
    SECTION("enqueue fails after one task was submitted") {
        pool->fail_on = 2;
        REQUIRE_THROWS(RunInitialization(cluster, k, data.data(), count, seed));
    }
    SECTION("one submitted task has a broken promise") {
        pool->fail_on = 2;
        pool->drop_task = true;
        REQUIRE_THROWS(RunInitialization(cluster, k, data.data(), count, seed));
    }
    // Also verify the same cluster and pool can be reused after a failed initialization.
    RunInitialization(cluster, k, data.data(), count, seed);
    const auto expected = SerialKMeansPlusPlus(data.data(), count, dim, k, seed);
    REQUIRE(std::memcmp(cluster.k_centroids_, expected.data(), expected.size() * sizeof(float)) ==
            0);
    pool->WaitUntilEmpty();
}

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

TEST_CASE("Full KMeans uses all rows and returns final exact assignments",
          "[ut][KMeansCluster][fused_full]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::KMeansCluster cluster(1, allocator.get());
    std::vector<float> data(65537, 0.0F);
    data.back() = 65537.0F;
    auto labels = cluster.RunFull(1, data.data(), data.size(), 1);
    REQUIRE(cluster.k_centroids_[0] == 1.0F);
    REQUIRE(labels.size() == data.size());
    REQUIRE(std::all_of(labels.begin(), labels.end(), [](auto id) { return id == 0; }));
    REQUIRE_THROWS(cluster.RunFull(0, data.data(), data.size()));
    REQUIRE_THROWS(cluster.RunFull(2, data.data(), 1));
    REQUIRE_THROWS(cluster.RunFull(1, data.data(), data.size(), 0));

    data = {0, 1, 2, 3, 10, 20, 21, 22, 40, 41, 43};
    labels = cluster.RunFull(3, data.data(), data.size(), 2);
    for (uint64_t row = 0; row < data.size(); ++row) {
        float best = std::numeric_limits<float>::max();
        int32_t nearest = 0;
        for (int32_t id = 0; id < 3; ++id) {
            const auto diff = data[row] - cluster.k_centroids_[id];
            if (diff * diff < best) {
                best = diff * diff;
                nearest = id;
            }
        }
        REQUIRE(labels[row] == nearest);
    }
}

TEST_CASE("Full KMeans supports ten thousand centers without approximate routing",
          "[ut][KMeansCluster][fused_full][large_k]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::KMeansCluster cluster(1, allocator.get());
    constexpr uint32_t k = 10000;
    std::vector<float> data(k);
    std::iota(data.begin(), data.end(), 0.0F);
    auto labels = cluster.RunFull(k, data.data(), data.size(), 1);
    for (uint64_t row = 0; row < data.size(); ++row) {
        REQUIRE(cluster.k_centroids_[labels[row]] == data[row]);
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

TEST_CASE("Full KMeans is reproducible across worker counts", "[ut][KMeansCluster][fused_full]") {
    constexpr uint64_t count = 4097;
    constexpr uint32_t k = 33;
    constexpr int32_t dim = 17;
    auto data = fixtures::generate_vectors(count, dim, false, 42);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto single_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    auto multi_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    single_pool->SetPoolSize(1);
    multi_pool->SetPoolSize(4);
    vsag::KMeansCluster single(dim, allocator.get(), single_pool);
    vsag::KMeansCluster multi(dim, allocator.get(), multi_pool);
    REQUIRE(single.RunFull(k, data.data(), count, 3) == multi.RunFull(k, data.data(), count, 3));
    REQUIRE(std::equal(
        single.k_centroids_, single.k_centroids_ + uint64_t{k} * dim, multi.k_centroids_));
}
