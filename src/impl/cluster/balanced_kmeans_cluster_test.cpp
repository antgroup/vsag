
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

#include "balanced_kmeans_cluster.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "allocator/memory_record_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

namespace {

/// Well separated spherical clusters of equal size, with the vector order
/// shuffled. `truth[i]` is the generating cluster of vector i. The centroid
/// axes are spaced far apart relative to the noise, so the nearest-centroid
/// answer is never in doubt.
std::vector<float>
make_separated(uint32_t k,
               uint64_t per_cluster,
               int32_t dim,
               std::mt19937& gen,
               std::vector<uint32_t>& truth) {
    REQUIRE(k <= static_cast<uint32_t>(dim));
    const uint64_t n = static_cast<uint64_t>(k) * per_cluster;
    std::vector<float> centroids(static_cast<uint64_t>(k) * dim, 0.0F);
    for (uint32_t c = 0; c < k; ++c) {
        centroids[static_cast<uint64_t>(c) * dim + c] = 1000.0F * (1.0F + static_cast<float>(c));
    }
    std::normal_distribution<float> noise(0.0F, 0.01F);
    std::vector<float> data(n * static_cast<uint64_t>(dim));
    truth.resize(n);
    for (uint64_t i = 0; i < n; ++i) {
        truth[i] = static_cast<uint32_t>(i / per_cluster);
    }
    std::shuffle(truth.begin(), truth.end(), gen);
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t c = truth[i];
        for (int32_t d = 0; d < dim; ++d) {
            data[i * dim + d] =
                centroids[static_cast<uint64_t>(c) * dim + d] + static_cast<float>(noise(gen));
        }
    }
    return data;
}

/// A skewed mixture: cluster sizes drop geometrically, so a free nearest-centroid
/// assignment would produce strongly uneven clusters (issue #2870 scenario).
std::vector<float>
make_skewed(uint32_t k, uint64_t largest, int32_t dim, std::mt19937& gen, uint64_t& count_out) {
    std::vector<uint64_t> sizes(k);
    uint64_t total = 0;
    uint64_t cur = largest;
    for (uint32_t c = 0; c < k; ++c) {
        sizes[c] = std::max<uint64_t>(1, cur);
        total += sizes[c];
        cur = cur / 3 + 1;
    }
    count_out = total;
    std::vector<float> centroids(static_cast<uint64_t>(k) * dim);
    std::uniform_real_distribution<float> coord(-100.0F, 100.0F);
    std::normal_distribution<float> noise(0.0F, 0.05F);
    for (uint32_t c = 0; c < k; ++c) {
        for (int32_t d = 0; d < dim; ++d) {
            centroids[static_cast<uint64_t>(c) * dim + d] = coord(gen);
        }
    }
    std::vector<float> data(total * static_cast<uint64_t>(dim));
    uint64_t i = 0;
    for (uint32_t c = 0; c < k; ++c) {
        for (uint64_t j = 0; j < sizes[c]; ++j, ++i) {
            for (int32_t d = 0; d < dim; ++d) {
                data[i * dim + d] =
                    centroids[static_cast<uint64_t>(c) * dim + d] + static_cast<float>(noise(gen));
            }
        }
    }
    return data;
}

std::vector<uint64_t>
cluster_sizes(const vsag::Vector<int>& labels, uint32_t k) {
    std::vector<uint64_t> sizes(k, 0);
    for (const int label : labels) {
        REQUIRE(label >= 0);
        REQUIRE(label < static_cast<int>(k));
        ++sizes[static_cast<uint32_t>(label)];
    }
    return sizes;
}

/// The balance guarantee of the issue: every size is floor(n/k) or ceil(n/k),
/// the spread is at most one, and no cluster is empty when 1 <= k <= n.
void
require_exact_balance(const std::vector<uint64_t>& sizes, uint64_t n, uint32_t k) {
    const uint64_t base = n / k;
    const uint64_t extra = n % k;
    REQUIRE(sizes.size() == k);
    uint64_t big = 0;
    for (const uint64_t size : sizes) {
        REQUIRE((size == base or size == base + 1));
        big += (size == base + 1) ? 1 : 0;
        if (base >= 1) {
            REQUIRE(size >= 1);
        }
    }
    REQUIRE(big == extra);
}

}  // namespace

TEST_CASE("BalancedKMeans exact sizes when n divides by k", "[ut][BalancedKMeansCluster]") {
    constexpr uint32_t k = 8;
    constexpr int32_t dim = 16;
    constexpr uint64_t per_cluster = 150;
    std::mt19937 gen(12345);
    std::vector<uint32_t> truth;
    auto data = make_separated(k, per_cluster, dim, gen, truth);

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansCluster cluster(dim, allocator.get());
    const auto labels = cluster.Run(k,
                                    data.data(),
                                    k * per_cluster,
                                    10,
                                    nullptr,
                                    1e-6F,
                                    vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                    42U);

    REQUIRE(labels.size() == k * per_cluster);
    const auto sizes = cluster_sizes(labels, k);
    require_exact_balance(sizes, k * per_cluster, k);
    // On perfectly separated equal-size clusters the balance costs nothing:
    // every generating cluster occupies exactly one bucket.
    for (uint32_t c = 0; c < k; ++c) {
        REQUIRE(sizes[c] == per_cluster);
    }
    std::vector<int> label_of_cluster(k, -1);
    for (uint64_t i = 0; i < labels.size(); ++i) {
        const uint32_t truth_c = truth[i];
        if (label_of_cluster[truth_c] == -1) {
            label_of_cluster[truth_c] = labels[i];
        } else {
            REQUIRE(label_of_cluster[truth_c] == labels[i]);
        }
    }
    REQUIRE(std::unique(label_of_cluster.begin(), label_of_cluster.end()) ==
            label_of_cluster.end());
}

TEST_CASE("BalancedKMeans exact sizes when n does not divide by k", "[ut][BalancedKMeansCluster]") {
    constexpr uint32_t k = 7;
    constexpr int32_t dim = 8;
    constexpr uint64_t per_cluster = 100;
    std::mt19937 gen(999);
    std::vector<uint32_t> truth;
    auto data = make_separated(k, per_cluster, dim, gen, truth);
    // Drop 9 vectors so n % k == 4 with n = 691.
    const uint64_t n = k * per_cluster - 9;
    data.resize(n * static_cast<uint64_t>(dim));

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansCluster cluster(dim, allocator.get());
    double loss = 0.0;
    const auto labels = cluster.Run(
        k, data.data(), n, 12, &loss, 1e-6F, vsag::KMeansInitMethod::KMEANS_PLUS_PLUS, 7U);

    const auto sizes = cluster_sizes(labels, k);
    require_exact_balance(sizes, n, k);
    REQUIRE(loss >= 0.0);
    REQUIRE(std::isfinite(loss));
    const auto& stats = cluster.Stats();
    REQUIRE(stats.cluster_count == k);
    REQUIRE(stats.vector_count == n);
    REQUIRE(stats.min_cluster_size == n / k);
    REQUIRE(stats.max_cluster_size == n / k + 1);
    REQUIRE(stats.iterations_run >= 1);
    REQUIRE(stats.iterations_run <= 12);
}

TEST_CASE("BalancedKMeans constrains a heavily skewed dataset", "[ut][BalancedKMeansCluster]") {
    constexpr uint32_t k = 16;
    constexpr int32_t dim = 12;
    std::mt19937 gen(2718);
    uint64_t n = 0;
    auto data = make_skewed(k, 20000, dim, gen, n);
    REQUIRE(n % k != 0);  // The dataset is deliberately not divisible.

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansCluster cluster(dim, allocator.get());
    double loss = -1.0;
    const auto labels =
        cluster.Run(k, data.data(), n, 8, &loss, 1e-6F, vsag::KMeansInitMethod::RANDOM, 11U);

    const auto sizes = cluster_sizes(labels, k);
    require_exact_balance(sizes, n, k);
    // The whole point of the feature: the largest bucket is ceil(n/k), where an
    // unconstrained k-means on this mixture would have left a multiple of it.
    REQUIRE(*std::max_element(sizes.begin(), sizes.end()) == n / k + 1);
    REQUIRE(std::isfinite(loss));
}

TEST_CASE("BalancedKMeans splits duplicated vectors evenly", "[ut][BalancedKMeansCluster]") {
    constexpr uint32_t k = 10;
    constexpr int32_t dim = 4;
    constexpr uint64_t n = 1000;
    // Every vector identical: nearest-centroid assignment would pile them into
    // one bucket. Duplicates are the extreme case of the issue's scenario.
    std::vector<float> data(n * static_cast<uint64_t>(dim), 1.5F);

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansCluster cluster(dim, allocator.get());
    const auto labels =
        cluster.Run(k, data.data(), n, 5, nullptr, 1e-6F, vsag::KMeansInitMethod::RANDOM, 3U);

    const auto sizes = cluster_sizes(labels, k);
    require_exact_balance(sizes, n, k);
    for (const uint64_t size : sizes) {
        REQUIRE(size == n / k);
    }
    // Every centroid is the mean of a cluster of vectors that are all exactly
    // 1.5, so the trained centroids coincide with the data.
    for (uint64_t j = 0; j < static_cast<uint64_t>(k) * dim; ++j) {
        REQUIRE(std::abs(cluster.k_centroids_[j] - 1.5F) < 1e-5F);
    }
}

TEST_CASE("BalancedKMeans k equals 1 and k equals n", "[ut][BalancedKMeansCluster]") {
    constexpr int32_t dim = 6;
    constexpr uint64_t n = 100;
    std::mt19937 gen(555);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    std::vector<float> data(n * static_cast<uint64_t>(dim));
    for (auto& v : data) {
        v = noise(gen);
    }

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    SECTION("k == 1 puts everything in one cluster") {
        vsag::BalancedKMeansCluster cluster(dim, allocator.get());
        const auto labels = cluster.Run(1, data.data(), n, 3);
        for (const int label : labels) {
            REQUIRE(label == 0);
        }
        REQUIRE(cluster.Stats().min_cluster_size == n);
        REQUIRE(cluster.Stats().max_cluster_size == n);
    }
    SECTION("k == n gives every cluster exactly one vector") {
        vsag::BalancedKMeansCluster cluster(dim, allocator.get());
        const auto labels = cluster.Run(static_cast<uint32_t>(n), data.data(), n, 2);
        const auto sizes = cluster_sizes(labels, static_cast<uint32_t>(n));
        require_exact_balance(sizes, n, n);
        for (const uint64_t size : sizes) {
            REQUIRE(size == 1);
        }
    }
}

TEST_CASE("BalancedKMeans rejects invalid arguments", "[ut][BalancedKMeansCluster]") {
    constexpr int32_t dim = 4;
    constexpr uint64_t n = 64;
    std::vector<float> data(n * static_cast<uint64_t>(dim), 0.5F);
    std::vector<float> centroids(4 * static_cast<uint64_t>(dim), 0.5F);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansCluster cluster(dim, allocator.get());

    SECTION("Run") {
        // k == 0
        REQUIRE_THROWS(cluster.Run(0, data.data(), n, 1));
        // n == 0
        REQUIRE_THROWS(cluster.Run(4, data.data(), 0, 1));
        // null data
        REQUIRE_THROWS(cluster.Run(4, nullptr, n, 1));
        // k > n
        REQUIRE_THROWS(cluster.Run(static_cast<uint32_t>(n + 1), data.data(), n, 1));
        // non-positive iterations
        REQUIRE_THROWS(cluster.Run(4, data.data(), n, 0));
        REQUIRE_THROWS(cluster.Run(4, data.data(), n, -1));
        // zero candidates
        REQUIRE_THROWS(cluster.Run(
            4, data.data(), n, 1, nullptr, 1e-6F, vsag::KMeansInitMethod::RANDOM, 1U, 0U));
        // non-finite values
        auto bad = data;
        bad[13] = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS(cluster.Run(4, bad.data(), n, 1));
        bad[13] = std::numeric_limits<float>::infinity();
        REQUIRE_THROWS(cluster.Run(4, bad.data(), n, 1));
    }
    SECTION("Assign") {
        REQUIRE_THROWS(cluster.Assign(0, centroids.data(), data.data(), n));
        REQUIRE_THROWS(cluster.Assign(4, nullptr, data.data(), n));
        REQUIRE_THROWS(cluster.Assign(4, centroids.data(), nullptr, n));
        REQUIRE_THROWS(cluster.Assign(4, centroids.data(), data.data(), 0));
        REQUIRE_THROWS(cluster.Assign(
            4, centroids.data(), data.data(), n, vsag::MetricType::METRIC_TYPE_L2SQR, 0U));
        auto bad_centroids = centroids;
        bad_centroids[5] = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS(cluster.Assign(4, bad_centroids.data(), data.data(), n));
        auto bad_data = data;
        bad_data[9] = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS(cluster.Assign(4, centroids.data(), bad_data.data(), n));
    }
    SECTION("Constructor") {
        REQUIRE_THROWS(vsag::BalancedKMeansCluster(0, allocator.get()));
        REQUIRE_THROWS(vsag::BalancedKMeansCluster(-4, allocator.get()));
    }
}

TEST_CASE("BalancedKMeans is reproducible across thread counts", "[ut][BalancedKMeansCluster]") {
    constexpr uint32_t k = 5;
    constexpr int32_t dim = 20;
    std::mt19937 gen(8121);
    uint64_t n = 0;
    auto data = make_skewed(k, 3000, dim, gen, n);

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto single_thread_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    single_thread_pool->SetPoolSize(1);
    auto multi_thread_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    multi_thread_pool->SetPoolSize(4);

    vsag::BalancedKMeansCluster single_thread(dim, allocator.get(), single_thread_pool);
    vsag::BalancedKMeansCluster multi_thread(dim, allocator.get(), multi_thread_pool);

    const uint32_t iterations = 6;
    const auto single_labels = single_thread.Run(k,
                                                 data.data(),
                                                 n,
                                                 iterations,
                                                 nullptr,
                                                 1e-6F,
                                                 vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                                 0x2870U);
    const auto multi_labels = multi_thread.Run(k,
                                               data.data(),
                                               n,
                                               iterations,
                                               nullptr,
                                               1e-6F,
                                               vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                               0x2870U);

    // No shared float accumulation happens in parallel code, so a run is
    // bit-identical for any thread count under the same seed.
    REQUIRE(single_labels == multi_labels);
    const uint64_t centroid_values = static_cast<uint64_t>(k) * dim;
    REQUIRE(std::equal(single_thread.k_centroids_,
                       single_thread.k_centroids_ + centroid_values,
                       multi_thread.k_centroids_));
    REQUIRE(single_thread.Stats().loss == multi_thread.Stats().loss);
}

TEST_CASE("BalancedKMeans Assign balances fixed centroids", "[ut][BalancedKMeansCluster]") {
    // Two centroids, two vectors, capacity one each: the vector with the
    // stronger preference keeps its favourite and the other takes the rest,
    // under the IP routing key (cost 1 - <x, c>) IVF uses.
    constexpr uint32_t k = 2;
    constexpr int32_t dim = 2;
    const std::vector<float> centroids = {0.0F, 0.0F, 10.0F, 0.0F};
    const std::vector<float> data = {10.0F, 0.0F, 0.5F, 0.0F};
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansCluster cluster(dim, allocator.get());

    const auto labels =
        cluster.Assign(k, centroids.data(), data.data(), 2, vsag::MetricType::METRIC_TYPE_IP, 2U);
    REQUIRE(labels[0] == 1);
    REQUIRE(labels[1] == 0);
    REQUIRE(cluster.Stats().min_cluster_size == 1);
    REQUIRE(cluster.Stats().max_cluster_size == 1);
}

TEST_CASE("BalancedKMeans Assign with fewer vectors than clusters spreads them",
          "[ut][BalancedKMeansCluster]") {
    constexpr uint32_t k = 10;
    constexpr int32_t dim = 3;
    constexpr uint64_t n = 6;
    std::mt19937 gen(77);
    std::uniform_real_distribution<float> coord(-1.0F, 1.0F);
    std::vector<float> centroids(static_cast<uint64_t>(k) * dim);
    for (auto& v : centroids) {
        v = coord(gen);
    }
    std::vector<float> data(n * static_cast<uint64_t>(dim));
    for (auto& v : data) {
        v = coord(gen);
    }

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansCluster cluster(dim, allocator.get());
    const auto labels = cluster.Assign(k, centroids.data(), data.data(), n);
    const auto sizes = cluster_sizes(labels, k);
    // floor(6/10) == 0 and ceil == 1: no bucket holds more than one vector.
    for (const uint64_t size : sizes) {
        REQUIRE(size <= 1);
    }
    REQUIRE(std::accumulate(sizes.begin(), sizes.end(), 0ULL) == n);
}

TEST_CASE("BalancedKMeans stays exact with one candidate and with full candidates",
          "[ut][BalancedKMeansCluster]") {
    constexpr uint32_t k = 4;
    constexpr int32_t dim = 8;
    constexpr uint64_t per_cluster = 60;
    std::mt19937 gen(313);
    std::vector<uint32_t> truth;
    auto data = make_separated(k, per_cluster, dim, gen, truth);
    // Skew by deleting so n % k == 2.
    const uint64_t n = k * per_cluster - 2;
    data.resize(n * static_cast<uint64_t>(dim));

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    SECTION("num_candidates == 1 forces on-demand rescans") {
        vsag::BalancedKMeansCluster cluster(dim, allocator.get());
        const auto labels = cluster.Run(k,
                                        data.data(),
                                        n,
                                        5,
                                        nullptr,
                                        1e-6F,
                                        vsag::KMeansInitMethod::RANDOM,
                                        5U,
                                        /*num_candidates=*/1U);
        const auto sizes = cluster_sizes(labels, k);
        require_exact_balance(sizes, n, k);
    }
    SECTION("num_candidates >= k keeps every cluster as a candidate") {
        vsag::BalancedKMeansCluster cluster(dim, allocator.get());
        const auto labels = cluster.Run(k,
                                        data.data(),
                                        n,
                                        5,
                                        nullptr,
                                        1e-6F,
                                        vsag::KMeansInitMethod::RANDOM,
                                        5U,
                                        /*num_candidates=*/1024U);
        const auto sizes = cluster_sizes(labels, k);
        require_exact_balance(sizes, n, k);
    }
}

TEST_CASE("BalancedKMeans assignment matches centroids and reported loss",
          "[ut][BalancedKMeansCluster]") {
    constexpr uint32_t k = 6;
    constexpr int32_t dim = 10;
    std::mt19937 gen(4242);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    constexpr uint64_t n = 500;
    std::vector<float> data(n * static_cast<uint64_t>(dim));
    for (auto& v : data) {
        v = noise(gen);
    }

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansCluster cluster(dim, allocator.get());
    double reported = 0.0;
    const auto labels = cluster.Run(k, data.data(), n, 4, &reported);
    REQUIRE(reported > 0.0);

    // Recompute the final assignment loss against the centroids the labels were
    // produced with: the mean of each cluster's members is the stored centroid,
    // so recompute the cost of the stored (labels, centroids) pair.
    std::vector<double> sums(static_cast<uint64_t>(k) * dim, 0.0);
    std::vector<uint64_t> sizes(k, 0);
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t label = static_cast<uint32_t>(labels[i]);
        ++sizes[label];
        for (int32_t d = 0; d < dim; ++d) {
            sums[static_cast<uint64_t>(label) * dim + d] += data[i * dim + d];
        }
    }
    double recomputed = 0.0;
    for (uint32_t j = 0; j < k; ++j) {
        REQUIRE(sizes[j] >= 1);
        for (int32_t d = 0; d < dim; ++d) {
            const float mean =
                static_cast<float>(sums[static_cast<uint64_t>(j) * dim + d] / sizes[j]);
            REQUIRE(std::abs(mean - cluster.k_centroids_[static_cast<uint64_t>(j) * dim + d]) <
                    1e-3F);
        }
    }
    // The reported loss is the cost of the last assignment step; it must be no
    // worse than the cost of placing every vector at its own centroid mean.
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t label = static_cast<uint32_t>(labels[i]);
        double dist = 0.0;
        for (int32_t d = 0; d < dim; ++d) {
            const double diff =
                static_cast<double>(data[i * dim + d]) -
                static_cast<double>(cluster.k_centroids_[static_cast<uint64_t>(label) * dim + d]);
            dist += diff * diff;
        }
        recomputed += dist;
    }
    REQUIRE(recomputed <= reported + 1.0);
}

TEST_CASE("BalancedKMeans avoids a dense n-by-k distance matrix", "[ut][BalancedKMeansCluster]") {
    // The assignment is required to bound its working set (issue #2870): the
    // trainer keeps per-vector candidate lists, never an n x k matrix. A
    // recording allocator measures the peak against that matrix's size.
    constexpr uint32_t k = 200;
    constexpr int32_t dim = 32;
    constexpr uint64_t n = 4000;
    std::mt19937 gen(12);
    std::uniform_real_distribution<float> coord(-1.0F, 1.0F);
    std::vector<float> data(n * static_cast<uint64_t>(dim));
    for (auto& v : data) {
        v = coord(gen);
    }
    const uint64_t dense_matrix_bytes = n * static_cast<uint64_t>(k) * sizeof(float);

    auto allocator = std::make_shared<fixtures::MemoryRecordAllocator>();
    // The recording allocator only measures what the trainer allocates through
    // the allocator it is handed; the dataset itself lives outside of it.
    vsag::BalancedKMeansCluster cluster(dim, allocator.get(), nullptr);
    const auto labels = cluster.Run(k, data.data(), n, 3);

    const uint64_t peak = allocator->GetMemoryPeak();
    REQUIRE(peak < dense_matrix_bytes / 4);
    // Sanity: the run still balances.
    const auto sizes = cluster_sizes(labels, k);
    require_exact_balance(sizes, n, k);
}
