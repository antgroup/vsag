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

#define VSAG_BALANCED_KMEANS_TEST_ACCESS
#include "balanced_kmeans_cluster.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "allocator/memory_record_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "kmeans_cluster.h"
#include "simd/fp32_simd.h"
#include "unittest.h"
#include "vsag_exception.h"

class BalancedKMeansClusterTestAccess {
public:
    static void
    SetFirstAssignmentBudgetFactor(vsag::BalancedKMeansCluster& cluster, double factor) {
        cluster.first_assignment_budget_factor_ = factor;
    }
};

namespace {

std::vector<float>
uniform_vectors(uint64_t count, uint64_t dim, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> datas(count * dim);
    for (auto& value : datas) {
        value = dist(gen);
    }
    return datas;
}

/// `dense_fraction` of the vectors sit in one tight blob; the rest are spread out.
std::vector<float>
skewed_vectors(uint64_t count, uint64_t dim, double dense_fraction, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> tight(0.0F, 0.01F);
    std::uniform_real_distribution<float> wide(-10.0F, 10.0F);
    std::bernoulli_distribution dense(dense_fraction);
    std::vector<float> datas(count * dim);
    for (uint64_t row = 0; row < count; ++row) {
        const bool in_blob = dense(gen);
        for (uint64_t d = 0; d < dim; ++d) {
            datas[row * dim + d] = in_blob ? 5.0F + tight(gen) : wide(gen);
        }
    }
    return datas;
}

/// Sizes are floor(count / k) or ceil(count / k), exactly `count % k` are the larger.
std::vector<uint64_t>
require_balanced(const vsag::Vector<int32_t>& labels, uint64_t count, uint64_t k) {
    REQUIRE(labels.size() == count);
    std::vector<uint64_t> sizes(k, 0);
    for (const auto label : labels) {
        REQUIRE(label >= 0);
        REQUIRE(static_cast<uint64_t>(label) < k);
        ++sizes[static_cast<uint64_t>(label)];
    }
    const uint64_t low = count / k;
    const uint64_t high = (count + k - 1) / k;
    uint64_t larger = 0;
    for (const auto size : sizes) {
        REQUIRE(size >= low);
        REQUIRE(size <= high);
        REQUIRE(size > 0);
        if (size == high and high != low) {
            ++larger;
        }
    }
    const auto [smallest, largest] = std::minmax_element(sizes.begin(), sizes.end());
    REQUIRE(*largest - *smallest <= 1);
    if (high != low) {
        REQUIRE(larger == count % k);
    }
    return sizes;
}

uint64_t
largest_cluster(const vsag::Vector<int>& labels, uint64_t k) {
    std::vector<uint64_t> sizes(k, 0);
    for (const auto label : labels) {
        ++sizes[static_cast<uint64_t>(label)];
    }
    return *std::max_element(sizes.begin(), sizes.end());
}

/// Minimum-cost perfect matching on a square matrix (Hungarian algorithm, O(n^3)).
double
min_cost_matching(const std::vector<std::vector<double>>& cost) {
    const auto n = static_cast<int64_t>(cost.size());
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(n + 1, 0.0);
    std::vector<int64_t> match(n + 1, 0);
    std::vector<int64_t> way(n + 1, 0);
    for (int64_t i = 1; i <= n; ++i) {
        match[0] = i;
        int64_t j0 = 0;
        std::vector<double> minv(n + 1, inf);
        std::vector<char> used(n + 1, 0);
        do {
            used[j0] = 1;
            const int64_t i0 = match[j0];
            int64_t j1 = 0;
            double delta = inf;
            for (int64_t j = 1; j <= n; ++j) {
                if (used[j] != 0) {
                    continue;
                }
                const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int64_t j = 0; j <= n; ++j) {
                if (used[j] != 0) {
                    u[match[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (match[j0] != 0);
        do {
            const int64_t j1 = way[j0];
            match[j0] = match[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    double total = 0.0;
    for (int64_t j = 1; j <= n; ++j) {
        total += cost[match[j] - 1][j - 1];
    }
    return total;
}

/// The optimal assignment for the given capacities, by matching vectors to seats.
double
optimal_assignment_cost(const std::function<double(uint64_t, uint64_t)>& cost,
                        uint64_t count,
                        const std::vector<uint64_t>& capacity) {
    std::vector<uint64_t> seat_cluster;
    for (uint64_t j = 0; j < capacity.size(); ++j) {
        seat_cluster.insert(seat_cluster.end(), capacity[j], j);
    }
    REQUIRE(seat_cluster.size() == count);
    std::vector<std::vector<double>> matrix(count, std::vector<double>(count));
    for (uint64_t row = 0; row < count; ++row) {
        for (uint64_t seat = 0; seat < count; ++seat) {
            matrix[row][seat] = cost(row, seat_cluster[seat]);
        }
    }
    return min_cost_matching(matrix);
}

/// The cost Assign minimizes under `metric`, computed directly.
std::function<double(uint64_t, uint64_t)>
direct_cost(vsag::BalancedAssignMetric metric,
            const std::vector<float>& datas,
            const std::vector<float>& centroids,
            uint64_t dim) {
    return [metric, &datas, &centroids, dim](uint64_t row, uint64_t cluster) -> double {
        const float* x = datas.data() + row * dim;
        const float* c = centroids.data() + cluster * dim;
        if (metric == vsag::BalancedAssignMetric::L2SQR) {
            return vsag::FP32ComputeL2Sqr(x, c, dim);
        }
        const double dot = vsag::FP32ComputeIP(x, c, dim);
        if (metric == vsag::BalancedAssignMetric::INNER_PRODUCT) {
            return 1.0 - dot;
        }
        return 1.0 - dot / std::sqrt(static_cast<double>(vsag::FP32ComputeIP(x, x, dim)));
    };
}

/// Unit-length rows, as IVF keeps its cosine centroids.
void
normalize_rows(std::vector<float>& rows, uint64_t dim) {
    for (uint64_t row = 0; row < rows.size() / dim; ++row) {
        float* x = rows.data() + row * dim;
        const float norm = std::sqrt(vsag::FP32ComputeIP(x, x, dim));
        for (uint64_t d = 0; d < dim; ++d) {
            x[d] /= norm;
        }
    }
}

/// How far above the optimum an auction may stop: count * epsilon, and epsilon is
/// 1e-4 of a cost gap that cannot exceed the range of all costs.
double
auction_tolerance(const std::function<double(uint64_t, uint64_t)>& cost,
                  uint64_t count,
                  uint64_t k) {
    double lowest = std::numeric_limits<double>::max();
    double highest = std::numeric_limits<double>::lowest();
    for (uint64_t row = 0; row < count; ++row) {
        for (uint64_t j = 0; j < k; ++j) {
            lowest = std::min(lowest, cost(row, j));
            highest = std::max(highest, cost(row, j));
        }
    }
    return 1e-4 * static_cast<double>(count) * (highest - lowest) + 1e-6;
}

template <typename Func>
void
require_invalid_argument(Func&& func) {
    try {
        func();
        FAIL("expected an INVALID_ARGUMENT exception");
    } catch (const vsag::VsagException& e) {
        REQUIRE(e.error_.type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

vsag::SafeThreadPoolPtr
make_pool(uint64_t threads) {
    auto pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    pool->SetPoolSize(threads);
    return pool;
}

}  // namespace

TEST_CASE("BalancedKMeans keeps every cluster at floor or ceil of count / k",
          "[ut][BalancedKMeansCluster]") {
    const auto [count, k, dim] = GENERATE(table<uint64_t, uint32_t, uint64_t>({
        {1000, 10, 8},  // divisible
        {1003, 10, 8},  // three clusters hold one more vector
        {997, 37, 16},
        {641, 64, 4},
        {300, 299, 3},  // almost one vector per cluster
        {50, 7, 2},
    }));
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto datas = uniform_vectors(count, dim, 7);
    vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
    vsag::BalancedKMeansOptions options;
    options.iterations = 10;
    options.random_seed = 42;
    double err = -1.0;
    auto labels = cluster.Run(k, datas.data(), count, options, &err);
    const auto sizes = require_balanced(labels, count, k);
    REQUIRE(std::isfinite(err));
    REQUIRE(err >= 0.0);
    REQUIRE(cluster.stats_.iterations >= 1);
    REQUIRE(cluster.stats_.iterations <= options.iterations);
    REQUIRE(cluster.stats_.loss_history.size() == static_cast<uint64_t>(cluster.stats_.iterations));
    REQUIRE(cluster.stats_.bids >= count);
    REQUIRE(cluster.stats_.working_set_bytes > 0);
    REQUIRE(cluster.stats_.fallback_assignments == 0);

    // Every centroid is the mean of its cluster, and err is the mean squared distance.
    std::vector<double> sums(static_cast<uint64_t>(k) * dim, 0.0);
    for (uint64_t row = 0; row < count; ++row) {
        for (uint64_t d = 0; d < dim; ++d) {
            sums[static_cast<uint64_t>(labels[row]) * dim + d] += datas[row * dim + d];
        }
    }
    double loss = 0.0;
    for (uint64_t row = 0; row < count; ++row) {
        loss += vsag::FP32ComputeL2Sqr(
            datas.data() + row * dim, cluster.k_centroids_ + labels[row] * dim, dim);
    }
    for (uint64_t j = 0; j < k; ++j) {
        for (uint64_t d = 0; d < dim; ++d) {
            const double mean = sums[j * dim + d] / static_cast<double>(sizes[j]);
            REQUIRE(std::abs(cluster.k_centroids_[j * dim + d] - mean) < 1e-4);
        }
    }
    REQUIRE(std::abs(loss / static_cast<double>(count) - err) <= 1e-6 * (1.0 + err));
}

TEST_CASE("BalancedKMeans handles k == 1 and k == count", "[ut][BalancedKMeansCluster]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 6;
    constexpr uint64_t count = 128;
    auto datas = uniform_vectors(count, dim, 3);
    vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
    vsag::BalancedKMeansOptions options;
    options.random_seed = 5;

    SECTION("k == 1 puts every vector in one cluster at the mean") {
        double err = -1.0;
        auto labels = cluster.Run(1, datas.data(), count, options, &err);
        require_balanced(labels, count, 1);
        for (uint64_t d = 0; d < dim; ++d) {
            double mean = 0.0;
            for (uint64_t row = 0; row < count; ++row) {
                mean += datas[row * dim + d];
            }
            mean /= static_cast<double>(count);
            REQUIRE(std::abs(cluster.k_centroids_[d] - mean) < 1e-5);
        }
        REQUIRE(err > 0.0);
    }

    SECTION("k == count gives every vector its own cluster") {
        double err = -1.0;
        auto labels = cluster.Run(count, datas.data(), count, options, &err);
        require_balanced(labels, count, count);
        REQUIRE(err < 1e-10);
    }
}

TEST_CASE("BalancedKMeans splits skewed and duplicated data evenly",
          "[ut][BalancedKMeansCluster]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansOptions options;
    options.random_seed = 11;
    options.iterations = 15;

    SECTION("a blob holding 90% of the vectors") {
        constexpr uint64_t dim = 8;
        constexpr uint64_t count = 5000;
        constexpr uint32_t k = 50;
        auto datas = skewed_vectors(count, dim, 0.9, 13);
        vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
        auto labels = cluster.Run(k, datas.data(), count, options);
        require_balanced(labels, count, k);
        REQUIRE(cluster.stats_.fallback_assignments == 0);

        // Plain k-means leaves most of the blob in a handful of clusters.
        vsag::KMeansCluster kmeans(static_cast<int32_t>(dim), allocator.get());
        auto plain = kmeans.Run(k,
                                datas.data(),
                                count,
                                15,
                                nullptr,
                                false,
                                1e-6F,
                                vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                11);
        REQUIRE(largest_cluster(plain, k) > 2 * count / k);
    }

    SECTION("every vector identical") {
        constexpr uint64_t dim = 4;
        constexpr uint64_t count = 1000;
        constexpr uint32_t k = 16;
        std::vector<float> datas(count * dim, 0.25F);
        vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
        double err = -1.0;
        auto labels = cluster.Run(k, datas.data(), count, options, &err);
        require_balanced(labels, count, k);
        REQUIRE(err < 1e-10);
    }

    SECTION("half of the vectors are copies of one vector") {
        constexpr uint64_t dim = 4;
        constexpr uint64_t count = 2001;
        constexpr uint32_t k = 40;
        auto datas = uniform_vectors(count, dim, 17);
        for (uint64_t row = 0; row < count / 2; ++row) {
            std::copy_n(datas.data(), dim, datas.data() + row * dim);
        }
        vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
        auto labels = cluster.Run(k, datas.data(), count, options);
        require_balanced(labels, count, k);
        REQUIRE(cluster.stats_.fallback_assignments == 0);
    }
}

TEST_CASE("BalancedKMeans assignment is optimal for its centroids", "[ut][BalancedKMeansCluster]") {
    const auto [count, k, candidates, seed] =
        GENERATE(table<uint64_t, uint32_t, uint32_t, uint32_t>({
            {60, 6, 16, 1},  // every cluster is a candidate
            {60, 6, 2, 2},   // two candidates: bounds and re-scoring decide
            {63, 5, 1, 3},   // three clusters take one more vector, one candidate each
            {96, 12, 3, 4},
            {80, 8, 4, 5},
        }));
    constexpr uint64_t dim = 4;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto datas = uniform_vectors(count, dim, seed);
    auto centroids = uniform_vectors(k, dim, seed + 100);
    const auto metric = GENERATE(vsag::BalancedAssignMetric::L2SQR,
                                 vsag::BalancedAssignMetric::INNER_PRODUCT,
                                 vsag::BalancedAssignMetric::COSINE);
    if (metric == vsag::BalancedAssignMetric::COSINE) {
        normalize_rows(centroids, dim);
    }
    const auto cost = direct_cost(metric, datas, centroids, dim);

    vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
    double mean_cost = 0.0;
    auto labels =
        cluster.Assign(centroids.data(), k, datas.data(), count, metric, candidates, &mean_cost);
    const auto sizes = require_balanced(labels, count, k);
    double total = 0.0;
    for (uint64_t row = 0; row < count; ++row) {
        total += cost(row, static_cast<uint64_t>(labels[row]));
    }
    REQUIRE(std::abs(total / static_cast<double>(count) - mean_cost) < 1e-9);
    const double optimum = optimal_assignment_cost(cost, count, sizes);
    REQUIRE(total >= optimum - 1e-6);
    REQUIRE(total <= optimum + auction_tolerance(cost, count, k));
    REQUIRE(cluster.stats_.fallback_assignments == 0);
}

TEST_CASE("BalancedKMeans Assign gives each vector its own cluster when k > count",
          "[ut][BalancedKMeansCluster]") {
    const auto [count, k, candidates, seed] =
        GENERATE(table<uint64_t, uint32_t, uint32_t, uint32_t>({
            {5, 8, 16, 11},   // every cluster is a candidate
            {12, 16, 2, 12},  // two candidates: bounds and re-scoring decide
            {20, 21, 3, 13},  // one cluster stays empty
            {1, 4, 1, 14},    // a single vector
        }));
    constexpr uint64_t dim = 4;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto datas = uniform_vectors(count, dim, seed);
    auto centroids = uniform_vectors(k, dim, seed + 100);
    const auto metric = GENERATE(vsag::BalancedAssignMetric::L2SQR,
                                 vsag::BalancedAssignMetric::INNER_PRODUCT,
                                 vsag::BalancedAssignMetric::COSINE);
    if (metric == vsag::BalancedAssignMetric::COSINE) {
        normalize_rows(centroids, dim);
    }
    const auto cost = direct_cost(metric, datas, centroids, dim);

    vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
    auto labels = cluster.Assign(centroids.data(), k, datas.data(), count, metric, candidates);
    REQUIRE(labels.size() == count);
    // floor(count / k) == 0 and ceil(count / k) == 1: no two vectors share a cluster.
    std::vector<uint64_t> sizes(k, 0);
    double total = 0.0;
    for (uint64_t row = 0; row < count; ++row) {
        const auto label = static_cast<uint64_t>(labels[row]);
        REQUIRE(label < k);
        ++sizes[label];
        REQUIRE(sizes[label] == 1);
        total += cost(row, label);
    }
    // The optimum of the rectangular problem: k - count more rows that cost nothing
    // anywhere make it a square one.
    std::vector<std::vector<double>> matrix(k, std::vector<double>(k, 0.0));
    for (uint64_t row = 0; row < count; ++row) {
        for (uint64_t j = 0; j < k; ++j) {
            matrix[row][j] = cost(row, j);
        }
    }
    const double optimum = min_cost_matching(matrix);
    REQUIRE(total >= optimum - 1e-6);
    REQUIRE(total <= optimum + auction_tolerance(cost, count, k));
    REQUIRE(cluster.stats_.fallback_assignments == 0);
}

TEST_CASE("BalancedKMeans Assign settles ties that float rounding splits",
          "[ut][BalancedKMeansCluster]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    // A vector remembers one cluster and bounds the other, equal one. Before costs were
    // rounded ahead of ranking, the remembered cost rounded up above that bound, every
    // vector asked to be scored again until the budget ran out, and none ever bid.
    SECTION("equal inner-product costs") {
        // 1 - 1.0 * 0.2 is 0.79999999702 in double and 0.80000001192 in float.
        const std::vector<float> centroids{0.2F, 0.2F};
        const std::vector<float> datas{1.0F, 1.0F};
        vsag::BalancedKMeansCluster cluster(1, allocator.get());
        auto labels = cluster.Assign(
            centroids.data(), 2, datas.data(), 2, vsag::BalancedAssignMetric::INNER_PRODUCT, 1);
        require_balanced(labels, 2, 2);
        REQUIRE(cluster.stats_.fallback_assignments == 0);
    }

    SECTION("equal squared-L2 costs") {
        // 0.09 + 0.01 - 2 * 0.03, each term a float, is 0.0400000019 in double and
        // 0.0400000028 in float: the same split between a vector's two equal clusters.
        const std::vector<float> centroids{0.1F, 0.1F};
        const std::vector<float> datas{0.3F, 0.3F};
        vsag::BalancedKMeansCluster cluster(1, allocator.get());
        auto labels = cluster.Assign(
            centroids.data(), 2, datas.data(), 2, vsag::BalancedAssignMetric::L2SQR, 1);
        require_balanced(labels, 2, 2);
        REQUIRE(cluster.stats_.fallback_assignments == 0);
    }
}

TEST_CASE("BalancedKMeans loss does not increase between iterations",
          "[ut][BalancedKMeansCluster]") {
    constexpr uint64_t dim = 16;
    constexpr uint64_t count = 3000;
    constexpr uint32_t k = 30;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto datas = skewed_vectors(count, dim, 0.5, 23);
    vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
    vsag::BalancedKMeansOptions options;
    options.random_seed = 29;
    options.iterations = 20;
    options.init_method =
        GENERATE(vsag::KMeansInitMethod::RANDOM, vsag::KMeansInitMethod::KMEANS_PLUS_PLUS);
    auto labels = cluster.Run(k, datas.data(), count, options);
    require_balanced(labels, count, k);
    const auto& history = cluster.stats_.loss_history;
    REQUIRE(not history.empty());
    for (uint64_t i = 1; i < history.size(); ++i) {
        REQUIRE(history[i] <= history[i - 1] * (1.0 + 1e-3) + 1e-9);
    }
}

TEST_CASE("BalancedKMeans is reproducible with a seed", "[ut][BalancedKMeansCluster]") {
    constexpr uint64_t dim = 12;
    constexpr uint64_t count = 2500;
    constexpr uint32_t k = 25;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto datas = skewed_vectors(count, dim, 0.6, 31);
    vsag::BalancedKMeansOptions options;
    options.random_seed = 37;
    options.iterations = 8;

    vsag::BalancedKMeansCluster single(static_cast<int32_t>(dim), allocator.get(), make_pool(1));
    vsag::BalancedKMeansCluster many(static_cast<int32_t>(dim), allocator.get(), make_pool(8));
    auto first = single.Run(k, datas.data(), count, options);
    auto second = many.Run(k, datas.data(), count, options);
    REQUIRE(first == second);
    REQUIRE(std::memcmp(single.k_centroids_, many.k_centroids_, k * dim * sizeof(float)) == 0);
    REQUIRE(single.stats_.loss_history == many.stats_.loss_history);

    // Running again on the same object replaces the previous result.
    auto again = single.Run(k, datas.data(), count, options);
    REQUIRE(again == first);
}

TEST_CASE("BalancedKMeans never holds a count x k matrix", "[ut][BalancedKMeansCluster]") {
    constexpr uint64_t dim = 8;
    constexpr uint64_t count = 20000;
    constexpr uint32_t k = 2000;
    auto datas = uniform_vectors(count, dim, 41);
    fixtures::MemoryRecordAllocator allocator;
    uint64_t peak = 0;
    uint64_t working_set = 0;
    {
        vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), &allocator, make_pool(4));
        vsag::BalancedKMeansOptions options;
        options.random_seed = 43;
        options.iterations = 2;
        auto labels = cluster.Run(k, datas.data(), count, options);
        require_balanced(labels, count, k);
        peak = allocator.GetMemoryPeak();
        working_set = cluster.stats_.working_set_bytes;
    }
    const uint64_t dense = count * k * sizeof(float);
    REQUIRE(peak < dense / 10);
    REQUIRE(working_set < dense / 10);
    REQUIRE(working_set <= peak);
}

TEST_CASE("BalancedKMeans keeps exact sizes on every budget path", "[ut][BalancedKMeansCluster]") {
    constexpr uint64_t dim = 8;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::BalancedKMeansOptions options;
    options.random_seed = 53;
    options.iterations = 12;

    SECTION("the two-level decomposition, with and without a remainder") {
        const auto [count, k] = GENERATE(table<uint64_t, uint32_t>({
            {1001, 20},  // 4 x 5 leaves, one larger leaf per super-cluster and one more
            {1000, 50},  // 7 super-clusters of 7 or 8 leaves
            {600, 3},    // two super-clusters of one and two leaves
            {500, 2},    // too few clusters to split
        }));
        auto datas = skewed_vectors(count, dim, 0.8, 47);
        vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
        options.work_budget = 0.0;
        double err = -1.0;
        auto labels = cluster.Run(k, datas.data(), count, options, &err);
        const auto sizes = require_balanced(labels, count, k);
        REQUIRE(cluster.stats_.two_level == (k > 2));
        REQUIRE(std::isfinite(err));
        // Every centroid is the mean of its leaf.
        for (uint64_t j = 0; j < k; ++j) {
            std::vector<double> mean(dim, 0.0);
            for (uint64_t row = 0; row < count; ++row) {
                if (static_cast<uint64_t>(labels[row]) == j) {
                    for (uint64_t d = 0; d < dim; ++d) {
                        mean[d] += datas[row * dim + d] / static_cast<double>(sizes[j]);
                    }
                }
            }
            for (uint64_t d = 0; d < dim; ++d) {
                REQUIRE(std::abs(cluster.k_centroids_[j * dim + d] - mean[d]) < 1e-3);
            }
        }
    }

    SECTION("a later iteration over budget keeps the previous result") {
        constexpr uint64_t count = 4000;
        constexpr uint32_t k = 40;
        auto datas = skewed_vectors(count, dim, 0.5, 71);
        // The first assignment may spend without limit, every later one almost nothing.
        vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
        BalancedKMeansClusterTestAccess::SetFirstAssignmentBudgetFactor(
            cluster, std::numeric_limits<double>::infinity());
        options.work_budget = 1e-9;
        double err = -1.0;
        auto labels = cluster.Run(k, datas.data(), count, options, &err);
        require_balanced(labels, count, k);
        REQUIRE(cluster.stats_.stopped_on_budget);
        // The result is the better of one flat iteration and the two-level run.
        REQUIRE(err == cluster.stats_.loss_history.back());
        if (not cluster.stats_.two_level) {
            REQUIRE(cluster.stats_.iterations == 1);
        }
        vsag::BalancedKMeansCluster two_level(static_cast<int32_t>(dim), allocator.get());
        options.work_budget = 0.0;
        double two_level_err = -1.0;
        two_level.Run(k, datas.data(), count, options, &two_level_err);
        REQUIRE(two_level.stats_.two_level);
        REQUIRE(err <= two_level_err + 1e-9);
    }

    SECTION("Assign places the vectors left over by its auction") {
        constexpr uint64_t count = 1003;
        constexpr uint32_t k = 17;
        auto datas = skewed_vectors(count, dim, 0.8, 79);
        auto centroids = uniform_vectors(k, dim, 83);
        vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
        double budget_cost = 0.0;
        auto labels = cluster.Assign(centroids.data(),
                                     k,
                                     datas.data(),
                                     count,
                                     vsag::BalancedAssignMetric::L2SQR,
                                     4,
                                     &budget_cost,
                                     0.0);
        require_balanced(labels, count, k);
        REQUIRE(cluster.stats_.fallback_assignments == count);
        double full_cost = 0.0;
        cluster.Assign(centroids.data(),
                       k,
                       datas.data(),
                       count,
                       vsag::BalancedAssignMetric::L2SQR,
                       4,
                       &full_cost,
                       std::numeric_limits<double>::infinity());
        REQUIRE(cluster.stats_.fallback_assignments == 0);
        REQUIRE(full_cost <= budget_cost + 1e-6);
    }
}

TEST_CASE("BalancedKMeans options change the run as documented", "[ut][BalancedKMeansCluster]") {
    constexpr uint64_t dim = 8;
    constexpr uint64_t count = 2000;
    constexpr uint32_t k = 20;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto datas = uniform_vectors(count, dim, 59);
    vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
    vsag::BalancedKMeansOptions options;
    options.random_seed = 61;
    options.iterations = 30;

    SECTION("a large tolerance stops early") {
        options.tolerance = 0.5;
        auto labels = cluster.Run(k, datas.data(), count, options);
        require_balanced(labels, count, k);
        REQUIRE(cluster.stats_.iterations < options.iterations);
    }

    SECTION("one candidate per vector relies on re-scoring") {
        options.candidate_count = 1;
        auto labels = cluster.Run(k, datas.data(), count, options);
        require_balanced(labels, count, k);
        REQUIRE(cluster.stats_.rescored_vectors > 0);
    }

    SECTION("more candidates than clusters") {
        options.candidate_count = 64;
        auto labels = cluster.Run(4, datas.data(), count, options);
        require_balanced(labels, count, 4);
        REQUIRE(cluster.stats_.rescored_vectors == 0);
    }

    SECTION("k-means++ seeding over several reduction blocks") {
        constexpr uint64_t many = 9000;
        auto more = uniform_vectors(many, dim, 89);
        options.init_method = vsag::KMeansInitMethod::KMEANS_PLUS_PLUS;
        options.iterations = 3;
        auto labels = cluster.Run(k, more.data(), many, options);
        require_balanced(labels, many, k);
    }

    SECTION("k-means++ seeding of identical vectors") {
        const std::vector<float> same(count * dim, 0.25F);
        options.init_method = vsag::KMeansInitMethod::KMEANS_PLUS_PLUS;
        double err = -1.0;
        auto labels = cluster.Run(k, same.data(), count, options, &err);
        require_balanced(labels, count, k);
        REQUIRE(err < 1e-10);
    }

    SECTION("Assign with one centroid") {
        double cost = -1.0;
        auto labels = cluster.Assign(
            datas.data(), 1, datas.data(), count, vsag::BalancedAssignMetric::L2SQR, 16, &cost);
        require_balanced(labels, count, 1);
        REQUIRE(cost > 0.0);
    }
}

TEST_CASE("BalancedKMeans rejects invalid arguments", "[ut][BalancedKMeansCluster]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 4;
    constexpr uint64_t count = 100;
    auto datas = uniform_vectors(count, dim, 67);
    vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(dim), allocator.get());
    const vsag::BalancedKMeansOptions options;

    require_invalid_argument([&] { cluster.Run(0, datas.data(), count, options); });
    require_invalid_argument([&] { cluster.Run(4, datas.data(), 0, options); });
    require_invalid_argument([&] { cluster.Run(101, datas.data(), count, options); });
    require_invalid_argument([&] { cluster.Run(4, nullptr, count, options); });
    // More vectors than 32-bit ids can name, refused before any of them is read.
    require_invalid_argument([&] {
        cluster.Run(4, datas.data(), uint64_t{std::numeric_limits<uint32_t>::max()} + 1, options);
    });

    auto bad = options;
    bad.iterations = 0;
    require_invalid_argument([&] { cluster.Run(4, datas.data(), count, bad); });
    bad = options;
    bad.candidate_count = 0;
    require_invalid_argument([&] { cluster.Run(4, datas.data(), count, bad); });
    bad = options;
    bad.epsilon_ratio = 0.0;
    require_invalid_argument([&] { cluster.Run(4, datas.data(), count, bad); });
    bad = options;
    bad.epsilon_ratio = std::numeric_limits<double>::quiet_NaN();
    require_invalid_argument([&] { cluster.Run(4, datas.data(), count, bad); });
    bad = options;
    bad.tolerance = -1.0;
    require_invalid_argument([&] { cluster.Run(4, datas.data(), count, bad); });
    bad = options;
    bad.work_budget = -1.0;
    require_invalid_argument([&] { cluster.Run(4, datas.data(), count, bad); });
    bad = options;
    bad.work_budget = std::numeric_limits<double>::quiet_NaN();
    require_invalid_argument([&] { cluster.Run(4, datas.data(), count, bad); });
    bad = options;
    bad.init_method = static_cast<vsag::KMeansInitMethod>(7);
    require_invalid_argument([&] { cluster.Run(4, datas.data(), count, bad); });

    auto with_nan = datas;
    with_nan[13] = std::numeric_limits<float>::quiet_NaN();
    require_invalid_argument([&] { cluster.Run(4, with_nan.data(), count, options); });

    // Finite values whose squared norms overflow float distances are refused, for the
    // vectors and for the centroids; large values below the limit are accepted.
    vsag::BalancedKMeansCluster one_dim(1, allocator.get());
    const std::vector<float> huge{1e20F, 1e20F};
    const std::vector<float> zeros{0.0F, 0.0F};
    const auto l2_one = vsag::BalancedAssignMetric::L2SQR;
    require_invalid_argument([&] { one_dim.Run(1, huge.data(), 2, options); });
    require_invalid_argument([&] { one_dim.Assign(zeros.data(), 2, huge.data(), 2, l2_one); });
    require_invalid_argument([&] { one_dim.Assign(huge.data(), 2, zeros.data(), 2, l2_one); });
    require_invalid_argument(
        [&] { one_dim.Assign(zeros.data(), 2, huge.data(), 2, l2_one, 16, nullptr, 0.0); });
    const std::vector<float> large{1e18F, -1e18F};
    require_balanced(one_dim.Assign(zeros.data(), 2, large.data(), 2, l2_one), 2, 2);

    vsag::BalancedKMeansCluster no_dim(0, allocator.get());
    require_invalid_argument([&] { no_dim.Run(4, datas.data(), count, options); });
    require_invalid_argument([&] { vsag::BalancedKMeansCluster missing_allocator(4, nullptr); });

    const auto l2 = vsag::BalancedAssignMetric::L2SQR;
    require_invalid_argument([&] { cluster.Assign(nullptr, 4, datas.data(), count, l2); });
    require_invalid_argument([&] { cluster.Assign(datas.data(), 0, datas.data(), count, l2); });
    require_invalid_argument([&] { cluster.Assign(datas.data(), 4, nullptr, count, l2); });
    require_invalid_argument([&] { cluster.Assign(datas.data(), 4, datas.data(), 0, l2); });
    require_invalid_argument([&] { cluster.Assign(datas.data(), 4, datas.data(), count, l2, 0); });
    require_invalid_argument(
        [&] { cluster.Assign(datas.data(), 4, datas.data(), count, l2, 16, nullptr, -1.0); });
    require_invalid_argument([&] {
        cluster.Assign(
            datas.data(), 4, datas.data(), count, static_cast<vsag::BalancedAssignMetric>(9));
    });
}
