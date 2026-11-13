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

#include "e_means_cluster.h"

#include <algorithm>
#include <random>

#include "impl/allocator/safe_allocator.h"
#include "kmeans_cluster.h"
#include "unittest.h"

namespace {

// Gaussian blobs around well-separated centers.
std::vector<float>
generate_blobs(uint32_t k, uint32_t dim, uint64_t count, std::vector<int>& truth) {
    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0.0F, 0.05F);
    std::uniform_real_distribution<float> center(-10.0F, 10.0F);
    std::vector<float> centers(static_cast<uint64_t>(k) * dim);
    for (auto& c : centers) {
        c = center(gen);
    }
    std::vector<float> data(count * dim);
    truth.assign(count, 0);
    for (uint64_t i = 0; i < count; ++i) {
        truth[i] = static_cast<int>(i % k);
        for (uint32_t d = 0; d < dim; ++d) {
            data[i * dim + d] = centers[static_cast<uint64_t>(truth[i]) * dim + d] + noise(gen);
        }
    }
    return data;
}

}  // namespace

// Over-clustering well-separated blobs: with random seeding a blob can still
// end up without a centroid, so require that some seed splits no label across
// two blobs and reaches a noise-level error.
TEST_CASE("EMeans Separates Distinct Blobs", "[ut][EMeansCluster]") {
    const uint32_t blobs = 8;
    const uint32_t k = 32;
    const uint32_t dim = 32;
    const uint64_t count = 8192;
    std::vector<int> truth;
    auto data = generate_blobs(blobs, dim, count, truth);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    bool pure = false;
    for (uint32_t seed = 1; seed <= 8 && not pure; ++seed) {
        vsag::EMeansCluster cluster(dim, allocator.get());
        double err = 0;
        auto labels = cluster.Run(k, data.data(), count, 5, &err, seed);
        REQUIRE(labels.size() == count);

        std::vector<int> owner(k, -1);
        bool ok = true;
        for (uint64_t i = 0; i < count && ok; ++i) {
            auto& o = owner[labels[i]];
            if (o < 0) {
                o = truth[i];
            }
            ok = o == truth[i];
        }
        if (ok) {
            REQUIRE(err < 2.0 * dim * 0.05 * 0.05 * 4.0);
            pure = true;
        }
    }
    REQUIRE(pure);
}

TEST_CASE("EMeans Is Deterministic For A Fixed Seed", "[ut][EMeansCluster]") {
    const uint32_t k = 8;
    const uint32_t dim = 16;
    const uint64_t count = 2048;
    std::vector<int> truth;
    auto data = generate_blobs(k, dim, count, truth);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    vsag::EMeansCluster a(dim, allocator.get());
    vsag::EMeansCluster b(dim, allocator.get());
    double err_a = 0;
    double err_b = 0;
    auto la = a.Run(k, data.data(), count, 3, &err_a, 11);
    auto lb = b.Run(k, data.data(), count, 3, &err_b, 11);
    REQUIRE(std::equal(la.begin(), la.end(), lb.begin()));
    REQUIRE(err_a == err_b);
}

// Dead-cluster revival must keep every cluster populated even though the random
// seeding can place several initial centroids inside one blob.
TEST_CASE("EMeans Leaves No Empty Cluster", "[ut][EMeansCluster]") {
    const uint32_t k = 32;
    const uint32_t dim = 8;
    const uint64_t count = 8192;
    std::vector<int> truth;
    auto data = generate_blobs(k, dim, count, truth);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    for (uint32_t seed = 1; seed <= 3; ++seed) {
        vsag::EMeansCluster cluster(dim, allocator.get());
        auto labels = cluster.Run(k, data.data(), count, 5, nullptr, seed, 512);
        std::vector<int> sizes(k, 0);
        for (auto l : labels) {
            sizes[l]++;
        }
        REQUIRE(*std::min_element(sizes.begin(), sizes.end()) > 0);
    }
}

// At 3 epochs e-means should be no worse than Lloyd k-means at 25 iterations
// (both from random seeding), on data with overlapping clusters.
TEST_CASE("EMeans Matches Lloyd At Fewer Epochs", "[ut][EMeansCluster]") {
    const uint32_t k = 64;
    const uint32_t dim = 32;
    const uint64_t count = 32768;
    std::mt19937 gen(5);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> data(count * dim);
    for (auto& v : data) {
        v = normal(gen);
    }
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    vsag::KMeansCluster lloyd(dim, allocator.get());
    double lloyd_err = 0;
    lloyd.Run(
        k, data.data(), count, 25, &lloyd_err, false, 0.0F, vsag::KMeansInitMethod::RANDOM, 3U);

    vsag::EMeansCluster e(dim, allocator.get());
    double e_err = 0;
    e.Run(k, data.data(), count, 3, &e_err, 3);

    REQUIRE(e_err <= lloyd_err * 1.01);
}

TEST_CASE("EMeans Default Hyper-parameters", "[ut][EMeansCluster]") {
    auto b = vsag::EMeansCluster::DefaultBatchSize(250000, 1024, 3);
    REQUIRE(b == 16384);
    REQUIRE((b & (b - 1)) == 0);
    REQUIRE(vsag::EMeansCluster::DefaultBatchSize(100, 8, 3) <= 100);
    // n > 2M divides the batch law by 8.
    REQUIRE(vsag::EMeansCluster::DefaultBatchSize(4'000'000, 1024, 3) <
            vsag::EMeansCluster::DefaultBatchSize(2'000'000, 1024, 3) * 8);

    auto alpha = vsag::EMeansCluster::DefaultPeakStep(250000, 1024, 3, b);
    REQUIRE(alpha > 0.0);
    REQUIRE(alpha <= 1.0);
    REQUIRE(vsag::EMeansCluster::DefaultPeakStep(1000, 2, 25, 1000) == 1.0);
}

TEST_CASE("EMeans Rejects Invalid Arguments", "[ut][EMeansCluster]") {
    const uint32_t dim = 4;
    std::vector<float> data(10ULL * dim, 1.0F);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::EMeansCluster cluster(dim, allocator.get());

    REQUIRE_THROWS(cluster.Run(0, data.data(), 10));
    REQUIRE_THROWS(cluster.Run(2, data.data(), 0));
    REQUIRE_THROWS(cluster.Run(2, nullptr, 10));
    REQUIRE_THROWS(cluster.Run(20, data.data(), 10));
    REQUIRE_THROWS(cluster.Run(2, data.data(), 10, 0));
    REQUIRE_THROWS(cluster.Run(2, data.data(), 10, 3, nullptr, 1, 0, 1.5));
}
