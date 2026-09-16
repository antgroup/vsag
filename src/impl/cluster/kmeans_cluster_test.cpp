
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

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"
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
