
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
