
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

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"

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
