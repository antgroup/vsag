
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

    auto centroids = fixtures::generate_vectors(k, dim, false, 315);

    for (int64_t i = 0; i < count; ++i) {
        auto label = random() % k;
        for (int64_t j = 0; j < dim; ++j) {
            result[i * dim + j] = centroids[label * dim + j] + 0.0002;
        }
        labels[label]++;
    }
    std::sort(labels.begin(), labels.end());
    return result;
}

TEST_CASE("Kmeans Basic Test", "[ut][KMeansCluster]") {
    using namespace vsag;
    std::vector<int> labels;
    int32_t k = 10;
    int32_t dim = 3;
    uint64_t count = 2000;
    auto datas = GenerateDataset(k, dim, count, labels);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::vector<SafeThreadPoolPtr> thread_pools = {
        SafeThreadPool::FactoryDefaultThreadPool(),
        nullptr,
    };
    std::vector<bool> use_balances{true, false};
    for (auto& thread_pool : thread_pools) {
        for (bool use_balance : use_balances) {
            INFO(fmt::format("use_balance: {}", use_balance));
            INFO(fmt::format("thread_pool: {}", thread_pool == nullptr));
            KMeansCluster cluster(dim, allocator.get(), thread_pool);
            auto pos = cluster.Run(k, datas.data(), count, 50, nullptr, false, 1e-6F, use_balance);
            std::vector<int> new_labels(k, 0);
            for (int j = 0; j < count; ++j) {
                new_labels[pos[j]]++;
            }
            std::sort(new_labels.begin(), new_labels.end());
            for (int i = 0; i < k; ++i) {
                REQUIRE(new_labels[i] == labels[i]);
            }
        }
    }
}
