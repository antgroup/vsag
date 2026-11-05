
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

#include "kmedoids_cluster.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"
#include "vsag_exception.h"

namespace {
std::vector<float>
GenerateBlobs(uint64_t blob_count, uint64_t dim, uint64_t per_blob, float distance, float noise) {
    std::vector<float> result(blob_count * per_blob * dim, 0.0F);
    std::mt19937 gen(1234);
    std::normal_distribution<float> distribution(0.0F, noise);
    for (uint64_t blob = 0; blob < blob_count; ++blob) {
        for (uint64_t point = 0; point < per_blob; ++point) {
            const auto row = blob * per_blob + point;
            for (uint64_t d = 0; d < dim; ++d) {
                result[row * dim + d] = (d == blob ? distance : 0.0F) + distribution(gen);
            }
        }
    }
    return result;
}

void
RequireDataPointCentroids(const vsag::KMedoidsCluster& cluster,
                          const std::vector<float>& datas,
                          uint64_t dim) {
    for (uint64_t c = 0; c < cluster.medoid_ids_.size(); ++c) {
        REQUIRE(cluster.medoid_ids_[c] >= 0);
        const auto id = static_cast<uint64_t>(cluster.medoid_ids_[c]);
        REQUIRE(std::memcmp(cluster.k_centroids_ + c * dim,
                            datas.data() + id * dim,
                            dim * sizeof(float)) == 0);
    }
}
}  // namespace

TEST_CASE("KMedoids returns input data points", "[ut][KMedoidsCluster]") {
    constexpr uint64_t k = 3;
    constexpr uint64_t dim = 16;
    constexpr uint64_t count_per_cluster = 200;
    auto datas = GenerateBlobs(k, dim, count_per_cluster, 20.0F, 0.5F);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::KMedoidsCluster cluster(static_cast<int32_t>(dim), allocator.get());
    double error = 0.0;
    auto labels = cluster.Run(k,
                              datas.data(),
                              datas.size() / dim,
                              10,
                              &error,
                              1e-6F,
                              vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                              12345);
    REQUIRE(labels.size() == datas.size() / dim);
    REQUIRE(error >= 0.0);
    RequireDataPointCentroids(cluster, datas, dim);
}

TEST_CASE("KMedoids is deterministic with a seed", "[ut][KMedoidsCluster]") {
    constexpr uint64_t k = 3;
    constexpr uint64_t dim = 16;
    auto datas = GenerateBlobs(k, dim, 100, 20.0F, 0.75F);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::KMedoidsCluster first(static_cast<int32_t>(dim), allocator.get());
    vsag::KMedoidsCluster second(static_cast<int32_t>(dim), allocator.get());
    double first_error = 0.0;
    double second_error = 0.0;
    auto first_labels = first.Run(k,
                                  datas.data(),
                                  datas.size() / dim,
                                  10,
                                  &first_error,
                                  1e-6F,
                                  vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                  7);
    auto second_labels = second.Run(k,
                                    datas.data(),
                                    datas.size() / dim,
                                    10,
                                    &second_error,
                                    1e-6F,
                                    vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                    7);
    REQUIRE(first_labels == second_labels);
    REQUIRE(first.medoid_ids_ == second.medoid_ids_);
    REQUIRE(std::fabs(first_error - second_error) <= 1e-6 * std::max(1.0, first_error));
}

TEST_CASE("KMedoids validates input", "[ut][KMedoidsCluster]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::KMedoidsCluster cluster(2, allocator.get());
    const float data[] = {0.0F, 0.0F, 1.0F, 1.0F};
    REQUIRE_THROWS_AS(cluster.Run(0, data, 2), vsag::VsagException);
    REQUIRE_THROWS_AS(cluster.Run(1, data, 0), vsag::VsagException);
    REQUIRE_THROWS_AS(cluster.Run(1, nullptr, 2), vsag::VsagException);
    REQUIRE_THROWS_AS(cluster.Run(3, data, 2), vsag::VsagException);
}

TEST_CASE("KMedoids handles one and all-point clusters", "[ut][KMedoidsCluster]") {
    constexpr uint64_t dim = 2;
    constexpr uint64_t count = 6;
    const std::vector<float> datas = {
        0.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F, 4.0F, 0.0F, 5.0F, 0.0F};
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::KMedoidsCluster one(static_cast<int32_t>(dim), allocator.get());
    auto one_labels = one.Run(
        1, datas.data(), count, 5, nullptr, 1e-6F, vsag::KMeansInitMethod::KMEANS_PLUS_PLUS, 11);
    REQUIRE(one_labels.size() == count);
    RequireDataPointCentroids(one, datas, dim);

    vsag::KMedoidsCluster all(static_cast<int32_t>(dim), allocator.get());
    auto all_labels = all.Run(count,
                              datas.data(),
                              count,
                              5,
                              nullptr,
                              1e-6F,
                              vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                              11);
    REQUIRE(all_labels.size() == count);
    RequireDataPointCentroids(all, datas, dim);
}
