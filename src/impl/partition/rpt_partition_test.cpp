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

#include "rpt_partition.h"

#include <cmath>
#include <limits>
#include <random>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"
#include "vsag_exception.h"

namespace {

using Partitions = std::vector<std::vector<int64_t>>;

// Asserts the size guarantees documented on vsag::rpt_partition and that every id in
// [0, count) appears exactly once.
void
CheckPartitions(const Partitions& partitions, uint64_t count, uint64_t bucket_size) {
    if (count == 0) {
        REQUIRE(partitions.empty());
        return;
    }
    uint64_t expected_leaves = (count + bucket_size - 1) / bucket_size;
    REQUIRE(partitions.size() == expected_leaves);

    std::vector<uint8_t> seen(count, 0);
    uint64_t max_size = 0;
    uint64_t min_size = std::numeric_limits<uint64_t>::max();
    for (const auto& partition : partitions) {
        REQUIRE_FALSE(partition.empty());
        max_size = std::max(max_size, partition.size());
        min_size = std::min(min_size, partition.size());
        for (int64_t id : partition) {
            REQUIRE(id >= 0);
            REQUIRE(static_cast<uint64_t>(id) < count);
            REQUIRE(seen[id] == 0);
            seen[id] = 1;
        }
    }

    REQUIRE(max_size <= bucket_size);
    if (count >= bucket_size) {
        REQUIRE(min_size >= bucket_size / 2);
        REQUIRE(max_size <= 2 * min_size);
    }
}

Partitions
RunPartition(const std::vector<float>& datas,
             uint64_t dim,
             uint64_t count,
             const vsag::RPTPartitionParams& params) {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::RPTPartition partitioner(dim, allocator.get());
    return partitioner.Run(datas.data(), count, params);
}

}  // namespace

TEST_CASE("RPTPartition Size Invariants", "[ut][RPTPartition]") {
    uint64_t dim = 16;
    std::mt19937 rng(2024);
    std::uniform_int_distribution<uint64_t> count_dist(0, 5000);
    std::uniform_int_distribution<uint64_t> bucket_dist(1, 600);

    for (int round = 0; round < 40; ++round) {
        uint64_t count = count_dist(rng);
        vsag::RPTPartitionParams params;
        params.bucket_size = bucket_dist(rng);
        params.seed = round;
        auto datas = fixtures::generate_vectors(count, dim, /*need_normalize=*/false, round);

        auto partitions = RunPartition(datas, dim, count, params);
        CheckPartitions(partitions, count, params.bucket_size);
    }
}

TEST_CASE("RPTPartition Boundary Cases", "[ut][RPTPartition]") {
    uint64_t dim = 8;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::RPTPartition partitioner(dim, allocator.get());
    vsag::RPTPartitionParams params;
    params.bucket_size = 10;

    SECTION("empty input") {
        auto partitions = partitioner.Run(nullptr, 0, params);
        REQUIRE(partitions.empty());
    }

    SECTION("fewer vectors than bucket size") {
        uint64_t count = 7;
        auto datas = fixtures::generate_vectors(count, dim);
        auto partitions = partitioner.Run(datas.data(), count, params);
        REQUIRE(partitions.size() == 1);
        REQUIRE(partitions[0].size() == count);
    }

    SECTION("bucket size of one") {
        uint64_t count = 25;
        params.bucket_size = 1;
        auto datas = fixtures::generate_vectors(count, dim);
        auto partitions = partitioner.Run(datas.data(), count, params);
        CheckPartitions(partitions, count, params.bucket_size);
    }

    SECTION("bucket size of zero") {
        auto datas = fixtures::generate_vectors(4, dim);
        params.bucket_size = 0;
        REQUIRE_THROWS_AS(partitioner.Run(datas.data(), 4, params), vsag::VsagException);
    }

    SECTION("null data with positive count") {
        REQUIRE_THROWS_AS(partitioner.Run(nullptr, 4, params), vsag::VsagException);
    }

    SECTION("duplicated vectors") {
        uint64_t count = 1000;
        std::vector<float> datas(count * dim, 1.5F);
        auto partitions = partitioner.Run(datas.data(), count, params);
        CheckPartitions(partitions, count, params.bucket_size);
    }

    SECTION("nan and inf values") {
        uint64_t count = 500;
        auto datas = fixtures::generate_vectors(count, dim);
        datas[3] = std::numeric_limits<float>::quiet_NaN();
        datas[dim * 7 + 1] = std::numeric_limits<float>::infinity();
        datas[dim * 8 + 2] = -std::numeric_limits<float>::infinity();
        auto partitions = partitioner.Run(datas.data(), count, params);
        CheckPartitions(partitions, count, params.bucket_size);
    }

    SECTION("max depth reached") {
        uint64_t count = 1234;
        params.max_depth = 2;
        auto datas = fixtures::generate_vectors(count, dim);
        auto partitions = partitioner.Run(datas.data(), count, params);
        CheckPartitions(partitions, count, params.bucket_size);
    }

    SECTION("zero max depth") {
        uint64_t count = 95;
        params.max_depth = 0;
        auto datas = fixtures::generate_vectors(count, dim);
        auto partitions = partitioner.Run(datas.data(), count, params);
        CheckPartitions(partitions, count, params.bucket_size);
    }
}

TEST_CASE("RPTPartition Reproducibility", "[ut][RPTPartition]") {
    uint64_t dim = 32;
    uint64_t count = 3000;
    auto datas = fixtures::generate_vectors(count, dim);
    vsag::RPTPartitionParams params;
    params.bucket_size = 100;
    params.seed = 42;

    auto first = RunPartition(datas, dim, count, params);
    auto second = RunPartition(datas, dim, count, params);
    REQUIRE(first == second);

    params.seed = 43;
    auto other = RunPartition(datas, dim, count, params);
    CheckPartitions(other, count, params.bucket_size);
    REQUIRE(first != other);
}

TEST_CASE("RPTPartition Preserves Locality", "[ut][RPTPartition]") {
    // Well separated gaussian clusters, each smaller than a bucket. A random projection
    // tree should keep most of each cluster together, while a random assignment would
    // spread every cluster across all buckets.
    uint64_t dim = 16;
    uint64_t clusters = 8;
    uint64_t per_cluster = 50;
    uint64_t count = clusters * per_cluster;
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0F, 0.01F);
    std::vector<float> datas(count * dim, 0.0F);
    for (uint64_t c = 0; c < clusters; ++c) {
        for (uint64_t i = 0; i < per_cluster; ++i) {
            float* vec = datas.data() + (c * per_cluster + i) * dim;
            vec[c % dim] = 10.0F;
            for (uint64_t j = 0; j < dim; ++j) {
                vec[j] += noise(rng);
            }
        }
    }

    vsag::RPTPartitionParams params;
    params.bucket_size = 100;
    auto partitions = RunPartition(datas, dim, count, params);
    CheckPartitions(partitions, count, params.bucket_size);

    // Fraction of same-cluster pairs that end up in the same partition.
    std::vector<uint64_t> partition_of(count);
    for (uint64_t p = 0; p < partitions.size(); ++p) {
        for (int64_t id : partitions[p]) {
            partition_of[id] = p;
        }
    }
    uint64_t same = 0;
    uint64_t total = 0;
    for (uint64_t c = 0; c < clusters; ++c) {
        for (uint64_t i = 0; i < per_cluster; ++i) {
            for (uint64_t j = i + 1; j < per_cluster; ++j) {
                if (partition_of[c * per_cluster + i] == partition_of[c * per_cluster + j]) {
                    ++same;
                }
                ++total;
            }
        }
    }
    double same_ratio = static_cast<double>(same) / static_cast<double>(total);
    // Random assignment into 4 buckets would give about 0.25.
    REQUIRE(same_ratio > 0.6);
}

TEST_CASE("RPTPartition Public API", "[ut][RPTPartition]") {
    uint64_t dim = 8;
    uint64_t count = 333;
    auto datas = fixtures::generate_vectors(count, dim);
    vsag::RPTPartitionParams params;
    params.bucket_size = 50;

    auto result = vsag::rpt_partition(dim, count, datas.data(), params);
    REQUIRE(result.has_value());
    CheckPartitions(result.value(), count, params.bucket_size);

    auto empty = vsag::rpt_partition(dim, 0, nullptr, params);
    REQUIRE(empty.has_value());
    REQUIRE(empty->empty());

    auto zero_dim = vsag::rpt_partition(0, count, datas.data(), params);
    REQUIRE_FALSE(zero_dim.has_value());
    REQUIRE(zero_dim.error().type == vsag::ErrorType::INVALID_ARGUMENT);

    params.bucket_size = 0;
    auto zero_bucket = vsag::rpt_partition(dim, count, datas.data(), params);
    REQUIRE_FALSE(zero_bucket.has_value());
    REQUIRE(zero_bucket.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}
