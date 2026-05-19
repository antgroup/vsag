
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

#include "multi_vector_computer.h"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

using namespace vsag;

namespace {

constexpr float kEpsilon = 1e-5F;

// Naive oracle: for each query token, find the minimum distance among all doc tokens and
// accumulate the results. For METRIC_TYPE_IP, use 1 - <q,d>; for METRIC_TYPE_L2SQR,
// use ||q-d||^2.
float
NaiveMaxSim(const std::vector<float>& query,
            uint32_t query_token_count,
            const std::vector<float>& doc,
            uint32_t doc_token_count,
            uint32_t dim,
            MetricType metric) {
    float total = 0.0F;
    for (uint32_t q = 0; q < query_token_count; ++q) {
        const float* q_tok = query.data() + q * dim;
        float min_dist = std::numeric_limits<float>::max();
        for (uint32_t d = 0; d < doc_token_count; ++d) {
            const float* d_tok = doc.data() + d * dim;
            float dist_val = 0.0F;
            if (metric == MetricType::METRIC_TYPE_IP) {
                float ip = 0.0F;
                for (uint32_t i = 0; i < dim; ++i) {
                    ip += q_tok[i] * d_tok[i];
                }
                dist_val = 1.0F - ip;
            } else {
                float l2 = 0.0F;
                for (uint32_t i = 0; i < dim; ++i) {
                    float diff = q_tok[i] - d_tok[i];
                    l2 += diff * diff;
                }
                dist_val = l2;
            }
            if (dist_val < min_dist) {
                min_dist = dist_val;
            }
        }
        total += min_dist;
    }
    return total;
}

}  // namespace

TEST_CASE("MultiVectorComputer hand-computed oracle (dim=4, q=2, d=3)",
          "[ut][MultiVectorComputer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint32_t dim = 4;
    constexpr uint32_t query_tok_count = 2;
    constexpr uint32_t doc_tok_count = 3;

    // query: q0=(1,0,0,0), q1=(0,1,1,0)
    std::vector<float> query = {
        1.0F,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        1.0F,
        0.0F,
    };
    // doc:   d0=(1,0,0,0), d1=(0,2,0,0), d2=(0,0,1,0)
    std::vector<float> doc = {
        1.0F,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        2.0F,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };

    SECTION("IP metric: hand-computed expected = -1") {
        // q0: IP=[1,0,0] -> dist=[0,1,1] -> min=0
        // q1: IP=[0,2,1] -> dist=[1,-1,0] -> min=-1
        // total = -1
        MultiVectorComputer computer(dim, MetricType::METRIC_TYPE_IP, allocator.get());
        computer.SetQuery(query.data(), query_tok_count);

        float dist = 0.0F;
        computer.ComputeDist(reinterpret_cast<const uint8_t*>(doc.data()), doc_tok_count, &dist);

        REQUIRE(std::abs(dist - (-1.0F)) < kEpsilon);

        float oracle = NaiveMaxSim(
            query, query_tok_count, doc, doc_tok_count, dim, MetricType::METRIC_TYPE_IP);
        REQUIRE(std::abs(dist - oracle) < kEpsilon);
    }

    SECTION("L2 metric: hand-computed expected = 1") {
        // q0 L2 with doc: 0, 5, 2 -> min=0
        // q1 L2 with doc: 3, 2, 1 -> min=1
        // total = 1
        MultiVectorComputer computer(dim, MetricType::METRIC_TYPE_L2SQR, allocator.get());
        computer.SetQuery(query.data(), query_tok_count);

        float dist = 0.0F;
        computer.ComputeDist(reinterpret_cast<const uint8_t*>(doc.data()), doc_tok_count, &dist);

        REQUIRE(std::abs(dist - 1.0F) < kEpsilon);

        float oracle = NaiveMaxSim(
            query, query_tok_count, doc, doc_tok_count, dim, MetricType::METRIC_TYPE_L2SQR);
        REQUIRE(std::abs(dist - oracle) < kEpsilon);
    }
}

TEST_CASE("MultiVectorComputer accessors", "[ut][MultiVectorComputer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    MultiVectorComputer computer(8, MetricType::METRIC_TYPE_IP, allocator.get());

    REQUIRE(computer.GetDim() == 8);
    REQUIRE(computer.GetMetric() == MetricType::METRIC_TYPE_IP);
    REQUIRE(computer.GetQueryTokenCount() == 0);

    std::vector<float> query(16, 0.5F);
    computer.SetQuery(query.data(), 2);
    REQUIRE(computer.GetQueryTokenCount() == 2);
}

TEST_CASE("MultiVectorComputer rebind query overwrites previous tokens",
          "[ut][MultiVectorComputer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint32_t dim = 2;
    MultiVectorComputer computer(dim, MetricType::METRIC_TYPE_L2SQR, allocator.get());

    // First query: 3 tokens
    std::vector<float> query_a = {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    computer.SetQuery(query_a.data(), 3);
    REQUIRE(computer.GetQueryTokenCount() == 3);

    // Rebind to a smaller query: 1 token
    std::vector<float> query_b = {2.0F, 2.0F};
    computer.SetQuery(query_b.data(), 1);
    REQUIRE(computer.GetQueryTokenCount() == 1);

    // doc: single token (3, 3)
    std::vector<float> doc = {3.0F, 3.0F};
    float dist = 0.0F;
    computer.ComputeDist(reinterpret_cast<const uint8_t*>(doc.data()), 1, &dist);

    // L2: ||(2,2)-(3,3)||^2 = 1 + 1 = 2
    REQUIRE(std::abs(dist - 2.0F) < kEpsilon);
}

TEST_CASE("MultiVectorComputer boundary: query_token_count=1, doc_token_count=1, dim=1",
          "[ut][MultiVectorComputer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    SECTION("IP, dim=1") {
        MultiVectorComputer computer(1, MetricType::METRIC_TYPE_IP, allocator.get());
        std::vector<float> query = {3.0F};
        std::vector<float> doc = {2.0F};
        computer.SetQuery(query.data(), 1);
        float dist = 0.0F;
        computer.ComputeDist(reinterpret_cast<const uint8_t*>(doc.data()), 1, &dist);
        // 1 - 3*2 = -5
        REQUIRE(std::abs(dist - (-5.0F)) < kEpsilon);
    }

    SECTION("L2, dim=1") {
        MultiVectorComputer computer(1, MetricType::METRIC_TYPE_L2SQR, allocator.get());
        std::vector<float> query = {3.0F};
        std::vector<float> doc = {2.0F};
        computer.SetQuery(query.data(), 1);
        float dist = 0.0F;
        computer.ComputeDist(reinterpret_cast<const uint8_t*>(doc.data()), 1, &dist);
        // (3-2)^2 = 1
        REQUIRE(std::abs(dist - 1.0F) < kEpsilon);
    }
}

TEST_CASE("MultiVectorComputer randomized vs naive oracle (dim=8, q=4, d=6)",
          "[ut][MultiVectorComputer]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint32_t dim = 8;
    constexpr uint32_t query_tok_count = 4;
    constexpr uint32_t doc_tok_count = 6;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_uniform(-1.0F, 1.0F);

    std::vector<float> query(query_tok_count * dim);
    std::vector<float> doc(doc_tok_count * dim);
    for (auto& v : query) {
        v = dist_uniform(rng);
    }
    for (auto& v : doc) {
        v = dist_uniform(rng);
    }

    for (auto metric : {MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_L2SQR}) {
        MultiVectorComputer computer(dim, metric, allocator.get());
        computer.SetQuery(query.data(), query_tok_count);

        float actual = 0.0F;
        computer.ComputeDist(reinterpret_cast<const uint8_t*>(doc.data()), doc_tok_count, &actual);

        float expected = NaiveMaxSim(query, query_tok_count, doc, doc_tok_count, dim, metric);
        REQUIRE(std::abs(actual - expected) < 1e-4F);
    }
}
