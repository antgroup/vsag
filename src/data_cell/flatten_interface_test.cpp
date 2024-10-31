
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

#include "flatten_interface_test.h"

#include <fstream>

#include "catch2/catch_template_test_macros.hpp"
#include "fixtures.h"
#include "simd/simd.h"

namespace vsag {
void
FlattenInterfaceTest::BasicTest(int64_t dim, uint64_t base_count, float error) {
    int64_t query_count = 100;
    auto vectors = fixtures::generate_vectors(base_count, dim);
    auto querys = fixtures::generate_vectors(query_count, dim, random());

    auto old_count = flatten_->TotalCount();
    flatten_->Train(vectors.data(), base_count);
    flatten_->BatchInsertVector(vectors.data(), base_count);
    REQUIRE(flatten_->TotalCount() == base_count + old_count);

    std::vector<InnerIdType> idx(base_count);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device()()));
    std::vector<float> dists(base_count);
    for (int64_t i = 0; i < query_count; ++i) {
        auto computer = flatten_->FactoryComputer(querys.data() + i * dim);
        flatten_->Query(dists.data(), computer, idx.data(), base_count);
        float gt;
        for (int64_t j = 0; j < base_count; ++j) {
            if (metric_ == vsag::MetricType::METRIC_TYPE_IP ||
                metric_ == vsag::MetricType::METRIC_TYPE_COSINE) {
                gt = 1 - InnerProduct(vectors.data() + idx[j] * dim, querys.data() + i * dim, &dim);
            } else if (metric_ == vsag::MetricType::METRIC_TYPE_L2SQR) {
                gt = L2Sqr(vectors.data() + idx[j] * dim, querys.data() + i * dim, &dim);
            }
            REQUIRE(std::abs(gt - dists[j]) < error);
        }
    }

    for (int64_t i = 0; i < query_count; ++i) {
        auto idx1 = random() % base_count;
        auto idx2 = random() % base_count;
        auto value = flatten_->ComputePairVectors(idx1, idx2);
        float gt = 1.0f;

        if (metric_ == vsag::MetricType::METRIC_TYPE_IP ||
            metric_ == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = 1 - InnerProduct(vectors.data() + idx1 * dim, vectors.data() + idx2 * dim, &dim);
        } else if (metric_ == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = L2Sqr(vectors.data() + idx1 * dim, vectors.data() + idx2 * dim, &dim);
        }
        REQUIRE(std::abs(gt - value) < error);
    }
}
void
FlattenInterfaceTest::TestSerializeAndDeserialize(int64_t dim,
                                                  FlattenInterfacePtr other,
                                                  float error) {
    fixtures::TempDir dir("flatten");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path.c_str(), std::ios::binary);
    IOStreamWriter writer(outfile);
    this->flatten_->Serialize(writer);
    outfile.close();

    std::ifstream infile(path.c_str(), std::ios::binary);
    IOStreamReader reader(infile);
    other->Deserialize(reader);

    int64_t query_count = 100;
    auto querys = fixtures::generate_vectors(query_count, dim, random());

    auto total_count = other->TotalCount();
    REQUIRE(total_count == this->flatten_->TotalCount());

    // Test Query
    {
        std::vector<float> dists1(total_count), dists2(total_count);
        std::vector<InnerIdType> idx(total_count);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device()()));
        for (int64_t i = 0; i < query_count; ++i) {
            auto computer = flatten_->FactoryComputer(querys.data() + i * dim);
            flatten_->Query(dists1.data(), computer, idx.data(), total_count);
            other->Query(dists2.data(), computer, idx.data(), total_count);
            for (int64_t j = 0; j < total_count; ++j) {
                REQUIRE(dists1[j] == dists2[j]);
            }
        }
    }

    // Test Compute pair vector
    {
        for (int64_t i = 0; i < query_count; ++i) {
            auto idx1 = random() % total_count;
            auto idx2 = random() % total_count;
            auto value1 = flatten_->ComputePairVectors(idx1, idx2);
            auto value2 = other->ComputePairVectors(idx1, idx2);
            REQUIRE(value1 == value2);
        }
    }

    // Test Add more
    {
        auto base_count = 100;
        auto vectors = fixtures::generate_vectors(base_count, dim, random());
        other->BatchInsertVector(vectors.data(), base_count);
        std::vector<InnerIdType> idx(base_count);
        std::iota(idx.begin(), idx.end(), total_count);
        std::vector<float> dists(base_count);
        for (int64_t i = 0; i < query_count; ++i) {
            auto computer = other->FactoryComputer(querys.data() + i * dim);
            other->Query(dists.data(), computer, idx.data(), base_count);
            float gt;
            for (int64_t j = 0; j < base_count; ++j) {
                if (metric_ == vsag::MetricType::METRIC_TYPE_IP ||
                    metric_ == vsag::MetricType::METRIC_TYPE_COSINE) {
                    gt = 1 - InnerProduct(vectors.data() + (idx[j] - total_count) * dim,
                                          querys.data() + i * dim,
                                          &dim);
                } else if (metric_ == vsag::MetricType::METRIC_TYPE_L2SQR) {
                    gt = L2Sqr(vectors.data() + (idx[j] - total_count) * dim,
                               querys.data() + i * dim,
                               &dim);
                }
                REQUIRE(std::abs(gt - dists[j]) < error);
            }
        }
    }

    infile.close();
}
}  // namespace vsag
