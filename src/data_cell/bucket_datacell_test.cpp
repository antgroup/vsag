
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

#include "bucket_datacell.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <utility>

#include "fixtures.h"
#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "simd/simd.h"

using namespace vsag;

namespace vsag {
class BucketInterfaceTest {
public:
    BucketInterfaceTest(BucketInterfacePtr bucket, MetricType metric)
        : bucket_(std::move(bucket)), metric_(metric){};

    void
    BasicTest(int64_t dim, uint64_t base_count, float error = 1e-5f);

    void
    TestSerializeAndDeserialize(int64_t dim, const BucketInterfacePtr& other);

public:
    BucketInterfacePtr bucket_{nullptr};

    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
};
}  // namespace vsag

void
BucketInterfaceTest::BasicTest(int64_t dim, uint64_t base_count, float error) {
    int64_t query_count = 100;
    auto vectors = fixtures::generate_vectors(base_count, dim);
    auto queries = fixtures::generate_vectors(query_count, dim, random());
    bucket_->Train(vectors.data(), base_count);
    auto bucket_count = bucket_->GetBucketCount();
    for (int64_t i = 0; i < base_count; ++i) {
        auto bucket_id = random() % bucket_count;
        bucket_->InsertVector(vectors.data() + i * dim, bucket_id, i);
    }

    std::vector<float> dists(base_count);
    for (int64_t i = 0; i < query_count; ++i) {
        auto computer = bucket_->FactoryComputer(queries.data() + i * dim);
        auto* dist = dists.data();
        for (auto bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
            // Test ScanBucketById
            bucket_->ScanBucketById(dist, computer, bucket_id);
            auto bucket_size = bucket_->GetBucketSize(bucket_id);
            const auto* labels = bucket_->GetInnerIds(bucket_id);

            float gt;
            for (int64_t j = 0; j < bucket_size; ++j) {
                if (metric_ == vsag::MetricType::METRIC_TYPE_IP or
                    metric_ == vsag::MetricType::METRIC_TYPE_COSINE) {
                    gt = 1 - InnerProduct(
                                 vectors.data() + labels[j] * dim, queries.data() + i * dim, &dim);
                } else if (metric_ == vsag::MetricType::METRIC_TYPE_L2SQR) {
                    gt = L2Sqr(vectors.data() + labels[j] * dim, queries.data() + i * dim, &dim);
                }
                REQUIRE(std::abs(gt - dist[j]) < error);
                // Test QueryOneById
                bucket_->Prefetch(bucket_id, j);
                auto point_dist = bucket_->QueryOneById(computer, bucket_id, j);
                REQUIRE(point_dist == dist[j]);
            }
            dist += bucket_size;
        }
        // exceptions
        REQUIRE_THROWS(bucket_->ScanBucketById(dist, computer, bucket_count * 2));
        REQUIRE_THROWS(bucket_->QueryOneById(computer, bucket_count * 2, 0));
        REQUIRE_THROWS(bucket_->QueryOneById(computer, 0, 10000));
    }

    // exceptions
    REQUIRE_THROWS(bucket_->InsertVector(vectors.data() + 1 * dim, bucket_count, 98));
}
void
BucketInterfaceTest::TestSerializeAndDeserialize(int64_t dim, const BucketInterfacePtr& other) {
    fixtures::TempDir dir("bucket");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path.c_str(), std::ios::binary);
    IOStreamWriter writer(outfile);
    this->bucket_->Serialize(writer);
    outfile.close();

    std::ifstream infile(path.c_str(), std::ios::binary);
    IOStreamReader reader(infile);
    other->Deserialize(reader);

    int64_t query_count = 100;
    auto queries = fixtures::generate_vectors(query_count, dim, random());

    auto bucket_count = other->GetBucketCount();
    REQUIRE(bucket_count == this->bucket_->GetBucketCount());

    for (BucketIdType bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
        auto bucket_size = this->bucket_->GetBucketSize(bucket_id);
        REQUIRE(bucket_size == other->GetBucketSize(bucket_id));
        const auto* labels = this->bucket_->GetInnerIds(bucket_id);
        const auto* other_labels = this->bucket_->GetInnerIds(bucket_id);
        for (int64_t i = 0; i < bucket_size; ++i) {
            REQUIRE(labels[i] == other_labels[i]);
        }
        std::vector<float> dists_1(bucket_size);
        std::vector<float> dists_2(bucket_size);

        for (int64_t i = 0; i < query_count; ++i) {
            auto computer = bucket_->FactoryComputer(queries.data() + i * dim);
            this->bucket_->ScanBucketById(dists_1.data(), computer, bucket_id);
            other->ScanBucketById(dists_2.data(), computer, bucket_id);
            for (int64_t j = 0; j < bucket_size; ++j) {
                REQUIRE(dists_1[j] == dists_2[j]);
            }
        }
    }

    infile.close();
}

void
TestBucketDataCell(BucketDataCellParamPtr& param,
                   IndexCommonParam& common_param,
                   float error = 1e-5) {
    auto count = GENERATE(100, 1000);
    auto bucket = BucketInterface::MakeInstance(param, common_param);

    BucketInterfaceTest test(bucket, common_param.metric_);
    test.BasicTest(common_param.dim_, count, error);
    auto other = BucketInterface::MakeInstance(param, common_param);
    test.TestSerializeAndDeserialize(common_param.dim_, other);
}

TEST_CASE("BucketDataCell Basic Test", "[ut][BucketDataCell] ") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64, 512);
    std::string io_type = GENERATE("memory_io", "block_memory_io");
    std::vector<std::pair<std::string, float>> quantizer_errors = {
        {"sq8", 2e-2f},
        {"fp32", 1e-5},
    };
    auto bucket_count = GENERATE(10, 20);
    MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    constexpr const char* param_temp =
        R"(
        {{
            "io_params": {{
                "type": "{}"
            }},
            "quantization_params": {{
                "type": "{}"
            }},
            "buckets_count": {}
        }}
        )";
    for (auto& quantizer_error : quantizer_errors) {
        for (auto& metric : metrics) {
            auto param_str = fmt::format(param_temp, io_type, quantizer_error.first, bucket_count);
            auto param_json = JsonType::parse(param_str);
            auto param = std::make_shared<BucketDataCellParameter>();
            param->FromJson(param_json);
            IndexCommonParam common_param;
            common_param.allocator_ = allocator;
            common_param.dim_ = dim;
            common_param.metric_ = metric;

            TestBucketDataCell(param, common_param, quantizer_error.second);
        }
    }
}
