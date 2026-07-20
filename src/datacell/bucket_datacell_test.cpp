
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
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "simd/simd.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

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
    test_serializion(*this->bucket_, *other);

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
}

void
TestBucketDataCell(BucketDataCellParamPtr& param1,
                   BucketDataCellParamPtr& param2,
                   IndexCommonParam& common_param,
                   float error = 1e-5) {
    auto count = GENERATE(100, 1000);
    auto bucket = BucketInterface::MakeInstance(param1, common_param);

    BucketInterfaceTest test(bucket, common_param.metric_);
    test.BasicTest(common_param.dim_, count, error);
    auto other = BucketInterface::MakeInstance(param2, common_param);
    test.TestSerializeAndDeserialize(common_param.dim_, other);
}

TEST_CASE("BucketDataCell Basic Test", "[ut][BucketDataCell] ") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = 128;
    std::string io_type = GENERATE("memory_io", "block_memory_io", "buffer_io", "async_io");
    std::vector<std::pair<std::string, float>> quantizer_errors = {
        {"sq8", 2e-2F},
        {"fp32", 1e-5F},
    };
    auto bucket_count = 20;
    MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    constexpr const char* param_temp =
        R"(
        {{
            "io_params": {{
                "type": "{}",
                "file_path": "{}"
            }},
            "quantization_params": {{
                "type": "{}"
            }},
            "buckets_count": {}
        }}
        )";
    fixtures::TempDir temp_dir("vsag_bucket_data_cell_test");
    auto quantizer_error = quantizer_errors[random() % quantizer_errors.size()];
    auto metric = metrics[random() % 3];
    std::string file_path1 = temp_dir.GenerateRandomFile(false);
    std::string file_path2 = temp_dir.GenerateRandomFile(false);

    auto param_str =
        fmt::format(param_temp, io_type, file_path1, quantizer_error.first, bucket_count);
    auto param_json = JsonType::Parse(param_str);
    auto param1 = std::make_shared<BucketDataCellParameter>();
    param1->FromJson(param_json);

    param_str = fmt::format(param_temp, io_type, file_path2, quantizer_error.first, bucket_count);
    param_json = JsonType::Parse(param_str);
    auto param2 = std::make_shared<BucketDataCellParameter>();
    param2->FromJson(param_json);

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = metric;

    TestBucketDataCell(param1, param2, common_param, quantizer_error.second);
}

TEST_CASE("BucketDataCell supports RabitQ", "[ut][BucketDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr uint64_t base_count = 24;
    constexpr BucketIdType bucket_count = 3;
    auto vectors = fixtures::generate_vectors(base_count, dim);
    auto queries = fixtures::generate_vectors(1, dim, 17);

    constexpr const char* param_str = R"(
        {
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "rabitq",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": 1
            },
            "buckets_count": 3
        }
        )";

    MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    for (auto metric : metrics) {
        auto param_json = JsonType::Parse(param_str);
        auto param = std::make_shared<BucketDataCellParameter>();
        param->FromJson(param_json);

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = metric;

        auto bucket = BucketInterface::MakeInstance(param, common_param);
        REQUIRE(bucket != nullptr);
        REQUIRE(bucket->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_RABITQ);

        bucket->Train(vectors.data(), base_count);
        for (uint64_t i = 0; i < base_count; ++i) {
            auto bucket_id = static_cast<BucketIdType>(i % bucket_count);
            bucket->InsertVector(vectors.data() + i * dim, bucket_id, static_cast<InnerIdType>(i));
        }

        auto computer = bucket->FactoryComputer(queries.data());
        for (BucketIdType bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
            auto bucket_size = bucket->GetBucketSize(bucket_id);
            std::vector<float> dists(bucket_size);
            bucket->ScanBucketById(dists.data(), computer, bucket_id);
            for (InnerIdType offset = 0; offset < bucket_size; ++offset) {
                REQUIRE(std::isfinite(dists[offset]));
                auto one_dist = bucket->QueryOneById(computer, bucket_id, offset);
                REQUIRE(std::isfinite(one_dist));
            }
        }
    }
}

TEST_CASE("BucketDataCell InsertVectorWithOffset", "[ut][BucketDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 16;
    constexpr uint64_t base_count = 32;
    auto vectors = fixtures::generate_vectors(base_count, dim);
    auto queries = fixtures::generate_vectors(1, dim, 23);

    constexpr const char* param_str = R"(
        {
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            },
            "buckets_count": 4
        }
        )";

    auto make_bucket = [&]() {
        auto param_json = JsonType::Parse(param_str);
        auto param = std::make_shared<BucketDataCellParameter>();
        param->FromJson(param_json);

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

        auto bucket = BucketInterface::MakeInstance(param, common_param);
        bucket->Train(vectors.data(), base_count);
        return bucket;
    };

    SECTION("fixed offset write is queryable") {
        auto appended = make_bucket();
        auto fixed = make_bucket();
        constexpr BucketIdType bucket_id = 2;
        constexpr InnerIdType inner_id = 7;
        constexpr InnerIdType offset_id = 5;

        auto append_offset =
            appended->InsertVector(vectors.data() + inner_id * dim, bucket_id, inner_id);
        for (InnerIdType offset = 0; offset <= offset_id; ++offset) {
            fixed->InsertVector(vectors.data() + offset * dim, bucket_id, offset);
        }
        fixed->InsertVectorWithOffset(
            vectors.data() + inner_id * dim, bucket_id, inner_id, offset_id);

        auto fixed_computer = fixed->FactoryComputer(queries.data());
        auto appended_computer = appended->FactoryComputer(queries.data());
        REQUIRE(fixed->GetBucketSize(bucket_id) == offset_id + 1);
        REQUIRE(fixed->GetInnerIds(bucket_id)[offset_id] == inner_id);
        REQUIRE(std::abs(fixed->QueryOneById(fixed_computer, bucket_id, offset_id) -
                         appended->QueryOneById(appended_computer, bucket_id, append_offset)) <
                1e-5F);
    }

    SECTION("append and fixed offset can be mixed") {
        auto bucket = make_bucket();
        constexpr BucketIdType bucket_id = 1;
        auto offset0 = bucket->InsertVector(vectors.data(), bucket_id, 0);
        auto offset1 = bucket->InsertVector(vectors.data() + dim, bucket_id, 1);
        auto offset2 = bucket->InsertVector(vectors.data() + 2 * dim, bucket_id, 2);
        REQUIRE(offset0 == 0);
        REQUIRE(offset1 == 1);
        REQUIRE(offset2 == 2);

        constexpr InnerIdType appended_inner_id = 3;
        bucket->InsertVectorWithOffset(
            vectors.data() + appended_inner_id * dim, bucket_id, appended_inner_id, offset2 + 1);
        constexpr InnerIdType fixed_inner_id = 9;
        bucket->InsertVectorWithOffset(
            vectors.data() + fixed_inner_id * dim, bucket_id, fixed_inner_id, offset1);

        REQUIRE(bucket->GetBucketSize(bucket_id) == offset2 + 2);
        REQUIRE(bucket->GetInnerIds(bucket_id)[offset1] == fixed_inner_id);
        REQUIRE(bucket->GetInnerIds(bucket_id)[offset2 + 1] == appended_inner_id);
    }

    SECTION("empty sentinel inner id is rejected") {
        auto bucket = make_bucket();
        constexpr BucketIdType bucket_id = 0;
        auto empty_inner_id = std::numeric_limits<InnerIdType>::max();

        REQUIRE_THROWS(bucket->InsertVector(vectors.data(), bucket_id, empty_inner_id));
        REQUIRE_THROWS(
            bucket->InsertVectorWithOffset(vectors.data(), bucket_id, empty_inner_id, 0));
    }

    SECTION("out of order fixed offset writes keep holes") {
        auto bucket = make_bucket();
        constexpr BucketIdType bucket_id = 1;

        bucket->InsertVectorWithOffset(vectors.data() + 2 * dim, bucket_id, 2, 2);
        REQUIRE(bucket->GetBucketSize(bucket_id) == 3);
        REQUIRE(bucket->GetInnerIds(bucket_id)[0] == std::numeric_limits<InnerIdType>::max());
        REQUIRE(bucket->GetInnerIds(bucket_id)[1] == std::numeric_limits<InnerIdType>::max());
        REQUIRE(bucket->GetInnerIds(bucket_id)[2] == 2);

        std::vector<uint8_t> hole_codes(sizeof(float) * dim, 1);
        bucket->GetCodesById(bucket_id, 0, hole_codes.data());
        REQUIRE(std::all_of(
            hole_codes.begin(), hole_codes.end(), [](uint8_t value) { return value == 0; }));
        std::fill(hole_codes.begin(), hole_codes.end(), 1);
        bucket->GetCodesById(bucket_id, 1, hole_codes.data());
        REQUIRE(std::all_of(
            hole_codes.begin(), hole_codes.end(), [](uint8_t value) { return value == 0; }));

        bucket->InsertVectorWithOffset(vectors.data(), bucket_id, 0, 0);
        REQUIRE(bucket->GetBucketSize(bucket_id) == 3);

        bucket->InsertVectorWithOffset(vectors.data() + dim, bucket_id, 1, 1);
        REQUIRE(bucket->GetBucketSize(bucket_id) == 3);
        for (InnerIdType offset = 0; offset < 3; ++offset) {
            REQUIRE(bucket->GetInnerIds(bucket_id)[offset] == offset);
        }
    }

    SECTION("dense fixed offset writes match append layout") {
        auto appended = make_bucket();
        auto fixed = make_bucket();
        constexpr BucketIdType bucket_id = 3;
        constexpr uint64_t insert_count = 8;

        for (uint64_t i = 0; i < insert_count; ++i) {
            auto offset = appended->InsertVector(
                vectors.data() + i * dim, bucket_id, static_cast<InnerIdType>(i));
            fixed->InsertVectorWithOffset(
                vectors.data() + i * dim, bucket_id, static_cast<InnerIdType>(i), offset);
        }

        REQUIRE(fixed->GetBucketSize(bucket_id) == appended->GetBucketSize(bucket_id));
        std::vector<uint8_t> appended_codes(sizeof(float) * dim);
        std::vector<uint8_t> fixed_codes(sizeof(float) * dim);
        for (uint64_t i = 0; i < insert_count; ++i) {
            REQUIRE(fixed->GetInnerIds(bucket_id)[i] == appended->GetInnerIds(bucket_id)[i]);
            appended->GetCodesById(bucket_id, i, appended_codes.data());
            fixed->GetCodesById(bucket_id, i, fixed_codes.data());
            REQUIRE(appended_codes == fixed_codes);
        }
    }

    SECTION("concurrent fixed offset writes do not conflict") {
        auto bucket = make_bucket();
        constexpr BucketIdType bucket_id = 0;
        constexpr uint64_t insert_count = 16;
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> exceptions(insert_count);
        threads.reserve(insert_count);

        for (uint64_t i = 0; i < insert_count; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    bucket->InsertVectorWithOffset(vectors.data() + i * dim,
                                                   bucket_id,
                                                   static_cast<InnerIdType>(i + insert_count),
                                                   static_cast<InnerIdType>(i));
                } catch (...) {
                    exceptions[i] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (auto& exception : exceptions) {
            if (exception != nullptr) {
                std::rethrow_exception(exception);
            }
        }

        REQUIRE(bucket->GetBucketSize(bucket_id) == insert_count);
        for (uint64_t i = 0; i < insert_count; ++i) {
            REQUIRE(bucket->GetInnerIds(bucket_id)[i] == i + insert_count);
        }
    }

    SECTION("out of order writes survive serialize deserialize") {
        auto bucket = make_bucket();
        constexpr BucketIdType bucket_id = 0;
        bucket->InsertVectorWithOffset(vectors.data() + 2 * dim, bucket_id, 2, 2);
        bucket->InsertVectorWithOffset(vectors.data() + dim, bucket_id, 1, 1);
        bucket->InsertVectorWithOffset(vectors.data(), bucket_id, 0, 0);
        REQUIRE(bucket->GetBucketSize(bucket_id) == 3);

        std::stringstream ss;
        IOStreamWriter writer(ss);
        bucket->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);

        auto restored = make_bucket();
        restored->Deserialize(reader);
        REQUIRE(restored->GetBucketSize(bucket_id) == 3);
        for (InnerIdType offset = 0; offset < 3; ++offset) {
            REQUIRE(restored->GetInnerIds(bucket_id)[offset] == offset);
        }
    }

    SECTION("merge other remains overwriteable by fixed offset") {
        auto dst = make_bucket();
        auto src = make_bucket();
        constexpr BucketIdType bucket_id = 0;
        dst->InsertVector(vectors.data(), bucket_id, 10);
        src->InsertVectorWithOffset(vectors.data() + dim, bucket_id, 20, 1);

        dst->MergeOther(src, 5);
        REQUIRE(dst->GetBucketSize(bucket_id) == 3);
        REQUIRE(dst->GetInnerIds(bucket_id)[0] == 10);
        REQUIRE(dst->GetInnerIds(bucket_id)[1] == std::numeric_limits<InnerIdType>::max());
        REQUIRE(dst->GetInnerIds(bucket_id)[2] == 25);

        dst->InsertVectorWithOffset(vectors.data() + 2 * dim, bucket_id, 30, 1);
        REQUIRE(dst->GetBucketSize(bucket_id) == 3);
        REQUIRE(dst->GetInnerIds(bucket_id)[0] == 10);
        REQUIRE(dst->GetInnerIds(bucket_id)[1] == 30);
        REQUIRE(dst->GetInnerIds(bucket_id)[2] == 25);
    }
}
