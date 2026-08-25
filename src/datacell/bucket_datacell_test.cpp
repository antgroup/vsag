
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
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index_common_param.h"
#include "io/reader_io/reader_io_parameter.h"
#include "quantization/fp32_quantizer.h"
#include "simd/simd.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

using namespace vsag;

namespace {

class FixedCentroidPartitionStrategy : public IVFPartitionStrategy {
public:
    FixedCentroidPartitionStrategy(const IndexCommonParam& common_param,
                                   BucketIdType bucket_count,
                                   float centroid_scale = 1.0F)
        : IVFPartitionStrategy(common_param, bucket_count), centroid_scale_(centroid_scale) {
    }

    void
    Train(const DatasetPtr) override {
        is_trained_ = true;
    }

    Vector<BucketIdType>
    ClassifyDatas(const void*, int64_t count, BucketIdType, QueryContext*) const override {
        return Vector<BucketIdType>(count, 0, allocator_);
    }

    void
    GetCentroid(BucketIdType bucket_id, Vector<float>& centroid) override {
        centroid_requests_.fetch_add(1, std::memory_order_relaxed);
        for (uint64_t i = 0; i < centroid.size(); ++i) {
            centroid[i] = static_cast<float>((bucket_id + 1) * (i + 1)) * 0.01F * centroid_scale_;
        }
    }

    [[nodiscard]] uint64_t
    GetCentroidRequestCount() const {
        return centroid_requests_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> centroid_requests_{0};
    float centroid_scale_{1.0F};
};

class TrackingReader : public Reader {
public:
    explicit TrackingReader(std::shared_ptr<std::string> data) : data_(std::move(data)) {
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        ++read_calls_;
        copy(offset, len, dest);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        copy(offset, len, dest);
        callback(IOErrorCode::IO_SUCCESS, "success");
    }

    bool
    MultiRead(uint8_t* dests,
              const uint64_t* lens,
              const uint64_t* offsets,
              uint64_t count) override {
        ++multi_read_calls_;
        multi_read_ranges_ += count;
        for (uint64_t i = 0; i < count; ++i) {
            copy(offsets[i], lens[i], dests);
            dests += lens[i];
        }
        return true;
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return data_->size();
    }

    void
    ResetStats() {
        read_calls_ = 0;
        multi_read_calls_ = 0;
        multi_read_ranges_ = 0;
    }

    uint64_t read_calls_{0};
    uint64_t multi_read_calls_{0};
    uint64_t multi_read_ranges_{0};

private:
    void
    copy(uint64_t offset, uint64_t len, void* dest) const {
        if (offset > data_->size() or len > data_->size() - offset) {
            throw VsagException(ErrorType::READ_ERROR, "tracking reader read out of bounds");
        }
        std::memcpy(dest, data_->data() + offset, len);
    }

    std::shared_ptr<std::string> data_;
};

struct TrackingWrite {
    uint64_t size{0};
    uint64_t offset{0};
};

struct TrackingWriteIOState {
    std::vector<std::vector<TrackingWrite>> writes_by_bucket;
};

class TrackingWriteIOParameter : public IOParameter {
public:
    explicit TrackingWriteIOParameter(std::shared_ptr<TrackingWriteIOState> state)
        : IOParameter("tracking_write_io"), state_(std::move(state)) {
    }

    void
    FromJson(const JsonType&) override {
    }

    JsonType
    ToJson() const override {
        return JsonType();
    }

    std::shared_ptr<TrackingWriteIOState> state_;
};

class TrackingWriteIO : public BasicIO<TrackingWriteIO> {
public:
    static constexpr bool InMemory = true;
    static constexpr bool SkipDeserialize = false;

    TrackingWriteIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : BasicIO<TrackingWriteIO>(common_param.allocator_.get()) {
        auto tracking_param = std::dynamic_pointer_cast<TrackingWriteIOParameter>(param);
        if (tracking_param == nullptr or tracking_param->state_ == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "TrackingWriteIO requires tracking state");
        }
        state_ = tracking_param->state_;
        bucket_id_ = state_->writes_by_bucket.size();
        state_->writes_by_bucket.emplace_back();
    }

    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        state_->writes_by_bucket[bucket_id_].emplace_back(TrackingWrite{size, offset});
        const auto next_size = offset + size;
        if (data_.size() < next_size) {
            data_.resize(next_size);
        }
        if (size > 0) {
            std::memcpy(data_.data() + offset, data, size);
        }
        this->size_ = std::max(this->size_, next_size);
    }

    void
    ResizeImpl(uint64_t size) {
        data_.resize(size);
        this->size_ = size;
    }

    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        if (offset > data_.size() or size > data_.size() - offset) {
            return false;
        }
        if (size > 0) {
            std::memcpy(data, data_.data() + offset, size);
        }
        return true;
    }

    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        need_release = false;
        if (offset > data_.size() or size > data_.size() - offset) {
            return nullptr;
        }
        return data_.data() + offset;
    }

    bool
    MultiReadImpl(uint8_t* data, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        for (uint64_t i = 0; i < count; ++i) {
            if (not ReadImpl(sizes[i], offsets[i], data)) {
                return false;
            }
            data += sizes[i];
        }
        return true;
    }

private:
    std::shared_ptr<TrackingWriteIOState> state_;
    uint64_t bucket_id_{0};
    std::vector<uint8_t> data_;
};

}  // namespace

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
    other->backend_ = DistanceEvaluationBackend::UNKNOWN;
    test_serializion(*this->bucket_, *other);
    REQUIRE(other->backend_ == SearchStatistics::BackendFromName(other->GetQuantizerName()));

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

TEST_CASE("BucketDataCell rejects invalid parameters", "[ut][BucketDataCell]") {
    IndexCommonParam common_param;

    REQUIRE(BucketInterface::MakeInstance(nullptr, common_param) == nullptr);
}

TEST_CASE("BucketDataCell rejects inconsistent serialized metadata", "[ut][BucketDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 4;
    constexpr const char* param_str = R"(
        {
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            },
            "buckets_count": 1
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
        return BucketInterface::MakeInstance(param, common_param);
    };

    auto bucket = make_bucket();
    auto vectors = fixtures::generate_vectors(1, dim);
    bucket->Train(vectors.data(), 1);
    bucket->InsertVector(vectors.data(), 0, 0);

    std::stringstream stream;
    IOStreamWriter writer(stream);
    bucket->Serialize(writer);
    const auto serialized = stream.str();

    SECTION("inner id vector is shorter than bucket size") {
        auto malformed = serialized;
        constexpr InnerIdType invalid_bucket_size = 2;
        std::memcpy(malformed.data() + malformed.size() - sizeof(invalid_bucket_size),
                    &invalid_bucket_size,
                    sizeof(invalid_bucket_size));

        std::stringstream malformed_stream(malformed);
        IOStreamReader reader(malformed_stream);
        auto restored = make_bucket();
        try {
            restored->Deserialize(reader);
            FAIL("inconsistent inner id metadata should be rejected");
        } catch (const VsagException& error) {
            REQUIRE(error.error_.type == ErrorType::INVALID_BINARY);
            REQUIRE(error.error_.message ==
                    "serialized bucket 0 inner id count is smaller than bucket size");
        }
    }

    SECTION("bucket size vector does not cover every bucket") {
        auto malformed = serialized;
        constexpr uint64_t invalid_bucket_size_count = 0;
        const uint64_t count_offset =
            static_cast<uint64_t>(malformed.size()) - sizeof(InnerIdType) - sizeof(uint64_t);
        std::memcpy(malformed.data() + count_offset,
                    &invalid_bucket_size_count,
                    sizeof(invalid_bucket_size_count));

        std::stringstream malformed_stream(malformed);
        IOStreamReader reader(malformed_stream);
        auto restored = make_bucket();
        try {
            restored->Deserialize(reader);
            FAIL("inconsistent bucket size metadata should be rejected");
        } catch (const VsagException& error) {
            REQUIRE(error.error_.type == ErrorType::INVALID_BINARY);
            REQUIRE(error.error_.message ==
                    "serialized bucket size vector does not match bucket count");
        }
    }
}

TEST_CASE("BucketDataCell ReaderIO queries serialized bucket codes",
          "[ut][BucketDataCell][ReaderIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 4;
    constexpr BucketIdType bucket_count = 3;
    constexpr uint64_t base_count = 6;
    auto vectors = fixtures::generate_vectors(base_count, dim);
    auto query = fixtures::generate_vectors(1, dim, 53);

    auto make_bucket = [&](const std::string& io_type) {
        auto param_json = JsonType::Parse(fmt::format(
            R"({{
                "io_params": {{
                    "type": "{}"
                }},
                "quantization_params": {{
                    "type": "fp32"
                }},
                "buckets_count": {}
            }})",
            io_type,
            bucket_count));
        auto param = std::make_shared<BucketDataCellParameter>();
        param->FromJson(param_json);

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
        return BucketInterface::MakeInstance(param, common_param);
    };

    auto source = make_bucket("memory_io");
    source->Train(vectors.data(), base_count);
    std::vector<BucketIdType> inserted_bucket_ids{0, 1, 2, 0, 1, 2};
    for (InnerIdType inner_id = 0; inner_id < base_count; ++inner_id) {
        source->InsertVector(vectors.data() + static_cast<uint64_t>(inner_id) * dim,
                             inserted_bucket_ids[inner_id],
                             inner_id);
    }

    std::stringstream stream;
    IOStreamWriter writer(stream);
    source->Serialize(writer);
    auto serialized = std::make_shared<std::string>(stream.str());

    uint64_t deserialized_read_bytes = 0;
    ReadFuncStreamReader stream_reader(
        [&](uint64_t offset, uint64_t size, void* dest) {
            if (offset > serialized->size() or size > serialized->size() - offset) {
                throw VsagException(ErrorType::READ_ERROR, "serialized bucket read out of bounds");
            }
            deserialized_read_bytes += size;
            std::memcpy(dest, serialized->data() + offset, size);
        },
        0,
        serialized->size());
    auto restored = make_bucket("reader_io");
    REQUIRE(restored != nullptr);
    restored->Deserialize(stream_reader);

    constexpr uint64_t serialized_code_bytes = base_count * dim * sizeof(float);
    REQUIRE(stream_reader.GetCursor() == serialized->size());
    REQUIRE(deserialized_read_bytes == serialized->size() - serialized_code_bytes);

    auto restored_computer = restored->FactoryComputer(query.data());
    REQUIRE_THROWS(restored->QueryOneById(restored_computer, 0, 0));

    auto tracking_reader = std::make_shared<TrackingReader>(serialized);
    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = tracking_reader;
    restored->InitIO(reader_param);

    auto source_computer = source->FactoryComputer(query.data());
    for (BucketIdType bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
        for (InnerIdType offset_id = 0; offset_id < 2; ++offset_id) {
            REQUIRE(restored->QueryOneById(restored_computer, bucket_id, offset_id) ==
                    source->QueryOneById(source_computer, bucket_id, offset_id));
        }
    }

    std::vector<BucketIdType> bucket_ids{2, 0, 1, 0, 2, 1, 0};
    std::vector<InnerIdType> offset_ids{1, 0, 1, 1, 0, 0, 0};
    std::vector<float> expected(bucket_ids.size());
    std::vector<float> actual(bucket_ids.size());
    for (uint64_t i = 0; i < bucket_ids.size(); ++i) {
        expected[i] = source->QueryOneById(source_computer, bucket_ids[i], offset_ids[i]);
    }

    tracking_reader->ResetStats();
    SearchStatistics stats;
    QueryContext ctx{nullptr, &stats};
    restored->Query(actual.data(),
                    restored_computer,
                    bucket_ids.data(),
                    offset_ids.data(),
                    static_cast<InnerIdType>(bucket_ids.size()),
                    &ctx);
    REQUIRE(actual == expected);
    REQUIRE(tracking_reader->read_calls_ == 0);
    REQUIRE(tracking_reader->multi_read_calls_ == bucket_count);
    REQUIRE(tracking_reader->multi_read_ranges_ == bucket_count);
    REQUIRE(stats.io_cnt.load(std::memory_order_relaxed) == bucket_count);
}

TEST_CASE("BucketDataCell ReaderIO shares one read cache across buckets",
          "[ut][BucketDataCell][ReaderIO][ReadCache]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr BucketIdType bucket_count = 2;
    constexpr uint64_t cache_page_count = 2;
    constexpr int64_t dim = Page::DEFAULT_PAGE_SIZE / sizeof(float);
    const auto cache_size = cache_page_count * Page::DEFAULT_PAGE_SIZE;

    auto make_bucket = [&](const std::string& io_type) {
        auto param_json = JsonType::Parse(fmt::format(
            R"({{
                "io_params": {{
                    "type": "{}",
                    "enable_read_cache": true,
                    "total_cache_size": {}
                }},
                "quantization_params": {{
                    "type": "fp32"
                }},
                "buckets_count": {}
            }})",
            io_type,
            cache_size,
            bucket_count));
        auto param = std::make_shared<BucketDataCellParameter>();
        param->FromJson(param_json);

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
        return BucketInterface::MakeInstance(param, common_param);
    };

    auto build_source = [&](float first_value) {
        auto source = make_bucket("memory_io");
        std::vector<float> first(dim, first_value);
        std::vector<float> second(dim, first_value + 1.0F);
        std::vector<float> third(dim, first_value + 2.0F);
        source->Train(first.data(), 1);
        source->InsertVector(first.data(), 0, 0);
        source->InsertVector(second.data(), 0, 1);
        source->InsertVector(third.data(), 1, 2);

        std::stringstream stream;
        IOStreamWriter writer(stream);
        source->Serialize(writer);
        return std::make_pair(source, std::make_shared<std::string>(stream.str()));
    };

    auto [first_source, first_serialized] = build_source(1.0F);
    auto [second_source, second_serialized] = build_source(11.0F);
    REQUIRE(second_serialized->size() == first_serialized->size());

    auto restored = make_bucket("reader_io");
    std::stringstream stream(*first_serialized);
    IOStreamReader stream_reader(stream);
    restored->Deserialize(stream_reader);

    auto make_reader_param = [&](const std::shared_ptr<TrackingReader>& reader) {
        auto reader_param = std::make_shared<ReaderIOParameter>();
        reader_param->reader = reader;
        reader_param->enable_read_cache_ = true;
        reader_param->read_cache_total_size_ = cache_size;
        return reader_param;
    };

    std::vector<float> query(dim, 0.0F);
    auto first_computer = first_source->FactoryComputer(query.data());
    auto restored_computer = restored->FactoryComputer(query.data());
    auto first_reader = std::make_shared<TrackingReader>(first_serialized);
    restored->InitIO(make_reader_param(first_reader));

    first_reader->ResetStats();
    REQUIRE(restored->QueryOneById(restored_computer, 0, 0) ==
            first_source->QueryOneById(first_computer, 0, 0));
    REQUIRE(restored->QueryOneById(restored_computer, 0, 1) ==
            first_source->QueryOneById(first_computer, 0, 1));
    REQUIRE(restored->QueryOneById(restored_computer, 0, 0) ==
            first_source->QueryOneById(first_computer, 0, 0));
    REQUIRE(first_reader->read_calls_ == cache_page_count);
    REQUIRE(first_reader->multi_read_calls_ == 0);

    const auto reads_before_other_bucket = first_reader->read_calls_;
    REQUIRE(restored->QueryOneById(restored_computer, 1, 0) ==
            first_source->QueryOneById(first_computer, 1, 0));
    REQUIRE(first_reader->read_calls_ == reads_before_other_bucket + 1);

    auto second_reader = std::make_shared<TrackingReader>(second_serialized);
    restored->InitIO(make_reader_param(second_reader));
    second_reader->ResetStats();
    auto second_computer = second_source->FactoryComputer(query.data());
    REQUIRE(restored->QueryOneById(restored_computer, 1, 0) ==
            second_source->QueryOneById(second_computer, 1, 0));
    REQUIRE(second_reader->read_calls_ == 1);

    std::vector<BucketIdType> bucket_ids{1, 0, 0, 1, 0};
    std::vector<InnerIdType> offset_ids{0, 1, 0, 0, 1};
    std::vector<float> expected(bucket_ids.size());
    std::vector<float> actual(bucket_ids.size());
    for (uint64_t i = 0; i < bucket_ids.size(); ++i) {
        expected[i] = second_source->QueryOneById(second_computer, bucket_ids[i], offset_ids[i]);
    }
    restored->Query(actual.data(),
                    restored_computer,
                    bucket_ids.data(),
                    offset_ids.data(),
                    static_cast<InnerIdType>(bucket_ids.size()));
    REQUIRE(actual == expected);

    auto no_cache_param = make_reader_param(second_reader);
    no_cache_param->enable_read_cache_ = false;
    no_cache_param->read_cache_total_size_ = 0;
    restored->InitIO(no_cache_param);
    second_reader->ResetStats();
    REQUIRE(restored->QueryOneById(restored_computer, 0, 0) ==
            second_source->QueryOneById(second_computer, 0, 0));
    REQUIRE(restored->QueryOneById(restored_computer, 0, 0) ==
            second_source->QueryOneById(second_computer, 0, 0));
    REQUIRE(second_reader->read_calls_ == 2);
}

TEST_CASE("BucketDataCell batch query", "[ut][BucketDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto query_allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 8;
    constexpr uint64_t bucket_count = 3;
    constexpr uint64_t vectors_per_bucket = 4;
    constexpr uint64_t base_count = bucket_count * vectors_per_bucket;
    const auto io_type = GENERATE(std::string("memory_io"), std::string("buffer_io"));
    auto vectors = fixtures::generate_vectors(base_count, dim);
    auto queries = fixtures::generate_vectors(1, dim, 41);
    fixtures::TempDir temp_dir("vsag_bucket_batch_query_test");

    auto make_bucket = [&]() {
        auto file_path = temp_dir.GenerateRandomFile(false);
        auto param_json = JsonType::Parse(fmt::format(
            R"({{
                "io_params": {{
                    "type": "{}",
                    "file_path": "{}"
                }},
                "quantization_params": {{
                    "type": "fp32"
                }},
                "buckets_count": {}
            }})",
            io_type,
            file_path,
            bucket_count));
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

    auto bucket = make_bucket();
    for (uint64_t bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
        for (uint64_t offset_id = 0; offset_id < vectors_per_bucket; ++offset_id) {
            auto inner_id = bucket_id * vectors_per_bucket + offset_id;
            bucket->InsertVector(vectors.data() + inner_id * dim,
                                 static_cast<BucketIdType>(bucket_id),
                                 static_cast<InnerIdType>(inner_id));
        }
    }

    std::vector<BucketIdType> bucket_ids{2, 0, 1, 0, 2, 1, 0};
    std::vector<InnerIdType> offset_ids{3, 1, 2, 2, 3, 0, 1};
    std::vector<float> expected(bucket_ids.size());
    std::vector<float> actual(bucket_ids.size());
    auto computer = bucket->FactoryComputer(queries.data());
    for (uint64_t i = 0; i < bucket_ids.size(); ++i) {
        expected[i] = bucket->QueryOneById(computer, bucket_ids[i], offset_ids[i]);
    }

    SearchStatistics stats;
    QueryContext ctx{query_allocator.get(), &stats};
    bucket->Query(actual.data(),
                  computer,
                  bucket_ids.data(),
                  offset_ids.data(),
                  static_cast<InnerIdType>(bucket_ids.size()),
                  &ctx);
    REQUIRE(actual == expected);
    if (io_type == "buffer_io") {
        // Four contiguous read ranges remain after sorting and de-duplicating the locations.
        REQUIRE(stats.io_cnt.load(std::memory_order_relaxed) == 4);
    }

    REQUIRE_NOTHROW(bucket->Query(nullptr, ComputerInterfacePtr{}, nullptr, nullptr, 0, &ctx));

    SECTION("invalid bucket is rejected") {
        BucketIdType invalid_bucket_id = -1;
        InnerIdType offset_id = 0;
        float dist = 0.0F;
        REQUIRE_THROWS(bucket->Query(&dist, computer, &invalid_bucket_id, &offset_id, 1));
    }

    SECTION("invalid offset is rejected") {
        BucketIdType bucket_id = 0;
        InnerIdType invalid_offset_id = 100;
        float dist = 0.0F;
        REQUIRE_THROWS(bucket->Query(&dist, computer, &bucket_id, &invalid_offset_id, 1));
    }

    SECTION("hole is rejected") {
        auto sparse_bucket = make_bucket();
        sparse_bucket->InsertVectorWithOffset(vectors.data() + 2 * dim, 0, 2, 2);
        auto sparse_computer = sparse_bucket->FactoryComputer(queries.data());
        BucketIdType bucket_id = 0;
        InnerIdType hole_offset_id = 0;
        float dist = 0.0F;
        REQUIRE_THROWS(
            sparse_bucket->Query(&dist, sparse_computer, &bucket_id, &hole_offset_id, 1));
    }
}

TEST_CASE("BucketDataCell batch query preserves residual correction", "[ut][BucketDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 8;
    constexpr uint64_t bucket_count = 2;
    constexpr uint64_t base_count = 8;
    auto vectors = fixtures::generate_vectors(base_count, dim);
    auto queries = fixtures::generate_vectors(1, dim, 43);
    MetricType metrics[] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE};

    for (auto metric : metrics) {
        auto param_json = JsonType::Parse(R"({
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            },
            "buckets_count": 2,
            "use_residual": true
        })");
        auto param = std::make_shared<BucketDataCellParameter>();
        param->FromJson(param_json);

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = metric;
        auto bucket = BucketInterface::MakeInstance(param, common_param);
        auto strategy =
            std::make_shared<FixedCentroidPartitionStrategy>(common_param, bucket_count);
        bucket->SetStrategy(strategy);
        bucket->Train(vectors.data(), base_count);
        for (uint64_t i = 0; i < base_count; ++i) {
            bucket->InsertVector(vectors.data() + i * dim,
                                 static_cast<BucketIdType>(i % bucket_count),
                                 static_cast<InnerIdType>(i));
        }

        std::vector<BucketIdType> bucket_ids{1, 0, 1, 0, 1};
        std::vector<InnerIdType> offset_ids{2, 3, 0, 1, 2};
        std::vector<float> expected(bucket_ids.size());
        std::vector<float> actual(bucket_ids.size());
        auto computer = bucket->FactoryComputer(queries.data());
        for (uint64_t i = 0; i < bucket_ids.size(); ++i) {
            expected[i] = bucket->QueryOneById(computer, bucket_ids[i], offset_ids[i]);
        }
        bucket->Query(actual.data(),
                      computer,
                      bucket_ids.data(),
                      offset_ids.data(),
                      static_cast<InnerIdType>(bucket_ids.size()));
        for (uint64_t i = 0; i < actual.size(); ++i) {
            REQUIRE(std::abs(actual[i] - expected[i]) < 1e-5F);
        }
    }
}

TEST_CASE("BucketDataCell batch query handles all bucket quantizers", "[ut][BucketDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 64;
    constexpr uint64_t train_count = 300;
    constexpr uint64_t bucket_count = 2;
    constexpr uint64_t vectors_per_bucket = 64;
    auto vectors = fixtures::generate_vectors(train_count, dim);
    auto queries = fixtures::generate_vectors(1, dim, 47);
    fixtures::TempDir temp_dir("vsag_bucket_batch_quantizers_test");

    const std::vector<std::pair<std::string, std::string>> quantizers = {
        {"fp32", R"({"type": "fp32"})"},
        {"sq8", R"({"type": "sq8"})"},
        {"sq4", R"({"type": "sq4"})"},
        {"sq4_uniform", R"({"type": "sq4_uniform"})"},
        {"sq8_uniform", R"({"type": "sq8_uniform"})"},
        {"bf16", R"({"type": "bf16"})"},
        {"fp16", R"({"type": "fp16"})"},
        {"pq", R"({"type": "pq", "pq_dim": 8, "pq_bits": 8})"},
        {"pqfs", R"({"type": "pqfs", "pq_dim": 8})"},
        {"rabitq",
         R"({"type": "rabitq", "rabitq_bits_per_dim_query": 32, "rabitq_bits_per_dim_base": 1})"},
    };

    for (const auto& io_type : {std::string("memory_io"), std::string("buffer_io")}) {
        for (const auto& [quantizer_name, quantizer_json] : quantizers) {
            CAPTURE(io_type, quantizer_name);
            JsonType param_json;
            JsonType io_json;
            io_json["type"].SetString(io_type);
            io_json["file_path"].SetString(temp_dir.GenerateRandomFile(false));
            param_json["io_params"].SetJson(io_json);
            param_json["quantization_params"].SetJson(JsonType::Parse(quantizer_json));
            param_json["buckets_count"].SetInt(bucket_count);
            auto param = std::make_shared<BucketDataCellParameter>();
            param->FromJson(param_json);

            IndexCommonParam common_param;
            common_param.allocator_ = allocator;
            common_param.dim_ = dim;
            common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
            auto bucket = BucketInterface::MakeInstance(param, common_param);
            bucket->Train(vectors.data(), train_count);
            for (uint64_t bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
                for (uint64_t offset_id = 0; offset_id < vectors_per_bucket; ++offset_id) {
                    auto inner_id = bucket_id * vectors_per_bucket + offset_id;
                    bucket->InsertVector(vectors.data() + inner_id * dim,
                                         static_cast<BucketIdType>(bucket_id),
                                         static_cast<InnerIdType>(inner_id));
                }
            }
            // PQFS queries require its normal IVF package state; Package is a no-op for others.
            bucket->Package();

            auto computer = bucket->FactoryComputer(queries.data());
            std::vector<BucketIdType> bucket_ids{1, 0, 1, 0, 1, 0};
            std::vector<InnerIdType> offset_ids{33, 1, 2, 34, 33, 2};
            std::vector<float> actual(bucket_ids.size());
            if (quantizer_name == "pqfs") {
                try {
                    bucket->Query(actual.data(),
                                  computer,
                                  bucket_ids.data(),
                                  offset_ids.data(),
                                  static_cast<InnerIdType>(bucket_ids.size()));
                    FAIL("PQFS batch point query should preserve QueryOneById rejection");
                } catch (const VsagException& error) {
                    REQUIRE(error.error_.type == ErrorType::INTERNAL_ERROR);
                    REQUIRE(error.error_.message ==
                            "PQFastScan doesn't support ComputeDist, only support "
                            "ComputeBatchDist");
                }
                continue;
            }

            std::vector<float> expected(bucket_ids.size());
            for (uint64_t i = 0; i < expected.size(); ++i) {
                expected[i] = bucket->QueryOneById(computer, bucket_ids[i], offset_ids[i]);
            }
            SearchStatistics stats;
            QueryContext ctx{nullptr, &stats};
            bucket->Query(actual.data(),
                          computer,
                          bucket_ids.data(),
                          offset_ids.data(),
                          static_cast<InnerIdType>(bucket_ids.size()),
                          &ctx);
            for (uint64_t i = 0; i < actual.size(); ++i) {
                REQUIRE(std::abs(actual[i] - expected[i]) < 1e-5F);
            }
            if (io_type == "buffer_io") {
                REQUIRE(stats.io_cnt.load(std::memory_order_relaxed) == 4);
            }
        }
    }
}

TEST_CASE("BucketDataCell batch insert groups one write per bucket", "[ut][BucketDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 8;
    constexpr BucketIdType bucket_count = 3;
    constexpr uint64_t base_count = 10;
    constexpr uint64_t code_size = dim * sizeof(float);
    auto vectors = fixtures::generate_vectors(base_count, dim);
    auto state = std::make_shared<TrackingWriteIOState>();
    auto io_param = std::make_shared<TrackingWriteIOParameter>(state);
    auto quantizer_param = std::make_shared<FP32QuantizerParameter>();

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto bucket = std::make_shared<
        BucketDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, TrackingWriteIO>>(
        quantizer_param, io_param, common_param, bucket_count);
    bucket->Train(vectors.data(), base_count);

    std::vector<BucketIdType> first_bucket_ids{2, 0, 2, 1, 0, 2};
    std::vector<InnerIdType> first_inner_ids{100, 101, 102, 103, 104, 105};
    std::vector<InnerIdType> first_offsets(first_bucket_ids.size(),
                                           std::numeric_limits<InnerIdType>::max());
    bucket->BatchInsertVector(vectors.data(),
                              first_bucket_ids.data(),
                              first_inner_ids.data(),
                              first_bucket_ids.size(),
                              first_offsets.data());

    REQUIRE(first_offsets == std::vector<InnerIdType>{0, 0, 1, 0, 1, 2});
    REQUIRE(state->writes_by_bucket.size() == bucket_count);
    REQUIRE(state->writes_by_bucket[0].size() == 1);
    REQUIRE(state->writes_by_bucket[0][0].size == 2 * code_size);
    REQUIRE(state->writes_by_bucket[0][0].offset == 0);
    REQUIRE(state->writes_by_bucket[1].size() == 1);
    REQUIRE(state->writes_by_bucket[1][0].size == code_size);
    REQUIRE(state->writes_by_bucket[1][0].offset == 0);
    REQUIRE(state->writes_by_bucket[2].size() == 1);
    REQUIRE(state->writes_by_bucket[2][0].size == 3 * code_size);
    REQUIRE(state->writes_by_bucket[2][0].offset == 0);

    std::vector<BucketIdType> second_bucket_ids{1, 2, 1, 0};
    std::vector<InnerIdType> second_inner_ids{200, 201, 202, 203};
    std::vector<InnerIdType> second_offsets(second_bucket_ids.size(),
                                            std::numeric_limits<InnerIdType>::max());
    bucket->BatchInsertVector(vectors.data() + first_bucket_ids.size() * dim,
                              second_bucket_ids.data(),
                              second_inner_ids.data(),
                              second_bucket_ids.size(),
                              second_offsets.data());

    REQUIRE(second_offsets == std::vector<InnerIdType>{1, 3, 2, 2});
    for (BucketIdType bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
        REQUIRE(state->writes_by_bucket[bucket_id].size() == 2);
    }
    REQUIRE(state->writes_by_bucket[0][1].size == code_size);
    REQUIRE(state->writes_by_bucket[0][1].offset == 2 * code_size);
    REQUIRE(state->writes_by_bucket[1][1].size == 2 * code_size);
    REQUIRE(state->writes_by_bucket[1][1].offset == code_size);
    REQUIRE(state->writes_by_bucket[2][1].size == code_size);
    REQUIRE(state->writes_by_bucket[2][1].offset == 3 * code_size);

    const std::vector<std::vector<InnerIdType>> expected_inner_ids{
        {101, 104, 203}, {103, 200, 202}, {100, 102, 105, 201}};
    const std::vector<std::vector<uint64_t>> expected_vector_ids{
        {1, 4, 9}, {3, 6, 8}, {0, 2, 5, 7}};
    std::vector<uint8_t> actual_codes(code_size);
    for (BucketIdType bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
        REQUIRE(bucket->GetBucketSize(bucket_id) == expected_inner_ids[bucket_id].size());
        for (uint64_t offset = 0; offset < expected_inner_ids[bucket_id].size(); ++offset) {
            REQUIRE(bucket->GetInnerIds(bucket_id)[offset] ==
                    expected_inner_ids[bucket_id][offset]);
            bucket->GetCodesById(bucket_id, offset, actual_codes.data());
            const auto* expected_codes = reinterpret_cast<const uint8_t*>(
                vectors.data() + expected_vector_ids[bucket_id][offset] * dim);
            REQUIRE(std::memcmp(actual_codes.data(), expected_codes, code_size) == 0);
        }
    }
}

TEST_CASE("BucketDataCell batch insert validates before writing", "[ut][BucketDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 4;
    constexpr BucketIdType bucket_count = 3;
    constexpr uint64_t count = 3;
    auto vectors = fixtures::generate_vectors(count, dim);
    auto state = std::make_shared<TrackingWriteIOState>();
    auto io_param = std::make_shared<TrackingWriteIOParameter>(state);
    auto quantizer_param = std::make_shared<FP32QuantizerParameter>();

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto bucket = std::make_shared<
        BucketDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, TrackingWriteIO>>(
        quantizer_param, io_param, common_param, bucket_count);
    bucket->Train(vectors.data(), count);

    std::vector<BucketIdType> bucket_ids{0, 1, 2};
    std::vector<InnerIdType> inner_ids{10, 11, 12};
    std::vector<InnerIdType> offsets(count, std::numeric_limits<InnerIdType>::max());
    auto require_unchanged = [&]() {
        REQUIRE(state->writes_by_bucket.size() == bucket_count);
        for (BucketIdType bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
            REQUIRE(state->writes_by_bucket[bucket_id].empty());
            REQUIRE(bucket->GetBucketSize(bucket_id) == 0);
        }
    };

    REQUIRE_NOTHROW(bucket->BatchInsertVector(nullptr, nullptr, nullptr, 0, nullptr));
    require_unchanged();

    REQUIRE_THROWS(bucket->BatchInsertVector(
        nullptr, bucket_ids.data(), inner_ids.data(), count, offsets.data()));
    require_unchanged();
    REQUIRE_THROWS(bucket->BatchInsertVector(
        vectors.data(), nullptr, inner_ids.data(), count, offsets.data()));
    require_unchanged();
    REQUIRE_THROWS(bucket->BatchInsertVector(
        vectors.data(), bucket_ids.data(), nullptr, count, offsets.data()));
    require_unchanged();
    REQUIRE_THROWS(bucket->BatchInsertVector(
        vectors.data(), bucket_ids.data(), inner_ids.data(), count, nullptr));
    require_unchanged();

    auto invalid_bucket_ids = bucket_ids;
    invalid_bucket_ids[1] = bucket_count;
    REQUIRE_THROWS(bucket->BatchInsertVector(
        vectors.data(), invalid_bucket_ids.data(), inner_ids.data(), count, offsets.data()));
    require_unchanged();

    auto invalid_inner_ids = inner_ids;
    invalid_inner_ids[1] = std::numeric_limits<InnerIdType>::max();
    REQUIRE_THROWS(bucket->BatchInsertVector(
        vectors.data(), bucket_ids.data(), invalid_inner_ids.data(), count, offsets.data()));
    require_unchanged();
}

TEST_CASE("BucketDataCell batch insert preserves RaBitQ PCA input stride", "[ut][BucketDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 32;
    constexpr uint64_t pca_dim = 16;
    constexpr uint64_t train_count = 64;
    constexpr uint64_t insert_count = 8;
    constexpr BucketIdType bucket_count = 2;
    auto vectors = fixtures::generate_vectors(train_count, dim);
    auto queries = fixtures::generate_vectors(3, dim, 61);

    auto make_bucket = [&]() {
        auto param_json = JsonType::Parse(fmt::format(
            R"({{
                "io_params": {{
                    "type": "memory_io"
                }},
                "quantization_params": {{
                    "type": "rabitq",
                    "pca_dim": {},
                    "rabitq_bits_per_dim_query": 32,
                    "rabitq_bits_per_dim_base": 1,
                    "fast_encode_rabitq": false
                }},
                "buckets_count": {}
            }})",
            pca_dim,
            bucket_count));
        auto param = std::make_shared<BucketDataCellParameter>();
        param->FromJson(param_json);

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
        return BucketInterface::MakeInstance(param, common_param);
    };

    auto incremental = make_bucket();
    auto batched = make_bucket();
    incremental->Train(vectors.data(), train_count);
    incremental->ExportModel(batched);

    std::vector<BucketIdType> bucket_ids{1, 0, 1, 1, 0, 0, 1, 0};
    std::vector<InnerIdType> inner_ids{70, 71, 72, 73, 74, 75, 76, 77};
    std::vector<InnerIdType> expected_offsets(insert_count);
    for (uint64_t i = 0; i < insert_count; ++i) {
        expected_offsets[i] =
            incremental->InsertVector(vectors.data() + i * dim, bucket_ids[i], inner_ids[i]);
    }

    std::vector<InnerIdType> actual_offsets(insert_count, std::numeric_limits<InnerIdType>::max());
    batched->BatchInsertVector(
        vectors.data(), bucket_ids.data(), inner_ids.data(), insert_count, actual_offsets.data());
    REQUIRE(actual_offsets == expected_offsets);

    for (BucketIdType bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
        REQUIRE(batched->GetBucketSize(bucket_id) == incremental->GetBucketSize(bucket_id));
        for (InnerIdType offset = 0; offset < incremental->GetBucketSize(bucket_id); ++offset) {
            REQUIRE(batched->GetInnerIds(bucket_id)[offset] ==
                    incremental->GetInnerIds(bucket_id)[offset]);
        }
    }

    for (uint64_t query_id = 0; query_id < 3; ++query_id) {
        auto incremental_computer = incremental->FactoryComputer(queries.data() + query_id * dim);
        auto batched_computer = batched->FactoryComputer(queries.data() + query_id * dim);
        for (uint64_t i = 0; i < insert_count; ++i) {
            const auto expected =
                incremental->QueryOneById(incremental_computer, bucket_ids[i], expected_offsets[i]);
            const auto actual =
                batched->QueryOneById(batched_computer, bucket_ids[i], actual_offsets[i]);
            REQUIRE(std::abs(actual - expected) < 1e-5F);
        }
    }
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

TEST_CASE("RaBitQ split bucket supports optimized build",
          "[ut][BucketDataCell][rabitq_split][optimized_build]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    const InnerIdType count = GENERATE(1, 31, 33);
    constexpr uint64_t training_count = 64;
    const uint32_t filter_bits = GENERATE(1U, 2U, 3U);
    constexpr BucketIdType bucket_count = 1;
    INFO("count=" << count << ", filter_bits=" << filter_bits);
    auto vectors = fixtures::generate_vectors(training_count, dim);
    auto query = fixtures::generate_vectors(1, dim, false, 113);
    constexpr const char* param_str = R"({
        "io_params": { "type": "memory_io" },
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 2,
            "fast_encode_rabitq": true
        },
        "buckets_count": 1,
        "use_residual": true
    })";

    auto make_bucket = [&]() {
        auto json = JsonType::Parse(param_str);
        json["quantization_params"]["rabitq_bits_per_dim_filter"].SetInt(filter_bits);
        auto param = std::make_shared<BucketDataCellParameter>();
        param->FromJson(json);
        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
        auto bucket = BucketInterface::MakeInstance(param, common_param);
        bucket->SetStrategy(
            std::make_shared<FixedCentroidPartitionStrategy>(common_param, bucket_count));
        return bucket;
    };

    auto normal = make_bucket();
    auto optimized = make_bucket();
    normal->Train(vectors.data(), training_count);
    normal->ExportModel(optimized);
    for (InnerIdType id = 0; id < count; ++id) {
        const auto bucket_id = static_cast<BucketIdType>(id % bucket_count);
        const auto offset_id = static_cast<InnerIdType>(id / bucket_count);
        normal->InsertVectorWithOffset(
            vectors.data() + static_cast<uint64_t>(id) * dim, bucket_id, id, offset_id);
    }

    auto thread_pool = SafeThreadPool::FactoryDefaultThreadPool();
    thread_pool->SetPoolSize(4);
    FlattenOptimizedBuildContext context{thread_pool, 4};
    REQUIRE(optimized->BeginOptimizedBuild(context, count));
    REQUIRE(optimized->IsOptimizedBuildActive());
    REQUIRE_FALSE(optimized->BeginOptimizedBuild(context, count));
    std::vector<std::future<void>> futures;
    futures.reserve(count);
    for (InnerIdType id = 0; id < count; ++id) {
        futures.emplace_back(thread_pool->GeneralEnqueue([&, id]() {
            const auto bucket_id = static_cast<BucketIdType>(id % bucket_count);
            const auto offset_id = static_cast<InnerIdType>(id / bucket_count);
            optimized->InsertVectorWithOffset(
                vectors.data() + static_cast<uint64_t>(id) * dim, bucket_id, id, offset_id);
        }));
    }
    for (auto& future : futures) {
        future.get();
    }
    optimized->FinalizeOptimizedBuild();
    REQUIRE_FALSE(optimized->IsOptimizedBuildActive());
    normal->Package();
    optimized->Package();

    std::vector<uint8_t> normal_code(normal->code_size_);
    std::vector<uint8_t> optimized_code(optimized->code_size_);
    for (InnerIdType id = 0; id < count; ++id) {
        const auto bucket_id = static_cast<BucketIdType>(id % bucket_count);
        const auto offset_id = static_cast<InnerIdType>(id / bucket_count);
        REQUIRE(optimized->GetInnerIds(bucket_id)[offset_id] == id);
        normal->GetCodesById(bucket_id, offset_id, normal_code.data());
        optimized->GetCodesById(bucket_id, offset_id, optimized_code.data());
        REQUIRE(normal_code == optimized_code);
    }

    const BucketIdType routed_bucket = 0;
    auto normal_computer = normal->FactoryComputerForBuckets(query.data(), &routed_bucket, 1);
    auto optimized_computer = optimized->FactoryComputerForBuckets(query.data(), &routed_bucket, 1);
    std::vector<float> normal_scan_dists(count);
    std::vector<float> optimized_scan_dists(count);
    std::vector<float> normal_filter_inner_products(count);
    std::vector<float> optimized_filter_inner_products(count);
    std::vector<InnerIdType> normal_ids(count);
    std::vector<InnerIdType> optimized_ids(count);
    normal->ScanBucketWithFilterInnerProduct(normal_scan_dists.data(),
                                             normal_filter_inner_products.data(),
                                             normal_computer,
                                             routed_bucket,
                                             nullptr,
                                             normal_ids.data());
    optimized->ScanBucketWithFilterInnerProduct(optimized_scan_dists.data(),
                                                optimized_filter_inner_products.data(),
                                                optimized_computer,
                                                routed_bucket,
                                                nullptr,
                                                optimized_ids.data());
    REQUIRE(normal_ids == optimized_ids);
    std::vector<float> normal_reorder_dists(count);
    std::vector<float> optimized_reorder_dists(count);
    SearchStatistics normal_stats;
    SearchStatistics optimized_stats;
    QueryContext normal_ctx{nullptr, &normal_stats};
    QueryContext optimized_ctx{nullptr, &optimized_stats};
    normal->QueryWithCandidateFilterInnerProductByInnerId(normal_reorder_dists.data(),
                                                          nullptr,
                                                          normal_filter_inner_products.data(),
                                                          normal_computer,
                                                          normal_ids.data(),
                                                          count,
                                                          &normal_ctx);
    optimized->QueryWithCandidateFilterInnerProductByInnerId(optimized_reorder_dists.data(),
                                                             nullptr,
                                                             optimized_filter_inner_products.data(),
                                                             optimized_computer,
                                                             optimized_ids.data(),
                                                             count,
                                                             &optimized_ctx);
    REQUIRE(normal_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) == count);
    REQUIRE(optimized_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) ==
            count);
    REQUIRE(normal_stats.rabitq_reorder_fallback_full_count.load(std::memory_order_relaxed) == 0);
    REQUIRE(optimized_stats.rabitq_reorder_fallback_full_count.load(std::memory_order_relaxed) ==
            0);

    for (InnerIdType id = 0; id < count; ++id) {
        REQUIRE(std::isfinite(normal_reorder_dists[id]));
        REQUIRE(std::abs(normal_reorder_dists[id] - optimized_reorder_dists[id]) < 1e-4F);
    }

    REQUIRE(optimized->BeginOptimizedBuild(context, count));
    optimized->AbortOptimizedBuild();
    REQUIRE_FALSE(optimized->IsOptimizedBuildActive());
}

TEST_CASE("RaBitQ split bucket scan spills computed masks above 256 blocks",
          "[ut][BucketDataCell][rabitq_split][residual]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 16;
    constexpr InnerIdType training_count = 64;
    constexpr InnerIdType count = 8193;
    constexpr BucketIdType bucket_id = 0;
    auto vectors = fixtures::generate_vectors(training_count, dim);
    auto query = fixtures::generate_vectors(1, dim, false, 127);
    constexpr const char* param_str = R"({
        "io_params": { "type": "memory_io" },
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 1,
            "fast_encode_rabitq": true
        },
        "buckets_count": 1,
        "use_residual": true
    })";

    auto param = std::make_shared<BucketDataCellParameter>();
    param->FromJson(JsonType::Parse(param_str));
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto bucket = BucketInterface::MakeInstance(param, common_param);
    bucket->SetStrategy(std::make_shared<FixedCentroidPartitionStrategy>(common_param, 1));
    bucket->Train(vectors.data(), training_count);
    for (InnerIdType id = 0; id < count; ++id) {
        const uint64_t training_id = static_cast<uint64_t>(id) % training_count;
        bucket->InsertVector(vectors.data() + training_id * dim, bucket_id, id);
    }
    bucket->Package();

    auto computer = bucket->FactoryComputerForBuckets(query.data(), &bucket_id, 1);
    std::vector<float> scan_dists(count);
    std::vector<float> filter_inner_products(count);
    std::vector<InnerIdType> scanned_ids(count);
    InnerIdType scanned_size = 0;
    bucket->ScanBucketWithFilterInnerProduct(scan_dists.data(),
                                             filter_inner_products.data(),
                                             computer,
                                             bucket_id,
                                             nullptr,
                                             scanned_ids.data(),
                                             count,
                                             &scanned_size);
    REQUIRE(scanned_size == count);
    REQUIRE(std::all_of(
        scan_dists.begin(), scan_dists.end(), [](float value) { return std::isfinite(value); }));
    REQUIRE(std::all_of(filter_inner_products.begin(),
                        filter_inner_products.end(),
                        [](float value) { return std::isfinite(value); }));

    constexpr std::array<InnerIdType, 5> selected_offsets{0, 31, 32, 8191, 8192};
    std::array<InnerIdType, selected_offsets.size()> selected_ids{};
    std::array<float, selected_offsets.size()> selected_inner_products{};
    for (uint64_t i = 0; i < selected_offsets.size(); ++i) {
        selected_ids[i] = scanned_ids[selected_offsets[i]];
        selected_inner_products[i] = filter_inner_products[selected_offsets[i]];
    }
    std::array<float, selected_offsets.size()> reorder_dists{};
    bucket->QueryWithCandidateFilterInnerProductByInnerId(reorder_dists.data(),
                                                          nullptr,
                                                          selected_inner_products.data(),
                                                          computer,
                                                          selected_ids.data(),
                                                          selected_ids.size());
    for (uint64_t i = 0; i < selected_offsets.size(); ++i) {
        REQUIRE(selected_ids[i] == selected_offsets[i]);
        const float expected = bucket->QueryOneById(computer, bucket_id, selected_offsets[i]);
        REQUIRE(std::abs(reorder_dists[i] - expected) < 1e-4F);
    }
}

TEST_CASE("RaBitQ split bucket prepares only routed residual computers",
          "[ut][BucketDataCell][rabitq_split][residual][routed_query]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 96;
    constexpr BucketIdType bucket_count = 4;
    const uint32_t filter_bits = GENERATE(1U, 2U, 3U);
    INFO("filter_bits=" << filter_bits);
    auto vectors = fixtures::generate_vectors(count, dim);
    auto query = fixtures::generate_vectors(1, dim, false, 71);
    constexpr const char* param_str = R"({
        "io_params": { "type": "memory_io" },
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 1,
            "fast_encode_rabitq": true
        },
        "buckets_count": 4,
        "use_residual": true
    })";

    auto param = std::make_shared<BucketDataCellParameter>();
    auto json = JsonType::Parse(param_str);
    json["quantization_params"]["rabitq_bits_per_dim_filter"].SetInt(filter_bits);
    param->FromJson(json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto bucket = BucketInterface::MakeInstance(param, common_param);
    auto strategy = std::make_shared<FixedCentroidPartitionStrategy>(common_param, bucket_count);
    bucket->SetStrategy(strategy);
    bucket->Train(vectors.data(), count);
    for (InnerIdType id = 0; id < count; ++id) {
        const auto bucket_id = static_cast<BucketIdType>(id % bucket_count);
        bucket->InsertVector(vectors.data() + static_cast<uint64_t>(id) * dim, bucket_id, id);
    }
    bucket->Package();

    const uint64_t centroid_requests_before_finalize = strategy->GetCentroidRequestCount();
    bucket->FinalizeLoad();
    REQUIRE(strategy->GetCentroidRequestCount() ==
            centroid_requests_before_finalize + bucket_count);

    std::array<BucketIdType, 3> routed_buckets{3, 1, 3};
    auto routed_computer = bucket->FactoryComputerForBuckets(
        query.data(), routed_buckets.data(), routed_buckets.size());
    auto candidate_computer = bucket->FactoryComputerForBuckets(
        query.data(), routed_buckets.data(), routed_buckets.size());

    std::array<InnerIdType, 2> published_candidate_ids{1, 3};
    std::array<float, 2> published_candidate_inner_products{};
    std::array<uint64_t, 2> published_candidate_versions{};
    for (uint64_t i = 0; i < published_candidate_ids.size(); ++i) {
        const auto bucket_id = static_cast<BucketIdType>(published_candidate_ids[i] % bucket_count);
        const auto bucket_size = bucket->GetBucketSize(bucket_id);
        std::vector<float> candidate_scan_dists(bucket_size);
        std::vector<float> candidate_scan_inner_products(bucket_size);
        std::vector<InnerIdType> candidate_scan_ids(bucket_size);
        InnerIdType candidate_scan_size = 0;
        published_candidate_versions[i] =
            bucket->ScanBucketWithFilterInnerProduct(candidate_scan_dists.data(),
                                                     candidate_scan_inner_products.data(),
                                                     candidate_computer,
                                                     bucket_id,
                                                     nullptr,
                                                     candidate_scan_ids.data(),
                                                     bucket_size,
                                                     &candidate_scan_size);
        REQUIRE(candidate_scan_size == bucket_size);
        REQUIRE(candidate_scan_ids[0] == published_candidate_ids[i]);
        REQUIRE(std::all_of(candidate_scan_dists.begin(),
                            candidate_scan_dists.end(),
                            [](float value) { return std::isfinite(value); }));
        REQUIRE(std::isfinite(candidate_scan_inner_products[0]));
        published_candidate_inner_products[i] = candidate_scan_inner_products[0];
    }
    std::array<InnerIdType, 3> sparse_candidate_ids{1, 3, 5};
    std::array<float, 3> sparse_candidate_inner_products{published_candidate_inner_products[0],
                                                         published_candidate_inner_products[1],
                                                         std::numeric_limits<float>::quiet_NaN()};
    std::array<float, 3> sparse_candidate_dists{};
    SearchStatistics sparse_candidate_stats;
    QueryContext sparse_candidate_ctx{nullptr, &sparse_candidate_stats};
    bucket->QueryWithCandidateFilterInnerProductByInnerId(sparse_candidate_dists.data(),
                                                          nullptr,
                                                          sparse_candidate_inner_products.data(),
                                                          candidate_computer,
                                                          sparse_candidate_ids.data(),
                                                          sparse_candidate_ids.size(),
                                                          &sparse_candidate_ctx);
    REQUIRE(std::all_of(sparse_candidate_dists.begin(),
                        sparse_candidate_dists.end(),
                        [](float value) { return std::isfinite(value); }));
    REQUIRE(sparse_candidate_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) ==
            published_candidate_ids.size());

    std::array<BucketIdType, 3> sparse_source_buckets{1, 3, 1};
    std::array<InnerIdType, 3> sparse_source_offsets{0, 0, 1};
    std::array<uint64_t, 3> sparse_source_versions{published_candidate_versions[0],
                                                   published_candidate_versions[1],
                                                   published_candidate_versions[0]};
    std::array<float, 3> source_candidate_dists{};
    SearchStatistics source_candidate_stats;
    QueryContext source_candidate_ctx{nullptr, &source_candidate_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(source_candidate_dists.data(),
                                                         nullptr,
                                                         sparse_candidate_inner_products.data(),
                                                         sparse_source_buckets.data(),
                                                         sparse_source_offsets.data(),
                                                         sparse_source_versions.data(),
                                                         candidate_computer,
                                                         sparse_candidate_ids.data(),
                                                         sparse_candidate_ids.size(),
                                                         &source_candidate_ctx);
    REQUIRE(source_candidate_dists == sparse_candidate_dists);
    REQUIRE(source_candidate_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) ==
            published_candidate_ids.size());

    const InnerIdType wrong_source_id = 1;
    const BucketIdType wrong_source_bucket = 1;
    const InnerIdType wrong_source_offset = 1;
    const uint64_t wrong_source_version = published_candidate_versions[0];
    const float wrong_source_expected = bucket->QueryOneById(candidate_computer, 1, 0);
    float wrong_source_actual = 0.0F;
    SearchStatistics wrong_source_stats;
    QueryContext wrong_source_ctx{nullptr, &wrong_source_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(&wrong_source_actual,
                                                         nullptr,
                                                         published_candidate_inner_products.data(),
                                                         &wrong_source_bucket,
                                                         &wrong_source_offset,
                                                         &wrong_source_version,
                                                         candidate_computer,
                                                         &wrong_source_id,
                                                         1,
                                                         &wrong_source_ctx);
    REQUIRE(std::abs(wrong_source_actual - wrong_source_expected) < 1e-5F);
    REQUIRE(wrong_source_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) == 0);

    const BucketIdType wrong_bucket_source_bucket = 3;
    const InnerIdType wrong_bucket_source_offset = 0;
    const uint64_t wrong_bucket_source_version = published_candidate_versions[1];
    float wrong_bucket_source_actual = 0.0F;
    SearchStatistics wrong_bucket_source_stats;
    QueryContext wrong_bucket_source_ctx{nullptr, &wrong_bucket_source_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(&wrong_bucket_source_actual,
                                                         nullptr,
                                                         published_candidate_inner_products.data(),
                                                         &wrong_bucket_source_bucket,
                                                         &wrong_bucket_source_offset,
                                                         &wrong_bucket_source_version,
                                                         candidate_computer,
                                                         &wrong_source_id,
                                                         1,
                                                         &wrong_bucket_source_ctx);
    REQUIRE(std::abs(wrong_bucket_source_actual - wrong_source_expected) < 1e-5F);
    REQUIRE(wrong_bucket_source_stats.rabitq_reorder_hint_full_count.load(
                std::memory_order_relaxed) == 0);

    for (const auto bucket_id : std::array<BucketIdType, 2>{1, 3}) {
        const auto bucket_size = bucket->GetBucketSize(bucket_id);
        std::vector<float> expected(bucket_size);
        std::vector<float> actual(bucket_size);
        bucket->ScanBucketById(expected.data(), routed_computer, bucket_id);
        bucket->ScanBucketById(actual.data(), routed_computer, bucket_id);
        REQUIRE(actual == expected);
    }

    std::array<InnerIdType, 8> inner_ids{3, 1, 7, 5, 11, 9, 15, 13};
    std::array<float, 8> scalar_expected{};
    for (uint64_t i = 0; i < inner_ids.size(); ++i) {
        const auto inner_id = inner_ids[i];
        scalar_expected[i] =
            bucket->QueryOneById(routed_computer, inner_id % bucket_count, inner_id / bucket_count);
    }

    SearchStatistics stats;
    QueryContext ctx{nullptr, &stats};
    std::array<float, 8> expected{};
    bucket->QueryWithDistanceHintByInnerId(
        expected.data(), nullptr, routed_computer, inner_ids.data(), inner_ids.size(), &ctx);
    REQUIRE(std::all_of(
        expected.begin(), expected.end(), [](float value) { return std::isfinite(value); }));
    REQUIRE(stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) ==
            inner_ids.size());

    std::array<float, 8> actual{};
    bucket->QueryWithDistanceHintByInnerId(
        actual.data(), nullptr, routed_computer, inner_ids.data(), inner_ids.size());
    REQUIRE(actual == expected);
    bucket->QueryWithFilterInnerProductByInnerId(
        actual.data(), nullptr, routed_computer, inner_ids.data(), inner_ids.size());
    REQUIRE(actual == scalar_expected);

    const std::array<float, 8> hints{0.25F, 1.5F, 2.75F, 4.0F, 5.25F, 6.5F, 7.75F, 9.0F};
    for (uint64_t i = 0; i < inner_ids.size(); ++i) {
        bucket->QueryWithDistanceHintByInnerId(
            expected.data() + i, hints.data() + i, routed_computer, inner_ids.data() + i, 1);
    }
    bucket->QueryWithDistanceHintByInnerId(
        actual.data(), hints.data(), routed_computer, inner_ids.data(), inner_ids.size());
    REQUIRE(actual == expected);
    const auto heap_bucket_size = bucket->GetBucketSize(3);
    std::vector<float> heap_scan_dists(heap_bucket_size);
    std::vector<float> heap_lower_bounds(heap_bucket_size);
    std::vector<float> heap_filter_inner_products(heap_bucket_size);
    bucket->ScanBucketWithDistanceLowerBound(heap_scan_dists.data(),
                                             heap_lower_bounds.data(),
                                             heap_filter_inner_products.data(),
                                             routed_computer,
                                             3);
    REQUIRE(std::isfinite(heap_filter_inner_products[0]));
    constexpr InnerIdType updated_inner_id = 3;
    SearchStatistics heap_stats_before_update;
    QueryContext heap_ctx_before_update{nullptr, &heap_stats_before_update};
    float heap_actual_before_update = 0.0F;
    bucket->QueryWithFilterInnerProductByInnerId(&heap_actual_before_update,
                                                 heap_filter_inner_products.data(),
                                                 routed_computer,
                                                 &updated_inner_id,
                                                 1,
                                                 &heap_ctx_before_update);
    REQUIRE(std::isfinite(heap_actual_before_update));
    REQUIRE(heap_stats_before_update.rabitq_reorder_hint_full_count.load(
                std::memory_order_relaxed) == 1);

    constexpr BucketIdType updated_bucket_id = 3;
    constexpr InnerIdType updated_offset_id = 0;
    bucket->InsertVectorWithOffset(
        vectors.data() + 20 * dim, updated_bucket_id, updated_inner_id, updated_offset_id);
    const float updated_expected =
        bucket->QueryOneById(routed_computer, updated_bucket_id, updated_offset_id);
    float updated_actual = 0.0F;
    bucket->QueryWithDistanceHintByInnerId(
        &updated_actual, nullptr, routed_computer, &updated_inner_id, 1);
    REQUIRE(std::abs(updated_actual - updated_expected) < 1e-5F);

    SearchStatistics sparse_candidate_stats_after_update;
    QueryContext sparse_candidate_ctx_after_update{nullptr, &sparse_candidate_stats_after_update};
    float sparse_candidate_updated_actual = 0.0F;
    bucket->QueryWithCandidateFilterInnerProductByInnerId(
        &sparse_candidate_updated_actual,
        nullptr,
        published_candidate_inner_products.data() + 1,
        candidate_computer,
        &updated_inner_id,
        1,
        &sparse_candidate_ctx_after_update);
    REQUIRE(std::abs(sparse_candidate_updated_actual - updated_expected) < 1e-5F);
    REQUIRE(sparse_candidate_stats_after_update.rabitq_reorder_hint_full_count.load(
                std::memory_order_relaxed) == 0);

    const BucketIdType stale_source_bucket = updated_bucket_id;
    const InnerIdType stale_source_offset = updated_offset_id;
    const uint64_t stale_source_version = published_candidate_versions[1];
    float stale_source_actual = 0.0F;
    SearchStatistics stale_source_stats;
    QueryContext stale_source_ctx{nullptr, &stale_source_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(
        &stale_source_actual,
        nullptr,
        published_candidate_inner_products.data() + 1,
        &stale_source_bucket,
        &stale_source_offset,
        &stale_source_version,
        candidate_computer,
        &updated_inner_id,
        1,
        &stale_source_ctx);
    REQUIRE(std::abs(stale_source_actual - updated_expected) < 1e-5F);
    REQUIRE(stale_source_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) == 0);

    SearchStatistics heap_stats_after_update;
    QueryContext heap_ctx_after_update{nullptr, &heap_stats_after_update};
    float heap_updated_actual = 0.0F;
    bucket->QueryWithFilterInnerProductByInnerId(&heap_updated_actual,
                                                 heap_filter_inner_products.data(),
                                                 routed_computer,
                                                 &updated_inner_id,
                                                 1,
                                                 &heap_ctx_after_update);
    REQUIRE(std::abs(heap_updated_actual - updated_expected) < 1e-5F);
    REQUIRE(heap_stats_after_update.rabitq_reorder_hint_full_count.load(
                std::memory_order_relaxed) == 0);

    REQUIRE_THROWS(bucket->ScanBucketById(nullptr, routed_computer, 0));
}

TEST_CASE("RaBitQ split bucket reuses non-residual filter inner products",
          "[ut][BucketDataCell][rabitq_split][routed_query]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 64;
    constexpr BucketIdType bucket_count = 2;
    const uint32_t filter_bits = GENERATE(1U, 2U, 3U);
    INFO("filter_bits=" << filter_bits);
    auto vectors = fixtures::generate_vectors(count, dim);
    auto query = fixtures::generate_vectors(1, dim, false, 79);
    constexpr const char* param_str = R"({
        "io_params": { "type": "memory_io" },
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 1,
            "fast_encode_rabitq": true
        },
        "buckets_count": 2,
        "use_residual": false
    })";

    auto param = std::make_shared<BucketDataCellParameter>();
    auto json = JsonType::Parse(param_str);
    json["quantization_params"]["rabitq_bits_per_dim_filter"].SetInt(filter_bits);
    param->FromJson(json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto bucket = BucketInterface::MakeInstance(param, common_param);
    bucket->Train(vectors.data(), count);
    for (InnerIdType id = 0; id < count; ++id) {
        bucket->InsertVector(
            vectors.data() + static_cast<uint64_t>(id) * dim, id % bucket_count, id);
    }
    bucket->Package();

    std::array<BucketIdType, 2> routed_buckets{1, 0};
    auto routed_computer = bucket->FactoryComputerForBuckets(
        query.data(), routed_buckets.data(), routed_buckets.size());
    auto candidate_computer = bucket->FactoryComputerForBuckets(
        query.data(), routed_buckets.data(), routed_buckets.size());
    std::array<InnerIdType, 2> published_candidate_ids{1, 0};
    std::array<float, 2> published_candidate_inner_products{};
    std::array<uint64_t, 2> published_candidate_versions{};
    for (uint64_t i = 0; i < published_candidate_ids.size(); ++i) {
        const auto bucket_id = static_cast<BucketIdType>(published_candidate_ids[i] % bucket_count);
        const auto bucket_size = bucket->GetBucketSize(bucket_id);
        std::vector<float> candidate_scan_dists(bucket_size);
        std::vector<float> candidate_scan_inner_products(bucket_size);
        std::vector<InnerIdType> candidate_scan_ids(bucket_size);
        InnerIdType candidate_scan_size = 0;
        published_candidate_versions[i] =
            bucket->ScanBucketWithFilterInnerProduct(candidate_scan_dists.data(),
                                                     candidate_scan_inner_products.data(),
                                                     candidate_computer,
                                                     bucket_id,
                                                     nullptr,
                                                     candidate_scan_ids.data(),
                                                     bucket_size,
                                                     &candidate_scan_size);
        REQUIRE(candidate_scan_size == bucket_size);
        REQUIRE(candidate_scan_ids[0] == published_candidate_ids[i]);
        REQUIRE(std::all_of(candidate_scan_dists.begin(),
                            candidate_scan_dists.end(),
                            [](float value) { return std::isfinite(value); }));
        REQUIRE(std::isfinite(candidate_scan_inner_products[0]));
        published_candidate_inner_products[i] = candidate_scan_inner_products[0];
    }
    std::array<InnerIdType, 3> sparse_candidate_ids{1, 0, 3};
    std::array<float, 3> sparse_candidate_inner_products{published_candidate_inner_products[0],
                                                         published_candidate_inner_products[1],
                                                         std::numeric_limits<float>::quiet_NaN()};
    std::array<float, 3> sparse_candidate_dists{};
    SearchStatistics sparse_candidate_stats;
    QueryContext sparse_candidate_ctx{nullptr, &sparse_candidate_stats};
    bucket->QueryWithCandidateFilterInnerProductByInnerId(sparse_candidate_dists.data(),
                                                          nullptr,
                                                          sparse_candidate_inner_products.data(),
                                                          candidate_computer,
                                                          sparse_candidate_ids.data(),
                                                          sparse_candidate_ids.size(),
                                                          &sparse_candidate_ctx);
    REQUIRE(std::all_of(sparse_candidate_dists.begin(),
                        sparse_candidate_dists.end(),
                        [](float value) { return std::isfinite(value); }));
    REQUIRE(sparse_candidate_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) ==
            published_candidate_ids.size());

    std::array<BucketIdType, 3> sparse_source_buckets{1, 0, 1};
    std::array<InnerIdType, 3> sparse_source_offsets{0, 0, 1};
    std::array<uint64_t, 3> sparse_source_versions{published_candidate_versions[0],
                                                   published_candidate_versions[1],
                                                   published_candidate_versions[0]};
    std::array<float, 3> source_candidate_dists{};
    SearchStatistics source_candidate_stats;
    QueryContext source_candidate_ctx{nullptr, &source_candidate_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(source_candidate_dists.data(),
                                                         nullptr,
                                                         sparse_candidate_inner_products.data(),
                                                         sparse_source_buckets.data(),
                                                         sparse_source_offsets.data(),
                                                         sparse_source_versions.data(),
                                                         candidate_computer,
                                                         sparse_candidate_ids.data(),
                                                         sparse_candidate_ids.size(),
                                                         &source_candidate_ctx);
    REQUIRE(source_candidate_dists == sparse_candidate_dists);
    REQUIRE(source_candidate_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) ==
            published_candidate_ids.size());

    const InnerIdType wrong_source_id = 1;
    const BucketIdType wrong_source_bucket = 1;
    const InnerIdType wrong_source_offset = 1;
    const uint64_t wrong_source_version = published_candidate_versions[0];
    const float wrong_source_expected = bucket->QueryOneById(candidate_computer, 1, 0);
    float wrong_source_actual = 0.0F;
    SearchStatistics wrong_source_stats;
    QueryContext wrong_source_ctx{nullptr, &wrong_source_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(&wrong_source_actual,
                                                         nullptr,
                                                         published_candidate_inner_products.data(),
                                                         &wrong_source_bucket,
                                                         &wrong_source_offset,
                                                         &wrong_source_version,
                                                         candidate_computer,
                                                         &wrong_source_id,
                                                         1,
                                                         &wrong_source_ctx);
    REQUIRE(std::abs(wrong_source_actual - wrong_source_expected) < 1e-5F);
    REQUIRE(wrong_source_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) == 0);

    const BucketIdType wrong_bucket_source_bucket = 0;
    const InnerIdType wrong_bucket_source_offset = 0;
    const uint64_t wrong_bucket_source_version = published_candidate_versions[1];
    float wrong_bucket_source_actual = 0.0F;
    SearchStatistics wrong_bucket_source_stats;
    QueryContext wrong_bucket_source_ctx{nullptr, &wrong_bucket_source_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(&wrong_bucket_source_actual,
                                                         nullptr,
                                                         published_candidate_inner_products.data(),
                                                         &wrong_bucket_source_bucket,
                                                         &wrong_bucket_source_offset,
                                                         &wrong_bucket_source_version,
                                                         candidate_computer,
                                                         &wrong_source_id,
                                                         1,
                                                         &wrong_bucket_source_ctx);
    REQUIRE(std::abs(wrong_bucket_source_actual - wrong_source_expected) < 1e-5F);
    REQUIRE(wrong_bucket_source_stats.rabitq_reorder_hint_full_count.load(
                std::memory_order_relaxed) == 0);

    for (const auto bucket_id : routed_buckets) {
        std::vector<float> scan_dists(bucket->GetBucketSize(bucket_id));
        bucket->ScanBucketById(scan_dists.data(), routed_computer, bucket_id);
    }

    std::array<InnerIdType, 6> inner_ids{7, 0, 5, 2, 9, 4};
    SearchStatistics stats;
    QueryContext ctx{nullptr, &stats};
    std::array<float, 6> expected{};
    bucket->QueryWithDistanceHintByInnerId(
        expected.data(), nullptr, routed_computer, inner_ids.data(), inner_ids.size(), &ctx);
    REQUIRE(std::all_of(
        expected.begin(), expected.end(), [](float value) { return std::isfinite(value); }));
    REQUIRE(stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) ==
            inner_ids.size());

    std::array<float, 6> actual{};
    bucket->QueryWithDistanceHintByInnerId(
        actual.data(), nullptr, routed_computer, inner_ids.data(), inner_ids.size());
    REQUIRE(actual == expected);

    const auto heap_bucket_size = bucket->GetBucketSize(0);
    std::vector<float> heap_scan_dists(heap_bucket_size);
    std::vector<float> heap_lower_bounds(heap_bucket_size);
    std::vector<float> heap_filter_inner_products(heap_bucket_size);
    bucket->ScanBucketWithDistanceLowerBound(heap_scan_dists.data(),
                                             heap_lower_bounds.data(),
                                             heap_filter_inner_products.data(),
                                             routed_computer,
                                             0);
    REQUIRE(std::isfinite(heap_filter_inner_products[0]));
    constexpr InnerIdType updated_inner_id = 0;
    SearchStatistics heap_stats_before_update;
    QueryContext heap_ctx_before_update{nullptr, &heap_stats_before_update};
    float heap_actual_before_update = 0.0F;
    bucket->QueryWithFilterInnerProductByInnerId(&heap_actual_before_update,
                                                 heap_filter_inner_products.data(),
                                                 routed_computer,
                                                 &updated_inner_id,
                                                 1,
                                                 &heap_ctx_before_update);
    REQUIRE(std::isfinite(heap_actual_before_update));
    REQUIRE(heap_stats_before_update.rabitq_reorder_hint_full_count.load(
                std::memory_order_relaxed) == 1);

    constexpr InnerIdType updated_offset_id = 0;
    bucket->InsertVectorWithOffset(
        vectors.data() + 20 * dim, 0, updated_inner_id, updated_offset_id);
    const float updated_expected = bucket->QueryOneById(routed_computer, 0, updated_offset_id);

    float candidate_updated_actual = 0.0F;
    bucket->QueryWithDistanceHintByInnerId(
        &candidate_updated_actual, nullptr, routed_computer, &updated_inner_id, 1);
    REQUIRE(std::abs(candidate_updated_actual - updated_expected) < 1e-5F);

    SearchStatistics sparse_candidate_stats_after_update;
    QueryContext sparse_candidate_ctx_after_update{nullptr, &sparse_candidate_stats_after_update};
    float sparse_candidate_updated_actual = 0.0F;
    bucket->QueryWithCandidateFilterInnerProductByInnerId(
        &sparse_candidate_updated_actual,
        nullptr,
        published_candidate_inner_products.data() + 1,
        candidate_computer,
        &updated_inner_id,
        1,
        &sparse_candidate_ctx_after_update);
    REQUIRE(std::abs(sparse_candidate_updated_actual - updated_expected) < 1e-5F);
    REQUIRE(sparse_candidate_stats_after_update.rabitq_reorder_hint_full_count.load(
                std::memory_order_relaxed) == 0);

    const BucketIdType stale_source_bucket = 0;
    const InnerIdType stale_source_offset = updated_offset_id;
    const uint64_t stale_source_version = published_candidate_versions[1];
    float stale_source_actual = 0.0F;
    SearchStatistics stale_source_stats;
    QueryContext stale_source_ctx{nullptr, &stale_source_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(
        &stale_source_actual,
        nullptr,
        published_candidate_inner_products.data() + 1,
        &stale_source_bucket,
        &stale_source_offset,
        &stale_source_version,
        candidate_computer,
        &updated_inner_id,
        1,
        &stale_source_ctx);
    REQUIRE(std::abs(stale_source_actual - updated_expected) < 1e-5F);
    REQUIRE(stale_source_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) == 0);

    SearchStatistics heap_stats_after_update;
    QueryContext heap_ctx_after_update{nullptr, &heap_stats_after_update};
    float heap_updated_actual = 0.0F;
    bucket->QueryWithFilterInnerProductByInnerId(&heap_updated_actual,
                                                 heap_filter_inner_products.data(),
                                                 routed_computer,
                                                 &updated_inner_id,
                                                 1,
                                                 &heap_ctx_after_update);
    REQUIRE(std::abs(heap_updated_actual - updated_expected) < 1e-5F);
    REQUIRE(heap_stats_after_update.rabitq_reorder_hint_full_count.load(
                std::memory_order_relaxed) == 0);
}

TEST_CASE("RaBitQ split bucket relocates existing inner IDs and invalidates provenance",
          "[ut][BucketDataCell][rabitq_split][relocation]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType train_count = 32;
    constexpr BucketIdType bucket_count = 2;
    const bool use_residual = GENERATE(false, true);
    const uint32_t filter_bits = GENERATE(1U, 2U, 3U);
    INFO("use_residual=" << use_residual << ", filter_bits=" << filter_bits);

    auto vectors = fixtures::generate_vectors(train_count, dim);
    auto query = fixtures::generate_vectors(1, dim, false, 97);
    constexpr const char* param_str = R"({
        "io_params": { "type": "memory_io" },
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 1,
            "fast_encode_rabitq": true
        },
        "buckets_count": 2,
        "use_residual": false
    })";

    auto json = JsonType::Parse(param_str);
    json["quantization_params"]["rabitq_bits_per_dim_filter"].SetInt(filter_bits);
    json["use_residual"].SetBool(use_residual);
    auto param = std::make_shared<BucketDataCellParameter>();
    param->FromJson(json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto bucket = BucketInterface::MakeInstance(param, common_param);
    auto strategy = std::make_shared<FixedCentroidPartitionStrategy>(common_param, bucket_count);
    bucket->SetStrategy(strategy);
    bucket->Train(vectors.data(), train_count);

    constexpr InnerIdType fixed_source_id = 0;
    constexpr InnerIdType append_source_id = 2;
    constexpr InnerIdType displaced_id = 1;
    bucket->InsertVector(vectors.data(), 0, fixed_source_id);
    bucket->InsertVector(vectors.data() + dim, 0, append_source_id);
    bucket->InsertVector(vectors.data() + 2 * dim, 1, displaced_id);
    bucket->Package();

    std::array<BucketIdType, 2> routed_buckets{0, 1};
    auto computer = bucket->FactoryComputerForBuckets(
        query.data(), routed_buckets.data(), routed_buckets.size());

    std::array<float, 2> old_source_dists{};
    std::array<float, 2> old_source_inner_products{};
    std::array<InnerIdType, 2> old_source_ids{};
    InnerIdType old_source_size = 0;
    const uint64_t old_source_version =
        bucket->ScanBucketWithFilterInnerProduct(old_source_dists.data(),
                                                 old_source_inner_products.data(),
                                                 computer,
                                                 0,
                                                 nullptr,
                                                 old_source_ids.data(),
                                                 old_source_ids.size(),
                                                 &old_source_size);
    REQUIRE(old_source_size == old_source_ids.size());
    REQUIRE(old_source_ids[0] == fixed_source_id);
    REQUIRE(old_source_ids[1] == append_source_id);

    float displaced_source_dist = 0.0F;
    float displaced_source_inner_product = 0.0F;
    InnerIdType displaced_source_scanned_id = std::numeric_limits<InnerIdType>::max();
    InnerIdType displaced_source_size = 0;
    const uint64_t displaced_source_version =
        bucket->ScanBucketWithFilterInnerProduct(&displaced_source_dist,
                                                 &displaced_source_inner_product,
                                                 computer,
                                                 1,
                                                 nullptr,
                                                 &displaced_source_scanned_id,
                                                 1,
                                                 &displaced_source_size);
    REQUIRE(displaced_source_size == 1);
    REQUIRE(displaced_source_scanned_id == displaced_id);

    bucket->InsertVectorWithOffset(vectors.data() + 3 * dim, 1, fixed_source_id, 0);

    std::array<float, 1> target_dists{};
    std::array<float, 1> target_inner_products{};
    std::array<InnerIdType, 1> target_ids{};
    InnerIdType target_size = 0;
    const uint64_t target_version =
        bucket->ScanBucketWithFilterInnerProduct(target_dists.data(),
                                                 target_inner_products.data(),
                                                 computer,
                                                 1,
                                                 nullptr,
                                                 target_ids.data(),
                                                 target_ids.size(),
                                                 &target_size);
    REQUIRE(target_size == 1);
    REQUIRE(target_ids[0] == fixed_source_id);
    REQUIRE(target_version > 0);
    REQUIRE(std::isfinite(target_inner_products[0]));

    const float fixed_expected = bucket->QueryOneById(computer, 1, 0);
    float fixed_actual = 0.0F;
    const BucketIdType old_bucket_id = 0;
    const InnerIdType old_offset_id = 0;
    SearchStatistics fixed_stats;
    QueryContext fixed_ctx{nullptr, &fixed_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(&fixed_actual,
                                                         nullptr,
                                                         old_source_inner_products.data(),
                                                         &old_bucket_id,
                                                         &old_offset_id,
                                                         &old_source_version,
                                                         computer,
                                                         &fixed_source_id,
                                                         1,
                                                         &fixed_ctx);
    REQUIRE(std::abs(fixed_actual - fixed_expected) < 1e-5F);
    REQUIRE(fixed_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) == 0);

    float displaced_dist = 0.0F;
    REQUIRE_THROWS(bucket->QueryWithDistanceHintByInnerId(
        &displaced_dist, nullptr, computer, &displaced_id, 1));

    float displaced_candidate_dist = 0.0F;
    const BucketIdType displaced_source_bucket = 1;
    const InnerIdType displaced_source_offset = 0;
    REQUIRE_NOTHROW(
        bucket->QueryWithCandidateFilterInnerProductBySource(&displaced_candidate_dist,
                                                             nullptr,
                                                             &displaced_source_inner_product,
                                                             &displaced_source_bucket,
                                                             &displaced_source_offset,
                                                             &displaced_source_version,
                                                             computer,
                                                             &displaced_id,
                                                             1));
    REQUIRE(displaced_candidate_dist == std::numeric_limits<float>::max());

    std::array<float, 2> source_after_fixed_dists{};
    std::array<float, 2> source_after_fixed_inner_products{};
    std::array<InnerIdType, 2> source_after_fixed_ids{};
    InnerIdType source_after_fixed_size = 0;
    const uint64_t source_after_fixed_version =
        bucket->ScanBucketWithFilterInnerProduct(source_after_fixed_dists.data(),
                                                 source_after_fixed_inner_products.data(),
                                                 computer,
                                                 0,
                                                 nullptr,
                                                 source_after_fixed_ids.data(),
                                                 source_after_fixed_ids.size(),
                                                 &source_after_fixed_size);
    REQUIRE(source_after_fixed_size == source_after_fixed_ids.size());
    REQUIRE(source_after_fixed_version != old_source_version);
    REQUIRE(source_after_fixed_ids[0] == std::numeric_limits<InnerIdType>::max());
    REQUIRE(source_after_fixed_ids[1] == append_source_id);
    REQUIRE(source_after_fixed_dists[0] == std::numeric_limits<float>::max());
    uint32_t empty_inner_product_bits = 0;
    std::memcpy(&empty_inner_product_bits,
                source_after_fixed_inner_products.data(),
                sizeof(empty_inner_product_bits));
    REQUIRE((empty_inner_product_bits & 0x7F800000U) == 0x7F800000U);
    REQUIRE((empty_inner_product_bits & 0x007FFFFFU) != 0U);

    const InnerIdType appended_offset =
        bucket->InsertVector(vectors.data() + 4 * dim, 1, append_source_id);
    REQUIRE(appended_offset == 1);
    REQUIRE(bucket->GetInnerIds(0)[1] == std::numeric_limits<InnerIdType>::max());

    const float append_expected = bucket->QueryOneById(computer, 1, appended_offset);
    float append_actual = 0.0F;
    const InnerIdType append_old_offset = 1;
    SearchStatistics append_stats;
    QueryContext append_ctx{nullptr, &append_stats};
    bucket->QueryWithCandidateFilterInnerProductBySource(
        &append_actual,
        nullptr,
        source_after_fixed_inner_products.data() + 1,
        &old_bucket_id,
        &append_old_offset,
        &source_after_fixed_version,
        computer,
        &append_source_id,
        1,
        &append_ctx);
    REQUIRE(std::abs(append_actual - append_expected) < 1e-5F);
    REQUIRE(append_stats.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed) == 0);

    constexpr InnerIdType concurrent_id = 7;
    const float* concurrent_vector0 = vectors.data() + 5 * dim;
    const float* concurrent_vector1 = vectors.data() + 6 * dim;
    const InnerIdType concurrent_offset0 =
        bucket->InsertVector(concurrent_vector0, 0, concurrent_id);
    const float concurrent_expected0 = bucket->QueryOneById(computer, 0, concurrent_offset0);
    const InnerIdType concurrent_offset1 =
        bucket->InsertVector(concurrent_vector1, 1, concurrent_id);
    const float concurrent_expected1 = bucket->QueryOneById(computer, 1, concurrent_offset1);
    bucket->InsertVector(concurrent_vector0, 0, concurrent_id);

    std::atomic<bool> start_relocation{false};
    constexpr uint32_t relocation_count = 256;
    auto relocation = std::async(std::launch::async, [&]() {
        while (not start_relocation.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (uint32_t i = 0; i < relocation_count; ++i) {
            const BucketIdType target_bucket = (i & 1U) == 0U ? 1 : 0;
            const float* target_vector =
                target_bucket == 0 ? concurrent_vector0 : concurrent_vector1;
            bucket->InsertVector(target_vector, target_bucket, concurrent_id);
            std::this_thread::yield();
        }
    });

    start_relocation.store(true, std::memory_order_release);
    for (uint32_t i = 0; i < relocation_count; ++i) {
        float concurrent_actual = 0.0F;
        REQUIRE_NOTHROW(
            bucket->QueryWithCandidateFilterInnerProductBySource(&concurrent_actual,
                                                                 nullptr,
                                                                 old_source_inner_products.data(),
                                                                 &old_bucket_id,
                                                                 &old_offset_id,
                                                                 &old_source_version,
                                                                 computer,
                                                                 &concurrent_id,
                                                                 1));
        const bool is_complete_old = std::abs(concurrent_actual - concurrent_expected0) < 1e-5F;
        const bool is_complete_new = std::abs(concurrent_actual - concurrent_expected1) < 1e-5F;
        const bool is_unavailable = concurrent_actual == std::numeric_limits<float>::max();
        REQUIRE((is_complete_old or is_complete_new or is_unavailable));
    }
    REQUIRE_NOTHROW(relocation.get());
}

TEST_CASE("RaBitQ split bucket rejects optimized build without fast encoding",
          "[ut][BucketDataCell][rabitq_split][optimized_build]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 8;
    constexpr BucketIdType bucket_count = 2;
    auto vectors = fixtures::generate_vectors(count, dim);
    constexpr const char* param_str = R"({
        "io_params": { "type": "memory_io" },
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 2,
            "fast_encode_rabitq": false
        },
        "buckets_count": 2
    })";

    auto param = std::make_shared<BucketDataCellParameter>();
    param->FromJson(JsonType::Parse(param_str));
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto bucket = BucketInterface::MakeInstance(param, common_param);
    bucket->Train(vectors.data(), count);

    auto thread_pool = SafeThreadPool::FactoryDefaultThreadPool();
    thread_pool->SetPoolSize(2);
    FlattenOptimizedBuildContext context{thread_pool, 2};
    REQUIRE_FALSE(bucket->BeginOptimizedBuild(context, count));
    REQUIRE_FALSE(bucket->IsOptimizedBuildActive());

    for (InnerIdType id = 0; id < count; ++id) {
        const auto bucket_id = static_cast<BucketIdType>(id % bucket_count);
        bucket->InsertVector(vectors.data() + static_cast<uint64_t>(id) * dim, bucket_id, id);
    }
    bucket->Package();
    InnerIdType stored_count = 0;
    for (BucketIdType bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
        stored_count += bucket->GetBucketSize(bucket_id);
    }
    REQUIRE(stored_count == count);
}

TEST_CASE("RaBitQ split bucket keeps only canonical packed filter codes",
          "[ut][BucketDataCell][rabitq_split][fastscan][serialization]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 70;
    constexpr InnerIdType count = 384;
    constexpr BucketIdType bucket_count = 3;
    auto vectors = fixtures::generate_vectors(count + 2, dim);
    auto query = fixtures::generate_vectors(1, dim, false, 41);
    constexpr const char* param_str = R"({
        "io_params": { "type": "memory_io" },
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 1,
            "fast_encode_rabitq": true
        },
        "buckets_count": 3,
        "use_residual": true
    })";

    auto make_bucket = [&](uint32_t filter_bits, float centroid_scale = 1.0F) {
        auto json = JsonType::Parse(param_str);
        json["quantization_params"]["rabitq_bits_per_dim_filter"].SetInt(filter_bits);
        auto param = std::make_shared<BucketDataCellParameter>();
        param->FromJson(json);
        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
        auto bucket = BucketInterface::MakeInstance(param, common_param);
        bucket->SetStrategy(std::make_shared<FixedCentroidPartitionStrategy>(
            common_param, bucket_count, centroid_scale));
        return bucket;
    };

    std::array<uint64_t, 3> memory_usage{};
    std::array<uint64_t, 3> serialized_size{};
    BucketInterfacePtr packed_bucket = nullptr;
    for (uint32_t filter_bits = 1; filter_bits <= 3; ++filter_bits) {
        auto bucket = make_bucket(filter_bits);
        bucket->Train(vectors.data(), count);
        for (InnerIdType id = 0; id < count; ++id) {
            bucket->InsertVector(vectors.data() + static_cast<uint64_t>(id) * dim,
                                 static_cast<BucketIdType>(id % bucket_count),
                                 id);
        }
        bucket->Package();
        bucket->Unpack();
        memory_usage[filter_bits - 1] = bucket->GetMemoryUsage();
        CountingStreamWriter writer;
        bucket->Serialize(writer);
        serialized_size[filter_bits - 1] = writer.GetCursor();
        if (filter_bits == 3) {
            packed_bucket = std::move(bucket);
        }
    }
    const auto [min_memory, max_memory] =
        std::minmax_element(memory_usage.begin(), memory_usage.end());
    const auto [min_size, max_size] =
        std::minmax_element(serialized_size.begin(), serialized_size.end());
    REQUIRE(*max_memory - *min_memory < 4096);
    REQUIRE(*max_size - *min_size < 4096);

    std::vector<std::vector<uint8_t>> expected_codes(
        count, std::vector<uint8_t>(packed_bucket->code_size_));
    for (InnerIdType id = 0; id < count; ++id) {
        const auto bucket_id = static_cast<BucketIdType>(id % bucket_count);
        const auto offset_id = static_cast<InnerIdType>(id / bucket_count);
        packed_bucket->GetCodesById(bucket_id, offset_id, expected_codes[id].data());
    }

    auto candidate_for_offset = [&](const BucketInterfacePtr& bucket,
                                    BucketIdType bucket_id,
                                    InnerIdType offset_id,
                                    InnerIdType inner_id) {
        auto computer = bucket->FactoryComputerForBuckets(query.data(), &bucket_id, 1);
        const auto bucket_size = bucket->GetBucketSize(bucket_id);
        std::vector<float> scan_dists(bucket_size);
        std::vector<float> filter_inner_products(bucket_size);
        bucket->ScanBucketWithFilterInnerProduct(
            scan_dists.data(), filter_inner_products.data(), computer, bucket_id);
        float result = 0.0F;
        bucket->QueryWithCandidateFilterInnerProductByInnerId(
            &result, nullptr, filter_inner_products.data() + offset_id, computer, &inner_id, 1);
        return result;
    };
    const float serialized_candidate = candidate_for_offset(packed_bucket, 1, 0, 1);

    std::stringstream stream;
    IOStreamWriter writer(stream);
    packed_bucket->Serialize(writer);
    stream.seekg(0, std::ios::beg);
    IOStreamReader reader(stream);
    auto restored = make_bucket(3);
    restored->Deserialize(reader);
    REQUIRE(reader.GetCursor() == writer.GetCursor());
    for (InnerIdType id = 0; id < count; ++id) {
        const auto bucket_id = static_cast<BucketIdType>(id % bucket_count);
        const auto offset_id = static_cast<InnerIdType>(id / bucket_count);
        std::vector<uint8_t> actual(restored->code_size_);
        restored->GetCodesById(bucket_id, offset_id, actual.data());
        REQUIRE(actual == expected_codes[id]);
    }
    REQUIRE(std::abs(candidate_for_offset(restored, 1, 0, 1) - serialized_candidate) < 1e-5F);

    constexpr BucketIdType added_bucket = 1;
    const auto added_offset = restored->InsertVector(
        vectors.data() + static_cast<uint64_t>(count) * dim, added_bucket, count);
    auto computer = restored->FactoryComputer(query.data());
    REQUIRE(std::isfinite(restored->QueryOneById(computer, added_bucket, added_offset)));

    auto source = make_bucket(3, 2.0F);
    restored->ExportModel(source);
    constexpr BucketIdType merged_bucket = 2;
    source->InsertVector(vectors.data() + static_cast<uint64_t>(count + 1) * dim, merged_bucket, 0);
    source->Package();
    std::vector<uint8_t> source_code(source->code_size_);
    source->GetCodesById(merged_bucket, 0, source_code.data());
    const auto old_bucket_size = restored->GetBucketSize(merged_bucket);
    restored->MergeOther(source, count + 1);
    std::vector<uint8_t> merged_code(restored->code_size_);
    restored->GetCodesById(merged_bucket, old_bucket_size, merged_code.data());
    REQUIRE(merged_code == source_code);
    const float merged_candidate =
        candidate_for_offset(restored, merged_bucket, old_bucket_size, count + 1);
    auto merged_computer = restored->FactoryComputerForBuckets(query.data(), &merged_bucket, 1);
    const float merged_expected =
        restored->QueryOneById(merged_computer, merged_bucket, old_bucket_size);
    REQUIRE(std::abs(merged_candidate - merged_expected) <=
            0.02F * std::max({1.0F, std::abs(merged_candidate), std::abs(merged_expected)}));
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
