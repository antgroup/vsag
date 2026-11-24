// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <thread>

#include "half_dense_test_fixture.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "vsag/factory.h"

TEST_CASE("BruteForce native half dense lifecycle", "[ut][bruteforce][half_input]") {
    for (bool bf16 : {false, true}) {
        for (auto metric : {vsag::MetricType::METRIC_TYPE_L2SQR,
                            vsag::MetricType::METRIC_TYPE_IP,
                            vsag::MetricType::METRIC_TYPE_COSINE}) {
            CAPTURE(bf16, metric);
            half_dense_test::Exercise(false, bf16, metric, 4);
        }
    }
}

TEST_CASE("BruteForce native half FP32 storage exposes lossless public raw export",
          "[ut][bruteforce][half_capabilities]") {
    for (bool bf16 : {false, true}) {
        CAPTURE(bf16);
        auto config = vsag::JsonType::Parse(
            R"({"dtype":"float16","metric_type":"l2","dim":8,"brute_force":{"base_quantization_type":"fp32"}})");
        config["dtype"].SetString(bf16 ? "bfloat16" : "float16");
        auto made = vsag::Factory::CreateIndex("brute_force", config.Dump());
        REQUIRE(made.has_value());
        auto index = made.value();
        REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_ADD_FROM_EMPTY));
        REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_RANGE_SEARCH));
        REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS));
        REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::NEED_TRAIN));
        half_dense_test::Rows rows(4, 10, bf16);
        auto added = index->Add(rows.Data(true));
        REQUIRE(added.has_value());
        REQUIRE(added.value().empty());
        const int64_t labels[] = {rows.ids[3], rows.ids[0]};
        auto result = index->GetRawVectorByIds(labels, 2);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetNumElements() == 2);
        REQUIRE(result.value()->GetDim() == half_dense_test::DIM);
        const float* decoded = result.value()->GetFloat32Vectors();
        REQUIRE(decoded != nullptr);
        for (int64_t j = 0; j < half_dense_test::DIM; ++j) {
            REQUIRE(decoded[j] == rows.fp32[3 * half_dense_test::DIM + j]);
            REQUIRE(decoded[half_dense_test::DIM + j] == rows.fp32[j]);
        }
    }
}

namespace {

class RejectSecondAddTask final : public vsag::ThreadPool {
public:
    ~RejectSecondAddTask() override {
        WaitUntilEmpty();
    }

    void
    WaitUntilEmpty() override {
        release_.store(true, std::memory_order_release);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void
    SetQueueSizeLimit(uint64_t) override {
    }

    void
    SetPoolSize(uint64_t) override {
    }

    std::future<void>
    Enqueue(std::function<void(void)> task) override {
        if (worker_.joinable()) {
            release_.store(true, std::memory_order_release);
            throw std::runtime_error("injected BruteForce enqueue failure");
        }
        worker_ = std::thread([this, task = std::move(task)]() {
            while (not release_.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            task();
        });
        return {};
    }

private:
    std::atomic<bool> release_{false};
    std::thread worker_;
};

}  // namespace

TEST_CASE("BruteForce half Add drains pending work after an enqueue failure",
          "[ut][bruteforce][half_add_drain]") {
    const bool bf16 = GENERATE(false, true);
    auto pool = std::make_shared<RejectSecondAddTask>();
    vsag::IndexCommonParam common;
    common.dim_ = half_dense_test::DIM;
    common.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(pool);
    const auto json = vsag::JsonType::Parse(R"({"base_quantization_type":"fp32"})");
    auto param = vsag::BruteForce::CheckAndMappingExternalParam(json, common);
    auto index = std::make_shared<vsag::BruteForce>(param, common);
    index->InitFeatures();
    half_dense_test::Rows rows(2, 50, bf16);
    auto data = rows.Data(true);
    std::exception_ptr failure;
    try {
        index->Add(data);
    } catch (...) {
        failure = std::current_exception();
    }
    const auto count_at_return = index->GetNumElements();
    pool->WaitUntilEmpty();
    REQUIRE(failure != nullptr);
    REQUIRE_THROWS_AS(std::rethrow_exception(failure), std::runtime_error);
    REQUIRE(count_at_return == 1);
    auto result = index->KnnSearch(rows.Data(true, 0, 1), 1, "{}", vsag::FilterPtr{});
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == rows.ids[0]);
    REQUIRE(result->GetDistances()[0] == 0.0F);
}
