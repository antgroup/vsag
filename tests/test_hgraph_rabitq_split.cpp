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

#include <fmt/format.h>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>

#include "functest.h"
#include "test_index.h"
#include "vsag/engine.h"
#include "vsag/resource.h"
#include "vsag/thread_pool.h"

namespace fixtures {

class HGraphRaBitQSplitTestIndex : public TestIndex {
public:
    static const std::string name;
    static fixtures::TempDir dir;
    static TestDatasetPool pool;

    // Build a hgraph parameter JSON exercising the RaBitQ split datacell.
    // - base_io_type: IO backend for the one-bit storage (and homogeneous
    //                 supplement when supplement_io_type is empty).
    // - supplement_io_type: IO backend for the supplement (y-bit) storage.
    //                       When empty, the supplement falls back to
    //                       base_io_type.
    static std::string
    GenerateBuildParam(const std::string& metric_type,
                       int64_t dim,
                       const std::string& base_io_type,
                       const std::string& supplement_io_type,
                       uint32_t rabitq_filter_bits = 3,
                       uint32_t rabitq_supplement_bits = 5,
                       bool fast_encode_rabitq = true);
};

const std::string HGraphRaBitQSplitTestIndex::name = "hgraph";
fixtures::TempDir HGraphRaBitQSplitTestIndex::dir{"hgraph_rabitq_split"};
TestDatasetPool HGraphRaBitQSplitTestIndex::pool{};

std::string
HGraphRaBitQSplitTestIndex::GenerateBuildParam(const std::string& metric_type,
                                               int64_t dim,
                                               const std::string& base_io_type,
                                               const std::string& supplement_io_type,
                                               uint32_t rabitq_filter_bits,
                                               uint32_t rabitq_supplement_bits,
                                               bool fast_encode_rabitq) {
    constexpr auto temp_with_supplement = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "rabitq",
            "precise_quantization_type": "rabitq",
            "base_io_type": "{}",
            "base_supplement_io_type": "{}",
            "base_file_path": "{}",
            "use_reorder": true,
            "build_by_base": true,
            "rabitq_bits_per_dim_base": {},
            "rabitq_bits_per_dim_precise": {},
            "rabitq_error_rate": 1.9,
            "fast_encode_rabitq": {},
            "max_degree": 32,
            "ef_construction": 200,
            "graph_storage_type": "compressed"
        }}
    }})";
    constexpr auto temp_without_supplement = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "rabitq",
            "precise_quantization_type": "rabitq",
            "base_io_type": "{}",
            "base_file_path": "{}",
            "use_reorder": true,
            "build_by_base": true,
            "rabitq_bits_per_dim_base": {},
            "rabitq_bits_per_dim_precise": {},
            "rabitq_error_rate": 1.9,
            "fast_encode_rabitq": {},
            "max_degree": 32,
            "ef_construction": 200,
            "graph_storage_type": "compressed"
        }}
    }})";
    if (supplement_io_type.empty()) {
        return fmt::format(temp_without_supplement,
                           metric_type,
                           dim,
                           base_io_type,
                           dir.GenerateRandomFile(),
                           rabitq_filter_bits,
                           rabitq_supplement_bits,
                           fast_encode_rabitq);
    }
    return fmt::format(temp_with_supplement,
                       metric_type,
                       dim,
                       base_io_type,
                       supplement_io_type,
                       dir.GenerateRandomFile(),
                       rabitq_filter_bits,
                       rabitq_supplement_bits,
                       fast_encode_rabitq);
}

}  // namespace fixtures

namespace {

class RejectSecondBuildThreadPool final : public vsag::ThreadPool {
public:
    ~RejectSecondBuildThreadPool() override {
        this->WaitUntilEmpty();
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
        const uint64_t submission = submissions_.fetch_add(1, std::memory_order_relaxed);
        if (submission > 0) {
            release_.store(true, std::memory_order_release);
            throw std::runtime_error("injected enqueue failure");
        }

        worker_ = std::thread([this, task = std::move(task)]() mutable {
            while (not release_.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            task();
            task_finished_.store(true, std::memory_order_release);
        });
        return {};
    }

    [[nodiscard]] bool
    TaskFinished() const {
        return task_finished_.load(std::memory_order_acquire);
    }

private:
    std::atomic<uint64_t> submissions_{0};
    std::atomic<bool> release_{false};
    std::atomic<bool> task_finished_{false};
    std::thread worker_{};
};

constexpr const char* kSplitSearchParam = R"(
{
    "hgraph": {
        "ef_search": 200,
        "rabitq_one_bit_search": true
    }
})";

}  // namespace

TEST_CASE("HGraph RaBitQ Split Homogeneous IO", "[ft][rabitq_split][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 600;

    auto metric = GENERATE("l2", "ip", "cosine");
    auto base_io = GENERATE("block_memory_io", "memory_io");

    auto fast_encode = GENERATE(false, true);
    auto store_raw_vector = GENERATE(false, true);
    INFO(fmt::format("metric={}, base_io={}, fast_encode={}, store_raw_vector={}",
                     metric,
                     base_io,
                     fast_encode,
                     store_raw_vector));
    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, base_io, "", 3, 5, fast_encode);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["store_raw_vector"].SetBool(store_raw_vector);
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param_json.Dump(), true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, metric);
    TestIndex::TestBuildIndex(index, dataset, true);
    REQUIRE_NOTHROW(index->GetStats());
    TestIndex::TestKnnSearch(index, dataset, kSplitSearchParam, 0.1F, true);
}

TEST_CASE("HGraph RaBitQ Split ODescent optimized build", "[ft][rabitq_split][hgraph][odescent]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 600;
    const std::string metric = GENERATE("l2", "ip", "cosine");

    auto param = HGraphRaBitQSplitTestIndex::GenerateBuildParam(
        metric, dim, "block_memory_io", "", 3, 5, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_type"].SetString("odescent");
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param_json.Dump(), true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, metric);

    TestIndex::TestBuildIndex(index, dataset, true);
    REQUIRE_NOTHROW(index->GetStats());
    TestIndex::TestKnnSearch(index, dataset, kSplitSearchParam, 0.1F, true);
}

TEST_CASE("HGraph RaBitQ Split drains accepted build tasks after enqueue failure",
          "[ft][rabitq_split][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 64;
    constexpr uint64_t base_count = 32;

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 3, 5, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["build_thread_count"].SetInt(2);

    auto rejecting_pool = std::make_shared<RejectSecondBuildThreadPool>();
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), rejecting_pool);
    vsag::Engine engine(&resource);
    auto index_result = engine.CreateIndex(HGraphRaBitQSplitTestIndex::name, param_json.Dump());
    REQUIRE(index_result.has_value());

    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    auto build_result = index_result.value()->Build(dataset->base_);

    REQUIRE_FALSE(build_result.has_value());
    REQUIRE(rejecting_pool->TaskFinished());
}

TEST_CASE("HGraph RaBitQ Split Build then batched Add", "[ft][rabitq_split][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 240;
    constexpr uint64_t initial_count = base_count / 2;
    const std::string metric = "l2";

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, "memory_io", "", 3, 5, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["store_raw_vector"].SetBool(true);
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param_json.Dump(), true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, metric);

    auto initial = vsag::Dataset::Make();
    initial->Dim(dim)
        ->NumElements(initial_count)
        ->Ids(dataset->base_->GetIds())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);
    auto added = vsag::Dataset::Make();
    added->Dim(dim)
        ->NumElements(base_count - initial_count)
        ->Ids(dataset->base_->GetIds() + initial_count)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors() + initial_count * dim)
        ->Owner(false);

    auto build_result = index->Build(initial);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    auto add_result = index->Add(added);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index->GetNumElements() == base_count);
    REQUIRE_NOTHROW(index->GetStats());
    TestIndex::TestKnnSearch(index, dataset, kSplitSearchParam, 0.1F, true);

    auto reloaded = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    TestIndex::TestSerializeFile(index, reloaded, dataset, kSplitSearchParam, true);
}

TEST_CASE("HGraph RaBitQ Split rejects unsupported non-empty Add", "[ft][rabitq_split][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 64;
    constexpr uint64_t base_count = 64;
    constexpr uint64_t initial_count = base_count / 2;
    const std::string metric = "l2";

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, "memory_io", "", 3, 5, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["use_reorder"].SetBool(false);
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param_json.Dump(), true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, metric);

    auto initial = vsag::Dataset::Make();
    initial->Dim(dim)
        ->NumElements(initial_count)
        ->Ids(dataset->base_->GetIds())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);
    auto added = vsag::Dataset::Make();
    added->Dim(dim)
        ->NumElements(base_count - initial_count)
        ->Ids(dataset->base_->GetIds() + initial_count)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors() + initial_count * dim)
        ->Owner(false);

    auto build_result = index->Build(initial);
    REQUIRE(build_result.has_value());
    auto add_result = index->Add(added);
    REQUIRE_FALSE(add_result.has_value());
    REQUIRE(index->GetNumElements() == initial_count);
}

TEST_CASE("HGraph RaBitQ Split Hybrid IO (memory + async supplement)",
          "[ft][rabitq_split][hybrid][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 600;
    const std::string metric = GENERATE("l2", "ip");

    auto build_param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, "block_memory_io", "async_io");
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, build_param, true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, metric);
    TestIndex::TestBuildIndex(index, dataset, true);
    REQUIRE_NOTHROW(index->GetStats());
    TestIndex::TestKnnSearch(index, dataset, kSplitSearchParam, 0.1F, true);

    // Round-trip serialize / deserialize so the supplement_io_params branch
    // in FlattenDataCellParameter::ToJson and FlattenInterface::MakeInstance
    // is exercised end-to-end.
    auto reload_param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, "block_memory_io", "async_io");
    auto reloaded = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, reload_param, true);
    TestIndex::TestSerializeFile(index, reloaded, dataset, kSplitSearchParam, true);
}

TEST_CASE("HGraph RaBitQ Split Reject Unsupported Hybrid", "[ft][rabitq_split][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    const std::string metric = "l2";

    // Only (block_memory_io one-bit + async_io supplement) is supported as a
    // hybrid combination today; any other heterogeneous pair must fail at
    // index creation time with a clear error from
    // FlattenInterface::MakeInstance.
    auto bad_param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, "memory_io", "async_io");
    auto result = vsag::Factory::CreateIndex(HGraphRaBitQSplitTestIndex::name, bad_param);
    REQUIRE_FALSE(result.has_value());
}
