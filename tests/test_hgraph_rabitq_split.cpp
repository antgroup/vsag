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

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <chrono>
#include <cmath>
#include <future>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "algorithm/hgraph/hgraph.h"
#include "functest.h"
#include "index/index_impl.h"
#include "storage/serialization_tags.h"
#include "storage/streaming_serialization_test_utils.h"
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

class OnlyLabelFilter final : public vsag::Filter {
public:
    explicit OnlyLabelFilter(int64_t label) : label_(label) {
    }

    bool
    CheckValid(int64_t label) const override {
        return label == label_;
    }

    float
    ValidRatio() const override {
        return 1.0F;
    }

private:
    int64_t label_;
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

TEST_CASE("HGraph MRLE RaBitQ Split", "[ft][rabitq_split][hgraph][MRLE]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 600;

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 3, 5, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["base_quantization_type"].SetString("tq");
    param_json["index_param"]["tq_chain"].SetString("mrle, rabitq");
    param_json["index_param"]["mrle_dim"].SetInt(64);

    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param_json.Dump(), true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);
    REQUIRE_NOTHROW(index->GetStats());
    TestIndex::TestKnnSearch(index, dataset, kSplitSearchParam, 0.05F, true);
}

TEST_CASE("HGraph tunes FP32 to MRLE RaBitQ Split", "[ft][rabitq_split][hgraph][MRLE][tune]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 600;

    auto source_param = fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "fp32",
            "use_reorder": false,
            "max_degree": 32,
            "ef_construction": 200,
            "graph_storage_type": "compressed"
        }}
    }})",
                                    dim);
    auto target_param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 3, 5, true);
    auto target_json = vsag::JsonType::Parse(target_param);
    target_json["index_param"]["base_quantization_type"].SetString("tq");
    target_json["index_param"]["tq_chain"].SetString("mrle, rabitq");
    target_json["index_param"]["mrle_dim"].SetInt(64);
    target_param = target_json.Dump();

    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, source_param, true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto tune_result = index->Tune(target_param, true);
    REQUIRE(tune_result.has_value());
    REQUIRE(tune_result.value());
    REQUIRE_NOTHROW(index->GetStats());
    TestIndex::TestKnnSearch(index, dataset, kSplitSearchParam, 0.05F, true);

    auto reloaded = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, target_param, true);
    TestIndex::TestSerializeFile(index, reloaded, dataset, kSplitSearchParam, true);
}

TEST_CASE("HGraph RaBitQ reorder work uses the rerank statistics phase",
          "[ft][rabitq_split][hgraph][statistics]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 200;

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 3, 5, true);
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = get_one_query(dataset->query_, 0);
    auto result = index->KnnSearch(query, 10, kSplitSearchParam);
    REQUIRE(result.has_value());
    auto statistics = vsag::JsonType::Parse(result.value()->GetStatistics());
    const auto lower_bound_probes = statistics["reorder_lower_bound_probe_count"].GetUint64();
    const auto reorder_distances = statistics["reorder_distance_count"].GetUint64();
    const auto rerank = statistics["distance_evaluations_by_phase"]["rerank"].GetUint64();

    // Direct fused traversal can supply lower-bound hints for every candidate, so a search may
    // legitimately need no additional lower-bound probes. It must still perform rerank work, and
    // every distance evaluated by either reorder path must be attributed to the rerank phase.
    REQUIRE(lower_bound_probes + reorder_distances > 0);
    REQUIRE(rerank == lower_bound_probes + reorder_distances);
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

TEST_CASE("HGraph RaBitQ Split validates dimension before optimized training",
          "[ft][rabitq_split][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 64;
    constexpr uint64_t base_count = 32;
    const int64_t invalid_dim = GENERATE(dim - 1, dim + 1);

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 3, 5, true);
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    auto invalid_dataset =
        HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(invalid_dim, base_count, "l2");

    auto invalid_result = index->Build(invalid_dataset->base_);
    REQUIRE_FALSE(invalid_result.has_value());
    REQUIRE(index->GetNumElements() == 0);

    auto valid_dataset =
        HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    auto valid_result = index->Build(valid_dataset->base_);
    REQUIRE(valid_result.has_value());
    REQUIRE(valid_result.value().empty());
    REQUIRE(index->GetNumElements() == base_count);
}

TEST_CASE("HGraph fused RaBitQ rejects ExportModel",
          "[ft][rabitq_split][hgraph][fused][export_model]") {
    using namespace fixtures;
    constexpr int64_t dim = 64;
    constexpr uint64_t base_count = 64;
    const std::string graph_type = GENERATE("odescent", "nsw");
    CAPTURE(graph_type);

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 1, 7, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["graph_type"].SetString(graph_type);
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(false);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["build_thread_count"].SetInt(1);
    param = param_json.Dump();

    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    auto source = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    auto source_build = source->Build(dataset->base_);
    REQUIRE(source_build.has_value());

    auto model_result = source->ExportModel();
    REQUIRE_FALSE(model_result.has_value());
    REQUIRE(model_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
}

TEST_CASE("HGraph fused RaBitQ automatically becomes immutable",
          "[ft][rabitq_split][hgraph][fused][immutable][serialize]") {
    using namespace fixtures;
    constexpr int64_t dim = 64;
    constexpr uint64_t base_count = 64;
    constexpr int64_t topk = 10;

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 1, 7, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(false);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["build_thread_count"].SetInt(1);
    param = param_json.Dump();

    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    auto query = get_one_query(dataset->query_, 0);
    auto require_immutable = [&](const TestIndex::IndexPtr& index) {
        auto index_impl = std::dynamic_pointer_cast<vsag::IndexImpl<vsag::HGraph>>(index);
        REQUIRE(index_impl != nullptr);
        auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index_impl->GetInnerIndex());
        REQUIRE(hgraph != nullptr);
        REQUIRE(hgraph->immutable_.load(std::memory_order_acquire));
        REQUIRE(index->GetMemoryUsageDetail().at("neighbors_mutex") == 0);
        REQUIRE(index->SetImmutable().has_value());

        auto result = index->KnnSearch(query, topk, kSplitSearchParam);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == topk);
    };

    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    REQUIRE(index->Build(dataset->base_).has_value());
    require_immutable(index);

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto binary_restored = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    REQUIRE(binary_restored->Deserialize(binary.value()).has_value());
    require_immutable(binary_restored);

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    auto streaming_restored = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    REQUIRE(streaming_restored->DeserializeStreaming(stream).has_value());
    require_immutable(streaming_restored);
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
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
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

TEST_CASE("HGraph RaBitQ split rejects non-finite one-bit queries",
          "[ft][rabitq_split][hgraph][fused][validation]") {
    using namespace fixtures;
    constexpr int64_t dim = 64;
    constexpr uint64_t base_count = 64;
    constexpr int64_t topk = 10;
    const bool fused = GENERATE(false, true);
    CAPTURE(fused);

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 1, 7, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(fused);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(false);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["build_thread_count"].SetInt(1);

    auto source = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param_json.Dump(), true);
    auto build_result = index->Build(source->base_);
    REQUIRE(build_result.has_value());

    const float invalid_values[] = {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()};
    for (const float invalid_value : invalid_values) {
        CAPTURE(invalid_value);
        std::vector<float> query_vector(source->query_->GetFloat32Vectors(),
                                        source->query_->GetFloat32Vectors() + dim);
        query_vector[0] = invalid_value;
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);
        auto result = index->KnnSearch(query, topk, kSplitSearchParam);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    if (fused) {
        uint64_t scored_count = 0;
        uint64_t largest_batch = 0;
        const auto score = [](int64_t label) { return static_cast<float>(label % 1000000); };
        vsag::SearchRequest request;
        request.topk_ = topk;
        request.params_str_ = fmt::format(
            R"({{"hgraph":{{"ef_search":{},"rabitq_one_bit_search":true}}}})", base_count);
        request.distance_batch_size_ = 3;
        request.distance_batch_func_ =
            [&](const int64_t* labels, uint64_t count, float* distances) {
                scored_count += count;
                largest_batch = std::max(largest_batch, count);
                for (uint64_t i = 0; i < count; ++i) {
                    distances[i] = score(labels[i]);
                }
            };
        auto result = index->SearchWithRequest(request);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == topk);
        REQUIRE(scored_count > 0);
        REQUIRE(largest_batch <= request.distance_batch_size_);
        for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
            REQUIRE(result.value()->GetDistances()[i] == score(result.value()->GetIds()[i]));
        }
    }
}

TEST_CASE("HGraph fused RaBitQ split rejects non-finite base vectors",
          "[ft][rabitq_split][hgraph][fused][validation]") {
    using namespace fixtures;
    constexpr int64_t dim = 64;
    constexpr uint64_t base_count = 64;

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 1, 7, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["graph_type"].SetString("odescent");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(true);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["build_thread_count"].SetInt(1);
    param = param_json.Dump();

    auto source = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    const float invalid_values[] = {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()};
    for (const float invalid_value : invalid_values) {
        CAPTURE(invalid_value);
        std::vector<float> invalid_vectors(
            source->base_->GetFloat32Vectors(),
            source->base_->GetFloat32Vectors() + base_count * static_cast<uint64_t>(dim));
        invalid_vectors[0] = invalid_value;
        auto invalid_base = vsag::Dataset::Make();
        invalid_base->NumElements(base_count)
            ->Dim(dim)
            ->Ids(source->base_->GetIds())
            ->Float32Vectors(invalid_vectors.data())
            ->Owner(false);
        auto invalid_index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
        auto build_result = invalid_index->Build(invalid_base);
        REQUIRE_FALSE(build_result.has_value());
        REQUIRE(build_result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
        REQUIRE(invalid_index->GetNumElements() == 0);
    }

    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    auto build_result = index->Build(source->base_);
    REQUIRE(build_result.has_value());
    REQUIRE(index->GetNumElements() == base_count);

    const auto* base_ids = source->base_->GetIds();
    const int64_t existing_label = base_ids[0];
    const auto* existing_vector = source->base_->GetFloat32Vectors();
    auto original_distance = index->CalcDistanceById(existing_vector, existing_label);
    REQUIRE(original_distance.has_value());
    const int64_t max_label = *std::max_element(base_ids, base_ids + base_count);
    uint64_t invalid_value_index = 0;
    for (const float invalid_value : invalid_values) {
        CAPTURE(invalid_value_index);
        std::vector<float> invalid_vector(existing_vector, existing_vector + dim);
        invalid_vector[0] = invalid_value;

        const int64_t added_label = max_label + static_cast<int64_t>(invalid_value_index) + 1;
        auto invalid_add = vsag::Dataset::Make();
        invalid_add->NumElements(1)
            ->Dim(dim)
            ->Ids(&added_label)
            ->Float32Vectors(invalid_vector.data())
            ->Owner(false);
        const uint64_t count_before_add = index->GetNumElements();
        auto add_result = index->Add(invalid_add);
        REQUIRE_FALSE(add_result.has_value());
        REQUIRE(add_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
        REQUIRE(index->GetNumElements() == count_before_add);
        REQUIRE_FALSE(index->CheckIdExist(added_label));

        auto invalid_update = vsag::Dataset::Make();
        invalid_update->NumElements(1)
            ->Dim(dim)
            ->Ids(&existing_label)
            ->Float32Vectors(invalid_vector.data())
            ->Owner(false);
        auto update_result = index->UpdateVector(existing_label, invalid_update, true);
        REQUIRE_FALSE(update_result.has_value());
        REQUIRE(update_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
        REQUIRE(index->GetNumElements() == base_count);
        auto unchanged_distance = index->CalcDistanceById(existing_vector, existing_label);
        REQUIRE(unchanged_distance.has_value());
        REQUIRE(unchanged_distance.value() == original_distance.value());
        ++invalid_value_index;
    }

    std::vector<float> overflowing_vector(static_cast<uint64_t>(dim),
                                          std::numeric_limits<float>::max());
    const int64_t overflowing_label = max_label + 100;
    auto overflowing_add = vsag::Dataset::Make();
    overflowing_add->NumElements(1)
        ->Dim(dim)
        ->Ids(&overflowing_label)
        ->Float32Vectors(overflowing_vector.data())
        ->Owner(false);
    const uint64_t count_before_overflow = index->GetNumElements();
    auto overflow_result = index->Add(overflowing_add);
    REQUIRE_FALSE(overflow_result.has_value());
    REQUIRE(overflow_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
    REQUIRE(index->GetNumElements() == count_before_overflow);
    REQUIRE_FALSE(index->CheckIdExist(overflowing_label));

    std::vector<float> short_vector(existing_vector, existing_vector + dim - 1);
    auto wrong_dim_update = vsag::Dataset::Make();
    wrong_dim_update->NumElements(1)
        ->Dim(dim - 1)
        ->Ids(&existing_label)
        ->Float32Vectors(short_vector.data())
        ->Owner(false);
    auto wrong_dim_result = index->UpdateVector(existing_label, wrong_dim_update, true);
    REQUIRE_FALSE(wrong_dim_result.has_value());
    REQUIRE(wrong_dim_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
    auto unchanged_distance = index->CalcDistanceById(existing_vector, existing_label);
    REQUIRE(unchanged_distance.has_value());
    REQUIRE(unchanged_distance.value() == original_distance.value());
}

TEST_CASE("HGraph fused RaBitQ split compact round trip",
          "[ft][rabitq_split][hgraph][fused][serialize]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 1024;
    constexpr int64_t topk = 10;
    constexpr uint64_t max_degree = 32;
    const uint32_t filter_bits = GENERATE(1U, 2U, 3U, 4U);
    const uint32_t supplement_bits = 8U - filter_bits;
    INFO(fmt::format("fused split {}+{}", filter_bits, supplement_bits));

    auto param = HGraphRaBitQSplitTestIndex::GenerateBuildParam(
        "l2", dim, "memory_io", "", filter_bits, supplement_bits, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["graph_type"].SetString("odescent");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(false);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["build_thread_count"].SetInt(4);
    param = param_json.Dump();

    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);
    REQUIRE(index->GetNumElements() == base_count);

    const auto* base_ids = dataset->base_->GetIds();
    const int64_t old_label = base_ids[base_count - 1];
    const int64_t new_label =
        *std::max_element(base_ids, base_ids + base_count) + static_cast<int64_t>(base_count) + 1;
    auto update = index->UpdateId(old_label, new_label);
    REQUIRE_FALSE(update.has_value());
    REQUIRE(update.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
    REQUIRE(index->CheckIdExist(old_label));
    REQUIRE_FALSE(index->CheckIdExist(new_label));

    const float recall = TestIndex::TestKnnSearch(index, dataset, kSplitSearchParam, 0.5F, true);
    REQUIRE(recall > 0.5F);

    auto query = get_one_query(dataset->query_, 0);
    auto expected = index->KnnSearch(query, topk, kSplitSearchParam);
    REQUIRE(expected.has_value());
    REQUIRE(expected.value()->GetDim() == topk);
    if (filter_bits == 1) {
        const auto stats = expected.value()->GetStatistics({"rabitq_full_count",
                                                            "rabitq_reorder_hint_full_count",
                                                            "rabitq_reorder_fallback_full_count"});
        REQUIRE(stats.size() == 3);
        REQUIRE(std::stoull(stats[0]) > 0);
        REQUIRE(std::stoull(stats[1]) == 0);
        // Legacy 1+7 traversal uses an approximate 4-bit query projection, so exact rerank must
        // recompute the full distance instead of treating that estimate as an exact hint.
        REQUIRE(std::stoull(stats[2]) > 0);
    }
    const std::vector<int64_t> expected_ids(expected.value()->GetIds(),
                                            expected.value()->GetIds() + topk);
    const std::vector<float> expected_distances(expected.value()->GetDistances(),
                                                expected.value()->GetDistances() + topk);

    auto require_same_result = [&](const TestIndex::IndexPtr& restored) {
        REQUIRE(restored->GetNumElements() == base_count);
        REQUIRE(restored->CheckIdExist(old_label));
        REQUIRE_FALSE(restored->CheckIdExist(new_label));
        auto actual = restored->KnnSearch(query, topk, kSplitSearchParam);
        REQUIRE(actual.has_value());
        REQUIRE(actual.value()->GetDim() == topk);
        for (int64_t i = 0; i < topk; ++i) {
            REQUIRE(actual.value()->GetIds()[i] == expected_ids[static_cast<uint64_t>(i)]);
            REQUIRE(std::abs(actual.value()->GetDistances()[i] -
                             expected_distances[static_cast<uint64_t>(i)]) <= 2e-6F);
        }
    };

    auto ordinary_result = index->Serialize();
    REQUIRE(ordinary_result.has_value());
    uint64_t ordinary_size = 0;
    for (const auto& key : ordinary_result.value().GetKeys()) {
        ordinary_size += ordinary_result.value().Get(key).size;
    }
    auto ordinary_restored = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    REQUIRE(ordinary_restored->Deserialize(ordinary_result.value()).has_value());
    require_same_result(ordinary_restored);

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    const auto bytes = stream.str();
    const auto base_codes =
        vsag::test::FindStreamingBlock(bytes, vsag::StreamSerializationTag::BASE_CODES);
    const auto bottom_graph =
        vsag::test::FindStreamingBlock(bytes, vsag::StreamSerializationTag::BOTTOM_GRAPH);

    // Fused BASE_CODES contains only the codec model. A second per-node split-code copy would
    // require at least dim * (x + y) / 8 bytes per vector and violate this bound.
    const uint64_t duplicated_split_bytes = base_count * static_cast<uint64_t>(dim);
    REQUIRE(base_codes.payload_size < duplicated_split_bytes);

    // For dim=128 and M=32, a fused node record is at most 384 bytes: links, metadata, one
    // byte/dimension of split codes, and cache-line padding. Allow 64 KiB for graph headers and
    // the shared codec model, but no second count-scaled code array.
    constexpr uint64_t fused_record_upper_bound =
        max_degree * sizeof(vsag::InnerIdType) + static_cast<uint64_t>(dim) + 2 * 64;
    constexpr uint64_t model_and_header_allowance = 64 * 1024;
    constexpr uint64_t container_allowance = 256 * 1024;
    REQUIRE(bottom_graph.payload_size <=
            base_count * fused_record_upper_bound + model_and_header_allowance);
    REQUIRE(ordinary_size <= base_count * fused_record_upper_bound + container_allowance);
    REQUIRE(bytes.size() <= base_count * fused_record_upper_bound + container_allowance);

    auto streaming_restored = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    std::stringstream deserialize_stream(bytes);
    REQUIRE(streaming_restored->DeserializeStreaming(deserialize_stream).has_value());
    require_same_result(streaming_restored);
}

TEST_CASE("HGraph fused RaBitQ split trailing duplicate label round trip",
          "[ft][rabitq_split][hgraph][fused][duplicate][serialize]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 64;
    constexpr int64_t topk = 10;

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 1, 7, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(false);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["support_duplicate"].SetBool(true);
    param_json["index_param"]["build_thread_count"].SetInt(4);
    param = param_json.Dump();

    auto source = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    std::vector<int64_t> labels(source->base_->GetIds(), source->base_->GetIds() + base_count);
    const int64_t skipped_label = labels.back();
    const int64_t duplicate_label = labels[base_count - 2];
    labels.back() = duplicate_label;
    auto base = vsag::Dataset::Make();
    base->NumElements(base_count)
        ->Dim(dim)
        ->Ids(labels.data())
        ->Float32Vectors(source->base_->GetFloat32Vectors())
        ->Owner(false);

    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value() == std::vector<int64_t>{duplicate_label});
    const uint64_t graph_count = index->GetNumElements();
    REQUIRE(graph_count == base_count - 1);
    REQUIRE(graph_count <= base_count);
    REQUIRE(index->CheckIdExist(duplicate_label));
    REQUIRE_FALSE(index->CheckIdExist(skipped_label));

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dim)
        ->Float32Vectors(source->base_->GetFloat32Vectors() + (base_count - 2) * dim)
        ->Owner(false);
    auto expected = index->KnnSearch(query, topk, kSplitSearchParam);
    REQUIRE(expected.has_value());
    REQUIRE(expected.value()->GetDim() == topk);
    REQUIRE(expected.value()->GetIds()[0] == duplicate_label);

    auto require_same_semantics = [&](const TestIndex::IndexPtr& restored) {
        REQUIRE(restored->GetNumElements() == graph_count);
        REQUIRE(restored->CheckIdExist(duplicate_label));
        REQUIRE_FALSE(restored->CheckIdExist(skipped_label));
        auto actual = restored->KnnSearch(query, topk, kSplitSearchParam);
        REQUIRE(actual.has_value());
        REQUIRE(actual.value()->GetDim() == topk);
        REQUIRE(actual.value()->GetIds()[0] == duplicate_label);
        for (int64_t i = 0; i < topk; ++i) {
            REQUIRE(actual.value()->GetIds()[i] == expected.value()->GetIds()[i]);
            REQUIRE(std::abs(actual.value()->GetDistances()[i] -
                             expected.value()->GetDistances()[i]) <= 2e-6F);
        }
    };

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto ordinary_restored = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    REQUIRE(ordinary_restored->Deserialize(binary.value()).has_value());
    require_same_semantics(ordinary_restored);

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    auto streaming_restored = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    std::stringstream deserialize_stream(stream.str());
    REQUIRE(streaming_restored->DeserializeStreaming(deserialize_stream).has_value());
    require_same_semantics(streaming_restored);
}

TEST_CASE("HGraph fused RaBitQ split preserves trailing vector aliases",
          "[ft][rabitq_split][hgraph][fused][duplicate][serialize]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 64;
    constexpr int64_t duplicate_count = 3;

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 1, 7, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(false);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["support_duplicate"].SetBool(true);
    param_json["index_param"]["build_thread_count"].SetInt(1);
    param = param_json.Dump();

    auto source = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    std::vector<float> vectors(source->base_->GetFloat32Vectors(),
                               source->base_->GetFloat32Vectors() + base_count * dim);
    std::copy_n(vectors.data(), dim, vectors.data() + (base_count - 2) * dim);
    std::copy_n(vectors.data(), dim, vectors.data() + (base_count - 1) * dim);
    std::vector<int64_t> labels(source->base_->GetIds(), source->base_->GetIds() + base_count);
    const std::vector<int64_t> duplicate_labels = {
        labels.front(), labels[base_count - 2], labels.back()};
    auto base = vsag::Dataset::Make();
    base->NumElements(base_count)
        ->Dim(dim)
        ->Ids(labels.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index->GetNumElements() == base_count);

    auto require_duplicate_semantics = [&](const TestIndex::IndexPtr& restored,
                                           uint32_t ef_search) {
        const auto search_param = fmt::format(
            R"({{"hgraph":{{"ef_search":{},"rabitq_one_bit_search":true}}}})", ef_search);
        auto result = restored->KnnSearch(query, duplicate_count, search_param);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == duplicate_count);
        std::vector<int64_t> result_labels(result.value()->GetIds(),
                                           result.value()->GetIds() + duplicate_count);
        for (const auto label : duplicate_labels) {
            REQUIRE(std::find(result_labels.begin(), result_labels.end(), label) !=
                    result_labels.end());
        }

        auto alias_only = std::make_shared<OnlyLabelFilter>(duplicate_labels.back());
        auto filtered = restored->KnnSearch(query, 1, search_param, alias_only);
        REQUIRE(filtered.has_value());
        REQUIRE(filtered.value()->GetDim() == 1);
        REQUIRE(filtered.value()->GetIds()[0] == duplicate_labels.back());

        auto alias_distance = restored->CalcDistanceById(vectors.data(), duplicate_labels.back());
        REQUIRE(alias_distance.has_value());
        REQUIRE(std::isfinite(alias_distance.value()));
        const float distance_tolerance = 2e-5F * std::max(1.0F, std::fabs(alias_distance.value()));
        for (int64_t i = 0; i < duplicate_count; ++i) {
            REQUIRE(std::fabs(result.value()->GetDistances()[i] - alias_distance.value()) <=
                    distance_tolerance);
        }
        REQUIRE(std::fabs(filtered.value()->GetDistances()[0] - alias_distance.value()) <=
                distance_tolerance);

        vsag::IteratorContext* iterator_context = nullptr;
        vsag::FilterPtr iterator_filter = nullptr;
        auto iterator_result = restored->KnnSearch(
            query, duplicate_count, search_param, iterator_filter, iterator_context, false);
        REQUIRE(iterator_result.has_value());
        REQUIRE(iterator_result.value()->GetDim() == duplicate_count);
        std::vector<int64_t> iterator_labels(iterator_result.value()->GetIds(),
                                             iterator_result.value()->GetIds() + duplicate_count);
        for (const auto label : duplicate_labels) {
            REQUIRE(std::find(iterator_labels.begin(), iterator_labels.end(), label) !=
                    iterator_labels.end());
        }
        delete iterator_context;
    };

    require_duplicate_semantics(index, 20);
    require_duplicate_semantics(index, 80);

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto ordinary_restored = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    REQUIRE(ordinary_restored->Deserialize(binary.value()).has_value());
    require_duplicate_semantics(ordinary_restored, 20);

    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    auto streaming_restored = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    std::stringstream deserialize_stream(stream.str());
    REQUIRE(streaming_restored->DeserializeStreaming(deserialize_stream).has_value());
    require_duplicate_semantics(streaming_restored, 80);
}

TEST_CASE("HGraph fused RaBitQ split honors disabled reorder",
          "[ft][rabitq_split][hgraph][fused][search][optimized_build]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 128;
    constexpr int64_t topk = 10;
    const uint32_t filter_bits = GENERATE(1U, 2U, 3U, 4U);
    const bool support_duplicate = GENERATE(false, true);
    const bool store_raw_vector = GENERATE(false, true);

    auto param = HGraphRaBitQSplitTestIndex::GenerateBuildParam(
        "l2", dim, "memory_io", "", filter_bits, 8U - filter_bits, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(store_raw_vector);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["support_duplicate"].SetBool(support_duplicate);
    param_json["index_param"]["build_thread_count"].SetInt(4);

    auto source = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, "l2");
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param_json.Dump(), true);
    auto build_result = index->Build(source->base_);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    auto query = get_one_query(source->query_, 0);

    for (const uint32_t ef_search : {20U, 80U}) {
        for (const uint32_t parallelism : {1U, 2U}) {
            CAPTURE(filter_bits, support_duplicate, store_raw_vector, ef_search, parallelism);
            const auto search_param = fmt::format(
                R"({{
                    "hgraph": {{
                        "ef_search": {},
                        "rabitq_one_bit_search": true,
                        "enable_reorder": false,
                        "parallelism": {}
                    }}
                }})",
                ef_search,
                parallelism);
            auto result = index->KnnSearch(query, topk, search_param);
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == topk);
            for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
                REQUIRE(std::isfinite(result.value()->GetDistances()[i]));
                REQUIRE(result.value()->GetDistances()[i] < std::numeric_limits<float>::max());
            }
            const auto stats = result.value()->GetStatistics({"rabitq_filter_count",
                                                              "rabitq_full_count",
                                                              "rabitq_filter_fallback_full_count",
                                                              "rabitq_reorder_hint_full_count",
                                                              "rabitq_reorder_fallback_full_count",
                                                              "reorder_distance_count"});
            REQUIRE(stats.size() == 6);
            REQUIRE(std::stoull(stats[0]) > 0);
            REQUIRE(std::stoull(stats[1]) == 0);
            REQUIRE(std::stoull(stats[2]) == 0);
            REQUIRE(std::stoull(stats[3]) == 0);
            REQUIRE(std::stoull(stats[4]) == 0);
            REQUIRE(std::stoull(stats[5]) == 0);
        }
    }
}

TEST_CASE("HGraph fused RaBitQ split reuses full distances across deferred finalize",
          "[ft][rabitq_split][hgraph][fused][search][full_hint]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 512;
    constexpr int64_t topk = 10;
    constexpr uint64_t query_count = 8;
    const uint32_t filter_bits = GENERATE(1U, 3U);
    const std::string metric_type = GENERATE("l2", "ip");
    const uint32_t supplement_bits = filter_bits == 1U ? 3U : 5U;

    auto param = HGraphRaBitQSplitTestIndex::GenerateBuildParam(
        metric_type, dim, "memory_io", "", filter_bits, supplement_bits, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(false);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["support_duplicate"].SetBool(false);
    param_json["index_param"]["build_thread_count"].SetInt(1);

    auto source =
        HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, metric_type);
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param_json.Dump(), true);
    auto build_result = index->Build(source->base_);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    for (const uint32_t ef_search : {20U, 40U, 80U}) {
        uint64_t full_count = 0;
        uint64_t hint_full_count = 0;
        uint64_t fallback_full_count = 0;
        uint64_t reorder_distance_count = 0;
        for (uint64_t query_index = 0; query_index < query_count; ++query_index) {
            CAPTURE(filter_bits, supplement_bits, metric_type, ef_search, query_index);
            const auto search_param = fmt::format(
                R"({{
                    "hgraph": {{
                        "ef_search": {},
                        "rabitq_one_bit_search": true,
                        "enable_reorder": true,
                        "parallelism": 1
                    }}
                }})",
                ef_search);
            auto query = get_one_query(source->query_, query_index);
            auto result = index->KnnSearch(query, topk, search_param);
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == topk);
            const auto stats = result.value()->GetStatistics({"rabitq_full_count",
                                                              "rabitq_reorder_hint_full_count",
                                                              "rabitq_reorder_fallback_full_count",
                                                              "reorder_distance_count"});
            REQUIRE(stats.size() == 4);
            full_count += std::stoull(stats[0]);
            hint_full_count += std::stoull(stats[1]);
            fallback_full_count += std::stoull(stats[2]);
            reorder_distance_count += std::stoull(stats[3]);
        }

        REQUIRE(full_count > 0);
        REQUIRE(hint_full_count + fallback_full_count == full_count);
        if (filter_bits == 1U) {
            REQUIRE(hint_full_count == 0);
            REQUIRE(fallback_full_count == full_count);
        } else {
            REQUIRE(hint_full_count == full_count);
            REQUIRE(fallback_full_count == 0);
        }
        if (ef_search <= 40U) {
            REQUIRE(reorder_distance_count < full_count);
        } else {
            REQUIRE(reorder_distance_count == 0);
        }
    }
}

TEST_CASE("HGraph fused RaBitQ split keeps zero-residual nodes without reorder",
          "[ft][rabitq_split][hgraph][fused][fused_zero_residual]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 64;
    constexpr int64_t topk = 10;
    const uint32_t filter_bits = GENERATE(1U, 2U, 3U, 4U);

    auto param = HGraphRaBitQSplitTestIndex::GenerateBuildParam(
        "l2", dim, "memory_io", "", filter_bits, 3U, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(false);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["support_duplicate"].SetBool(false);
    param_json["index_param"]["build_thread_count"].SetInt(1);

    std::vector<float> vectors(base_count * static_cast<uint64_t>(dim));
    for (int64_t d = 0; d < dim; ++d) {
        const float value = static_cast<float>((d % 17) - 8) * 0.125F;
        for (uint64_t row = 0; row < base_count; ++row) {
            vectors[row * static_cast<uint64_t>(dim) + static_cast<uint64_t>(d)] = value;
        }
    }
    std::vector<int64_t> labels(base_count);
    std::iota(labels.begin(), labels.end(), 0);
    auto base = vsag::Dataset::Make();
    base->NumElements(base_count)
        ->Dim(dim)
        ->Ids(labels.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param_json.Dump(), true);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    const auto search_param = R"({
        "hgraph": {
            "ef_search": 20,
            "rabitq_one_bit_search": true,
            "enable_reorder": false,
            "parallelism": 1
        }
    })";
    auto result = index->KnnSearch(query, topk, search_param);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == topk);
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        REQUIRE(std::isfinite(result.value()->GetDistances()[i]));
        REQUIRE(result.value()->GetDistances()[i] < std::numeric_limits<float>::max());
    }
    const auto stats = result.value()->GetStatistics({"rabitq_filter_count",
                                                      "rabitq_full_count",
                                                      "rabitq_filter_fallback_full_count",
                                                      "reorder_distance_count"});
    REQUIRE(stats.size() == 4);
    REQUIRE(std::stoull(stats[0]) > 0);
    REQUIRE(std::stoull(stats[1]) == 0);
    REQUIRE(std::stoull(stats[2]) == 0);
    REQUIRE(std::stoull(stats[3]) == 0);
}

TEST_CASE("HGraph fused RaBitQ split expands a sole representative and its aliases",
          "[ft][rabitq_split][hgraph][fused][duplicate][search]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 3;

    auto param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam("l2", dim, "memory_io", "", 2, 6, true);
    auto param_json = vsag::JsonType::Parse(param);
    param_json["index_param"]["graph_io_type"].SetString("memory_io");
    param_json["index_param"]["graph_storage_type"].SetString("flat");
    param_json["index_param"]["reorder_source"].SetString("base");
    param_json["index_param"]["rabitq_fused_datacell"].SetBool(true);
    param_json["index_param"]["rabitq_use_fht"].SetBool(true);
    param_json["index_param"]["store_raw_vector"].SetBool(false);
    param_json["index_param"]["use_mci"].SetBool(false);
    param_json["index_param"]["support_duplicate"].SetBool(true);
    param_json["index_param"]["duplicate_distance_threshold"].SetFloat(1.0F);
    param_json["index_param"]["build_thread_count"].SetInt(1);
    param = param_json.Dump();

    std::vector<float> vectors(base_count * dim);
    for (int64_t d = 0; d < dim; ++d) {
        vectors[d] = static_cast<float>((d * 17) % 29) * 0.125F;
    }
    for (int64_t d = 0; d < dim; ++d) {
        vectors[dim + d] = vectors[d] + static_cast<float>((d % 3) + 1) * 0.005F;
        vectors[2 * dim + d] = vectors[d] - static_cast<float>((d % 5) + 1) * 0.008F;
    }
    std::vector<int64_t> labels = {100, 200, 300};
    auto base = vsag::Dataset::Make();
    base->NumElements(base_count)
        ->Dim(dim)
        ->Ids(labels.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index->GetNumElements() == base_count);

    const auto stats = vsag::JsonType::Parse(index->GetStats());
    REQUIRE(stats["duplicate_ratio"].GetFloat() > 0.6F);

    std::vector<float> expected_distances;
    expected_distances.reserve(base_count);
    for (const auto label : labels) {
        auto distance = index->CalcDistanceById(vectors.data(), label);
        REQUIRE(distance.has_value());
        REQUIRE(std::isfinite(distance.value()));
        expected_distances.push_back(distance.value());
    }
    REQUIRE(std::fabs(expected_distances.front() - expected_distances.back()) > 1e-4F);
    const auto expected_distance = [&](int64_t label) {
        const auto found = std::find(labels.begin(), labels.end(), label);
        return expected_distances[static_cast<uint64_t>(std::distance(labels.begin(), found))];
    };

    auto require_result = [&](const vsag::DatasetPtr& result,
                              const std::vector<int64_t>& expected_labels) {
        REQUIRE(result != nullptr);
        REQUIRE(result->GetDim() == static_cast<int64_t>(expected_labels.size()));
        std::vector<int64_t> actual_labels(result->GetIds(), result->GetIds() + result->GetDim());
        std::sort(actual_labels.begin(), actual_labels.end());
        auto sorted_expected = expected_labels;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        REQUIRE(actual_labels == sorted_expected);
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            const auto distance = expected_distance(result->GetIds()[i]);
            const float tolerance = 2e-5F * std::max(1.0F, std::fabs(distance));
            REQUIRE(std::fabs(result->GetDistances()[i] - distance) <= tolerance);
        }
    };

    const auto single_search_param = R"({"hgraph":{"ef_search":3,"rabitq_one_bit_search":true}})";
    const auto parallel_search_param = R"({
        "hgraph":{
            "ef_search":3,
            "rabitq_one_bit_search":true,
            "parallelism":2
        }
    })";
    auto alias_only = std::make_shared<OnlyLabelFilter>(labels.back());

    for (const auto* search_param : {single_search_param, parallel_search_param}) {
        auto knn = index->KnnSearch(query, base_count, search_param);
        REQUIRE(knn.has_value());
        require_result(knn.value(), labels);

        auto filtered_knn = index->KnnSearch(query, 1, search_param, alias_only);
        REQUIRE(filtered_knn.has_value());
        require_result(filtered_knn.value(), {labels.back()});

        const auto max_distance =
            *std::max_element(expected_distances.begin(), expected_distances.end());
        const float radius = max_distance + 2e-5F * std::max(1.0F, std::fabs(max_distance));
        auto range = index->RangeSearch(query, radius, search_param);
        REQUIRE(range.has_value());
        require_result(range.value(), labels);

        auto filtered_range = index->RangeSearch(query, radius, search_param, alias_only);
        REQUIRE(filtered_range.has_value());
        require_result(filtered_range.value(), {labels.back()});
    }

    for (const bool one_bit_search : {false, true}) {
        const auto iterator_search_param = fmt::format(
            R"({{"hgraph":{{"ef_search":3,"rabitq_one_bit_search":{}}}}})", one_bit_search);
        vsag::IteratorContext* iterator_context = nullptr;
        vsag::FilterPtr iterator_filter = nullptr;
        std::vector<int64_t> iterator_labels;
        for (uint64_t i = 0; i < base_count; ++i) {
            auto page = index->KnnSearch(query,
                                         1,
                                         iterator_search_param,
                                         iterator_filter,
                                         iterator_context,
                                         i + 1 == base_count);
            REQUIRE(page.has_value());
            REQUIRE(page.value()->GetDim() == 1);
            iterator_labels.push_back(page.value()->GetIds()[0]);
            const auto distance = expected_distance(page.value()->GetIds()[0]);
            const float tolerance = 2e-5F * std::max(1.0F, std::fabs(distance));
            REQUIRE(std::fabs(page.value()->GetDistances()[0] - distance) <= tolerance);
        }
        std::sort(iterator_labels.begin(), iterator_labels.end());
        REQUIRE(iterator_labels == labels);
        delete iterator_context;

        iterator_context = nullptr;
        auto filtered_page =
            index->KnnSearch(query, 1, iterator_search_param, alias_only, iterator_context, false);
        REQUIRE(filtered_page.has_value());
        require_result(filtered_page.value(), {labels.back()});
        delete iterator_context;
    }
}
