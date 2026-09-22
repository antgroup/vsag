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

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "algorithm/mci/mci_builder.h"
#include "algorithm/mci/mci_runner.h"
#include "datacell/clique_datacell.h"
#include "impl/allocator/default_allocator.h"
#include "unittest.h"
#include "vsag/bitset.h"
#include "vsag/dataset.h"
#include "vsag/factory.h"
#include "vsag/filter.h"

namespace {

class HalfRatioAllValidFilter : public vsag::Filter {
public:
    explicit HalfRatioAllValidFilter(const std::vector<int64_t>& ids) : ids_(ids) {
    }

    bool
    CheckValid(int64_t id) const override {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

    float
    ValidRatio() const override {
        return 0.5F;
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = ids_.data();
        count = static_cast<int64_t>(ids_.size());
    }

private:
    std::vector<int64_t> ids_;
};

class CountingValidIdsFilter : public HalfRatioAllValidFilter {
public:
    explicit CountingValidIdsFilter(const std::vector<int64_t>& ids)
        : HalfRatioAllValidFilter(ids) {
    }

    bool
    CheckValid(int64_t id) const override {
        check_count_.fetch_add(1, std::memory_order_relaxed);
        return HalfRatioAllValidFilter::CheckValid(id);
    }

    [[nodiscard]] uint64_t
    CheckCount() const {
        return check_count_.load(std::memory_order_relaxed);
    }

private:
    mutable std::atomic<uint64_t> check_count_{0};
};

class CallbackOnlyFilter : public vsag::Filter {
public:
    bool
    CheckValid(int64_t id) const override {
        return id >= 0;
    }

    float
    ValidRatio() const override {
        return 0.5F;
    }
};

// Keeps the `valid_count` labels that start at `first_label` and, when requested, also exposes a
// dense bitmap whose valid inner ids are given by `bitmap_valid_ids`. The two sets are only the same
// object when labels and inner ids coincide.
class RangeBitmapFilter : public vsag::Filter {
public:
    RangeBitmapFilter(int64_t first_label,
                      int64_t total,
                      int64_t valid_count,
                      const std::vector<int64_t>& bitmap_valid_ids,
                      bool expose_bitmap)
        : first_label_(first_label),
          valid_count_(valid_count),
          bitmap_(static_cast<uint64_t>(total), 0),
          expose_bitmap_(expose_bitmap) {
        valid_ids_.resize(static_cast<uint64_t>(valid_count));
        std::iota(valid_ids_.begin(), valid_ids_.end(), first_label_);
        for (auto id : bitmap_valid_ids) {
            if (id >= 0 and id < total) {
                bitmap_[static_cast<uint64_t>(id)] = 1;
            }
        }
    }

    bool
    CheckValid(int64_t id) const override {
        return id >= first_label_ and id < first_label_ + valid_count_;
    }

    float
    ValidRatio() const override {
        return static_cast<float>(valid_count_) / static_cast<float>(bitmap_.size());
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = valid_ids_.data();
        count = static_cast<int64_t>(valid_ids_.size());
    }

    [[nodiscard]] const uint8_t*
    GetValidBitmap(uint64_t* size) const override {
        if (not expose_bitmap_) {
            if (size != nullptr) {
                *size = 0;
            }
            return nullptr;
        }
        if (size != nullptr) {
            *size = bitmap_.size();
        }
        return bitmap_.data();
    }

private:
    int64_t first_label_{0};
    int64_t valid_count_{0};
    std::vector<int64_t> valid_ids_;
    std::vector<uint8_t> bitmap_;
    bool expose_bitmap_{false};
};

struct TestEdge {
    uint32_t u{0};
    uint32_t v{0};
};

class WorkerFailAllocator : public vsag::Allocator {
public:
    WorkerFailAllocator() : owner_(std::this_thread::get_id()) {
    }

    std::string
    Name() override {
        return "worker-fail-allocator";
    }

    void*
    Allocate(uint64_t size) override {
        if (std::this_thread::get_id() != owner_) {
            throw std::bad_alloc();
        }
        auto* result = std::malloc(size);
        if (result == nullptr) {
            throw std::bad_alloc();
        }
        return result;
    }

    void
    Deallocate(void* pointer) override {
        std::free(pointer);
    }

    void*
    Reallocate(void* pointer, uint64_t size) override {
        if (std::this_thread::get_id() != owner_) {
            throw std::bad_alloc();
        }
        auto* result = std::realloc(pointer, size);
        if (result == nullptr) {
            throw std::bad_alloc();
        }
        return result;
    }

private:
    std::thread::id owner_;
};

std::string
generate_hgraph_mci_params(int64_t dim,
                           const std::string& knng_source = vsag::HGRAPH_MCI_KNNG_SOURCE_HGRAPH) {
    auto params = vsag::JsonType::Parse(R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {
                "base_quantization_type": "fp32",
                "graph_type": "odescent",
                "max_degree": 6,
                "alpha": 1.2,
                "graph_iter_turn": 6,
                "neighbor_sample_rate": 0.3,
                "mci_mcs": 8,
                "mci_clique_max": 4,
                "mci_incremental_clique_max": 4,
                "mci_alpha": 1.2
            }
        }
    )");
    params["dim"].SetInt(dim);
    params["index_param"][vsag::HGRAPH_MCI_KNNG_SOURCE].SetString(knng_source);
    return params.Dump();
}

vsag::DatasetPtr
make_dataset(std::vector<int64_t>& ids,
             std::vector<float>& vectors,
             int64_t offset,
             int64_t count,
             int64_t dim) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data() + offset)
        ->Float32Vectors(vectors.data() + offset * dim)
        ->Owner(false);
    return dataset;
}

}  // namespace

TEST_CASE("HGraph companion MCI selects the KNNG build source", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 32;
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 8000);
    std::vector<float> vectors(total * dim);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto verify_source = [&](const std::string& source) {
        auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim, source));
        REQUIRE(index.has_value());
        auto build_result = index.value()->Build(make_dataset(ids, vectors, 0, total, dim));
        REQUIRE(build_result.has_value());
        REQUIRE(build_result.value().empty());
        const auto stats = vsag::JsonType::Parse(index.value()->GetStats());
        REQUIRE(stats["mci_has_index"].GetBool());
        REQUIRE(stats["mci_covered_nodes"].GetInt() == total);
    };

    SECTION("completed HGraph") {
        verify_source(vsag::HGRAPH_MCI_KNNG_SOURCE_HGRAPH);
    }
    SECTION("dedicated ODescent") {
        verify_source(vsag::HGRAPH_MCI_KNNG_SOURCE_ODESCENT);
    }
}

TEST_CASE("HGraph companion MCI serializes concurrent initial Add", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t batch_count = 24;
    constexpr int64_t total = batch_count * 2;
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 6000);
    std::vector<float> vectors(total * dim);
    for (int64_t i = 0; i < total * dim; ++i) {
        vectors[i] = static_cast<float>((i * 17) % 101);
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    auto add_batch = [&](int64_t offset) {
        ready.fetch_add(1, std::memory_order_release);
        while (not start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        return index.value()->Add(make_dataset(ids, vectors, offset, batch_count, dim));
    };
    auto first = std::async(std::launch::async, add_batch, 0);
    auto second = std::async(std::launch::async, add_batch, batch_count);
    while (ready.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    REQUIRE(first.get().has_value());
    REQUIRE(second.get().has_value());
    REQUIRE(index.value()->GetNumElements() == total);
    const auto stats = vsag::JsonType::Parse(index.value()->GetStats());
    REQUIRE(stats["mci_has_index"].GetBool());
}

TEST_CASE("HGraph companion MCI incrementally updates cliques after Add", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t base_count = 24;
    constexpr int64_t add_count = 4;
    constexpr int64_t total = base_count + add_count;

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);

    std::vector<float> vectors(total * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(make_dataset(ids, vectors, 0, base_count, dim));
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index.value()->GetMemoryUsageDetail().count("mci_cliques") == 1);
    std::stringstream streaming;
    REQUIRE_FALSE(index.value()->SerializeStreaming(streaming).has_value());
    const auto memory_before_add = index.value()->GetMemoryUsage();

    auto add_result = index.value()->Add(make_dataset(ids, vectors, base_count, add_count, dim));
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index.value()->GetMemoryUsage() > memory_before_add);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dim)
        ->Float32Vectors(vectors.data() + (base_count + 1) * dim)
        ->Owner(false);

    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    auto result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":0.5,)"
                                 R"("mci_seed_coverage":0,"hgraph_valid_ratio_threshold":1.0}})",
                                 filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    const auto search_statistics = vsag::JsonType::Parse(result.value()->GetStatistics());
    REQUIRE(search_statistics["distance_evaluations_by_phase"]["approximate"].GetUint64() > 0);
    REQUIRE(search_statistics["distance_evaluations"].GetUint64() ==
            search_statistics["dist_cmp"].GetUint64());
    REQUIRE(search_statistics["distance_evaluations_by_backend"]["fp32"].GetUint64() ==
            search_statistics["distance_evaluations"].GetUint64());
    const auto expected_seed_count =
        static_cast<uint64_t>(std::ceil(std::sqrt(static_cast<double>(total)) * 0.5));
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) ==
            expected_seed_count);

    result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0,"timeout_ms":0}})",
                                 filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"is_timeout"})[0] == "true");

    const auto stats = vsag::JsonType::Parse(index.value()->GetStats());
    REQUIRE(stats["mci_max_clique_size"].GetInt() > 0);

    auto callback_filter = std::make_shared<CallbackOnlyFilter>();
    result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 callback_filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("disabled")");

    auto invalid = vsag::Bitset::Make();
    for (int64_t i = 0; i < total; i += 2) {
        invalid->Set(ids[i]);
    }
    result = index.value()->KnnSearch(
        query,
        3,
        R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":100.0,)"
        R"("hgraph_valid_ratio_threshold":1.0}})",
        invalid);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) == total / 2);
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        REQUIRE_FALSE(invalid->Test(result.value()->GetIds()[i]));
    }

    auto all_invalid = vsag::Bitset::Make();
    for (auto id : ids) {
        all_invalid->Set(id);
    }
    result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 all_invalid);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("hgraph")");

    const std::vector<int64_t> mixed_ids{9000, ids[base_count + 1]};
    auto mixed_filter = std::make_shared<HalfRatioAllValidFilter>(mixed_ids);
    result =
        index.value()->KnnSearch(query,
                                 1,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":0.1,)"
                                 R"("mci_seed_coverage":0,"hgraph_valid_ratio_threshold":1.0}})",
                                 mixed_filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 1);
    REQUIRE(result.value()->GetIds()[0] == ids[base_count + 1]);
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) == 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("hgraph")");

    const std::vector<int64_t> stale_ids{9000, 9001, 9002};
    auto stale_filter = std::make_shared<CountingValidIdsFilter>(stale_ids);
    result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 stale_filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("hgraph")");
    REQUIRE(stale_filter->CheckCount() > 0);
}

TEST_CASE("HGraph cache-accelerated NSW build creates MCI companion", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 16;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 3000);
    std::vector<float> vectors(count * dim, 0.0F);
    std::vector<std::string> source_ids(count);
    for (int64_t i = 0; i < count; ++i) {
        vectors[i * dim] = static_cast<float>(i);
        vectors[i * dim + 1] = static_cast<float>(i % 3);
        source_ids[i] = "source-" + std::to_string(i);
    }
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(count)
                       ->Dim(dim)
                       ->Ids(ids.data())
                       ->Float32Vectors(vectors.data())
                       ->SourceID(source_ids.data())
                       ->Owner(false);
    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    params["index_param"]["graph_type"].SetString("nsw");

    auto source_index = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(source_index.has_value());
    auto result = source_index.value()->Build(dataset);
    REQUIRE(result.has_value());
    std::stringstream cache;
    REQUIRE(source_index.value()->ExportCache(cache).has_value());

    auto cached_index = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(cached_index.has_value());
    REQUIRE(cached_index.value()->ImportCache(cache).has_value());
    result = cached_index.value()->Build(dataset);
    REQUIRE(result.has_value());
    const auto stats = vsag::JsonType::Parse(cached_index.value()->GetStats());
    REQUIRE(stats["mci_has_index"].GetBool());
}

TEST_CASE("HGraph companion MCI incrementally adds INT8 vectors", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t base_count = 12;
    constexpr int64_t add_count = 2;
    constexpr int64_t total = base_count + add_count;

    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    params["dtype"].SetString("int8");
    params["index_param"]["base_quantization_type"].SetString("int8");
    auto index = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(index.has_value());

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 2000);
    std::vector<int8_t> vectors(total * dim);
    for (int64_t i = 0; i < total * dim; ++i) {
        vectors[i] = static_cast<int8_t>((i * 7) % 31);
    }
    auto make_int8_dataset = [&](int64_t offset, int64_t count) {
        return vsag::Dataset::Make()
            ->NumElements(count)
            ->Dim(dim)
            ->Ids(ids.data() + offset)
            ->Int8Vectors(vectors.data() + offset * dim)
            ->Owner(false);
    };

    auto result = index.value()->Build(make_int8_dataset(0, base_count));
    REQUIRE(result.has_value());
    REQUIRE(result.value().empty());
    result = index.value()->Add(make_int8_dataset(base_count, add_count));
    REQUIRE(result.has_value());
    REQUIRE(result.value().empty());
    REQUIRE(index.value()->GetNumElements() == total);

    auto query =
        vsag::Dataset::Make()->NumElements(1)->Dim(dim)->Int8Vectors(vectors.data())->Owner(false);
    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    auto search_result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 filter);
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(std::stoull(search_result.value()->GetStatistics({"dist_cmp"})[0]) > 0);
    REQUIRE(std::stoull(search_result.value()->GetStatistics({"hops"})[0]) > 0);
}

TEST_CASE("MCI builder expands negative inner-product distances", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 3;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors{10.0F, 0.0F, 9.9F, 0.0F, 9.8F, 0.0F};
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 2, 0, 1};

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.total = total;
    graph.row_stride = 2;
    graph.uniform_count = 2;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.alpha = 1.2F;
    params.thread_count = 1;
    params.metric = vsag::MetricType::METRIC_TYPE_IP;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE(std::any_of(
        cliques.begin(), cliques.end(), [](const auto& clique) { return clique.size() == total; }));
}

TEST_CASE("MCI builder preserves L2 differences for large coordinates", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 3;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors{1.0e8F, 1.0F, 1.0e8F, 2.0F, 1.0e8F, 100.0F};
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 2, 0, 1};

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.total = total;
    graph.row_stride = 2;
    graph.uniform_count = 2;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.alpha = 1.2F;
    params.thread_count = 1;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE(std::any_of(cliques.begin(), cliques.end(), [](const auto& clique) {
        return std::find(clique.begin(), clique.end(), 0) != clique.end() and
               std::find(clique.begin(), clique.end(), 1) != clique.end();
    }));
}

TEST_CASE("MCI builder includes duplicate-vector seed edges", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 3;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors(total * dim, 1.0F);
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 2, 0, 1};

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.total = total;
    graph.row_stride = 2;
    graph.uniform_count = 2;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.alpha = 1.2F;
    params.thread_count = 1;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE(std::any_of(
        cliques.begin(), cliques.end(), [](const auto& clique) { return clique.size() == total; }));
}

TEST_CASE("MCI builder keeps maximal cliques larger than clique_max", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 6;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors{
        0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 10.0F, 10.0F, 11.0F, 10.0F, 10.0F, 11.0F};
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 0, 0, 0, 2, 0, 1, 1, 1, 1,
                                                   0, 1, 2, 2, 2, 2, 4, 5, 3, 3, 3, 3,
                                                   5, 3, 4, 4, 4, 4, 3, 4, 5, 5, 5, 5};
    const std::vector<uint32_t> counts(total, 2);

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.counts = counts.data();
    graph.total = total;
    graph.row_stride = 6;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 5;
    params.clique_max = 2;
    params.max_degree = 2;
    params.alpha = 2.1F;
    params.thread_count = 2;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE_FALSE(cliques.empty());
    // clique_max is a lower bound on the size of a maximal clique, never a cap: an enumerated clique
    // is stored in full, so it may grow up to the candidate pool (mcs neighbours plus the seed).
    REQUIRE(std::all_of(cliques.begin(), cliques.end(), [&](const auto& clique) {
        return clique.size() >= params.clique_max and clique.size() <= params.candidate_limit + 1;
    }));
    // Both clusters of this data are triangles, so a size-3 maximal clique must survive intact.
    REQUIRE(std::any_of(
        cliques.begin(), cliques.end(), [](const auto& clique) { return clique.size() == 3; }));
}

TEST_CASE("MCI builder stores a full candidate-pool clique", "[ut][hgraph][mci]") {
    // Every pair of vectors is equally far apart, so each seed sees a complete graph over its
    // candidate pool and the only maximal clique has size candidate_limit + 1.
    constexpr uint64_t total = 7;
    constexpr uint64_t dim = 7;
    constexpr uint64_t candidate_limit = 6;
    std::vector<float> vectors(total * dim, 0.0F);
    for (uint64_t id = 0; id < total; ++id) {
        vectors[id * dim + id] = 1.0F;
    }
    std::vector<vsag::InnerIdType> neighbors(total * candidate_limit);
    for (uint64_t id = 0; id < total; ++id) {
        uint64_t rank = 0;
        for (uint64_t other = 0; other < total; ++other) {
            if (other == id) {
                continue;
            }
            neighbors[id * candidate_limit + rank] = static_cast<vsag::InnerIdType>(other);
            ++rank;
        }
    }
    const std::vector<uint32_t> counts(total, candidate_limit);

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.counts = counts.data();
    graph.total = total;
    graph.row_stride = candidate_limit;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = candidate_limit;
    params.clique_max = 3;
    params.max_degree = 32;
    params.alpha = 1.2F;
    params.thread_count = 2;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE_FALSE(cliques.empty());
    REQUIRE(std::all_of(cliques.begin(), cliques.end(), [&](const auto& clique) {
        return clique.size() >= params.clique_max;
    }));
    REQUIRE(std::any_of(cliques.begin(), cliques.end(), [&](const auto& clique) {
        return clique.size() == candidate_limit + 1;
    }));
}

TEST_CASE("MCI builder transports worker exceptions", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 3;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors{0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 2, 0, 1};

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.total = total;
    graph.row_stride = 2;
    graph.uniform_count = 2;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.alpha = 1.2F;
    params.thread_count = 2;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    WorkerFailAllocator allocator;
    REQUIRE_THROWS_AS(vsag::BuildMCICliques(vectors.data(), graph, params, &allocator),
                      std::bad_alloc);
}

TEST_CASE("MCI runner handles empty and growing local graphs", "[ut][hgraph][mci]") {
    vsag::mci::ccrmce_runner<TestEdge, uint32_t> runner;
    std::vector<std::vector<uint32_t>> cliques(16);
    std::vector<std::atomic<int>> clique_counts(10);
    for (auto& count : clique_counts) {
        count.store(0, std::memory_order_relaxed);
    }

    std::vector<TestEdge> edges;
    REQUIRE(runner.run(edges, cliques, 2, clique_counts, 16) == 0);

    for (uint32_t vertex_count : {5U, 8U, 9U}) {
        edges.clear();
        for (uint32_t lhs = 0; lhs < vertex_count; ++lhs) {
            for (uint32_t rhs = lhs + 1; rhs < vertex_count; ++rhs) {
                edges.push_back({lhs, rhs});
            }
        }
        REQUIRE(runner.run(edges, cliques, 2, clique_counts, 16) > 0);
    }
}

TEST_CASE("MCI runner preserves the P extension in must cliques", "[ut][hgraph][mci]") {
    const std::vector<TestEdge> input_edges{{0, 1},
                                            {0, 2},
                                            {0, 4},
                                            {0, 5},
                                            {1, 3},
                                            {1, 4},
                                            {1, 5},
                                            {2, 3},
                                            {2, 4},
                                            {2, 5},
                                            {3, 4},
                                            {3, 5},
                                            {4, 5}};
    auto edges = input_edges;
    std::set<std::pair<uint32_t, uint32_t>> adjacency;
    for (const auto& edge : input_edges) {
        adjacency.emplace(std::min(edge.u, edge.v), std::max(edge.u, edge.v));
    }

    vsag::mci::ccrmce_runner<TestEdge, uint32_t> runner;
    std::vector<std::vector<uint32_t>> cliques(16);
    std::vector<std::atomic<int>> clique_counts(6);
    for (auto& count : clique_counts) {
        count.store(0, std::memory_order_relaxed);
    }

    const auto clique_count = runner.run(edges, cliques, 4, clique_counts, 16);
    REQUIRE(clique_count > 0);
    bool has_p_extension = false;
    for (uint32_t clique_id = 0; clique_id < clique_count; ++clique_id) {
        const auto& clique = cliques[clique_id];
        for (uint64_t i = 0; i < clique.size(); ++i) {
            for (uint64_t j = i + 1; j < clique.size(); ++j) {
                if (adjacency.count(
                        {std::min(clique[i], clique[j]), std::max(clique[i], clique[j])}) == 0) {
                    has_p_extension = true;
                }
            }
        }
    }
    REQUIRE(has_p_extension);
}

TEST_CASE("Clique base view pins CSR storage", "[ut][hgraph][mci]") {
    vsag::DefaultAllocator allocator;
    vsag::CliqueDataCell cell(&allocator);
    auto assign_clique = [&]() {
        vsag::Vector<vsag::InnerIdType> p_maxc(&allocator);
        vsag::Vector<vsag::InnerIdType> maxcs(&allocator);
        vsag::Vector<vsag::InnerIdType> p_node_to_cid(&allocator);
        vsag::Vector<vsag::InnerIdType> node_to_cids(&allocator);
        p_maxc.insert(p_maxc.end(), {0, 2});
        maxcs.insert(maxcs.end(), {0, 1});
        p_node_to_cid.insert(p_node_to_cid.end(), {0, 1, 2});
        node_to_cids.insert(node_to_cids.end(), {0, 0});
        cell.Assign(std::move(p_maxc),
                    std::move(maxcs),
                    std::move(p_node_to_cid),
                    std::move(node_to_cids),
                    2);
    };
    assign_clique();
    REQUIRE(cell.HasCliqueIndex(2));
    cell.MarkUnavailable();
    REQUIRE_FALSE(cell.HasCliqueIndex(2));
    cell.MarkAvailable(2);
    REQUIRE(cell.HasCliqueIndex(2));

    vsag::CliqueDataCellBaseView view;
    REQUIRE(cell.TryGetBaseView(2, view));
    std::atomic<bool> writer_started{false};
    auto writer = std::async(std::launch::async, [&]() {
        writer_started.store(true, std::memory_order_release);
        assign_clique();
    });
    while (not writer_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    REQUIRE(writer.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);

    view.guard.unlock();
    REQUIRE(writer.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    writer.get();
}

TEST_CASE("HGraph Merge rebuilds the MCI companion", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 16;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 5000);
    std::vector<float> vectors(count * dim, 0.0F);
    for (int64_t i = 0; i < count; ++i) {
        vectors[i * dim] = static_cast<float>(i);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto source = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    auto destination = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(source.value()->Build(make_dataset(ids, vectors, 0, count, dim)).has_value());

    vsag::MergeUnit unit;
    unit.index = source.value();
    unit.id_map_func = [](int64_t id) { return std::make_tuple(true, id); };
    REQUIRE(destination.value()->Merge({unit}).has_value());
    const auto stats = vsag::JsonType::Parse(destination.value()->GetStats());
    REQUIRE(stats["mci_has_index"].GetBool());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);
    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    auto result = destination.value()->KnnSearch(
        query,
        3,
        R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
        R"("hgraph_valid_ratio_threshold":1.0}})",
        filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
}

TEST_CASE("HGraph MCI inlines an identity-mapped filter bitmap", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 64;
    constexpr int64_t valid_count = total / 2;
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<float> vectors(total * dim);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }
    std::vector<int64_t> bitmap_valid_ids(valid_count);
    std::iota(bitmap_valid_ids.begin(), bitmap_valid_ids.end(), 0);

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    // Query with the vector of a filtered-out element, so that a wrong validity answer is visible.
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data() + (total - 1) * dim)
                     ->Owner(false);
    const std::string search_params =
        R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":0.1,"mci_seed_coverage":0,)"
        R"("hgraph_valid_ratio_threshold":1.0}})";

    auto bitmap_filter =
        std::make_shared<RangeBitmapFilter>(0, total, valid_count, bitmap_valid_ids, true);
    auto bitmap_result = index.value()->KnnSearch(query, 5, search_params, bitmap_filter);
    REQUIRE(bitmap_result.has_value());
    REQUIRE(bitmap_result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(vsag::JsonType::Parse(bitmap_result.value()->GetStatistics())["mci_bitmap_fast_path"]
                .GetBool());
    REQUIRE(bitmap_result.value()->GetDim() > 0);
    for (int64_t i = 0; i < bitmap_result.value()->GetDim(); ++i) {
        REQUIRE(bitmap_result.value()->GetIds()[i] < valid_count);
    }

    // The same filter without a bitmap searches through the callback path and must agree exactly.
    auto callback_filter =
        std::make_shared<RangeBitmapFilter>(0, total, valid_count, bitmap_valid_ids, false);
    auto callback_result = index.value()->KnnSearch(query, 5, search_params, callback_filter);
    REQUIRE(callback_result.has_value());
    REQUIRE_FALSE(
        vsag::JsonType::Parse(callback_result.value()->GetStatistics())["mci_bitmap_fast_path"]
            .GetBool());
    REQUIRE(callback_result.value()->GetDim() == bitmap_result.value()->GetDim());
    for (int64_t i = 0; i < callback_result.value()->GetDim(); ++i) {
        REQUIRE(callback_result.value()->GetIds()[i] == bitmap_result.value()->GetIds()[i]);
    }
}

TEST_CASE("HGraph MCI ignores a bitmap whose ids are not inner ids", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 64;
    constexpr int64_t valid_count = total / 2;
    constexpr int64_t first_label = 8000;
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), first_label);
    std::vector<float> vectors(total * dim);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }
    // The exposed bitmap marks exactly the ids this filter rejects. Reusing it as an inner-id
    // bitmap would return them, so the label table not being the identity must disable the bitmap.
    std::vector<int64_t> bitmap_valid_ids(valid_count);
    std::iota(bitmap_valid_ids.begin(), bitmap_valid_ids.end(), valid_count);

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data() + (total - 1) * dim)
                     ->Owner(false);
    const std::string search_params =
        R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":5.0,)"
        R"("hgraph_valid_ratio_threshold":1.0}})";

    auto filter = std::make_shared<RangeBitmapFilter>(
        first_label, total, valid_count, bitmap_valid_ids, true);
    auto result = index.value()->KnnSearch(query, 5, search_params, filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE_FALSE(
        vsag::JsonType::Parse(result.value()->GetStatistics())["mci_bitmap_fast_path"].GetBool());
    REQUIRE(result.value()->GetDim() > 0);
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        const auto id = result.value()->GetIds()[i];
        REQUIRE(id >= first_label);
        REQUIRE(id < first_label + valid_count);
    }
}

TEST_CASE("HGraph MCI skips clique expansion when seeds cover every valid point",
          "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 64;

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);

    std::vector<float> vectors(total * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    // ceil(sqrt(64) * 1.0) == 8 seeds: a filter with at most 8 valid labels is enumerated
    // completely, so the seed phase already produces the exact top-k.
    const char* search_params = R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                R"("hgraph_valid_ratio_threshold":1.0}})";
    // The partial case has to keep the ratio-based budget, so the coverage budget is disabled.
    const char* partial_params = R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("mci_seed_coverage":0,"hgraph_valid_ratio_threshold":1.0}})";

    auto exact_topk = [&](int64_t valid_count) {
        std::vector<std::pair<float, int64_t>> scored;
        for (int64_t i = 0; i < valid_count; ++i) {
            float distance = 0.0F;
            for (int64_t d = 0; d < dim; ++d) {
                const float delta = vectors[i * dim + d] - vectors[d];
                distance += delta * delta;
            }
            scored.emplace_back(distance, ids[i]);
        }
        std::sort(scored.begin(), scored.end());
        std::vector<int64_t> topk;
        for (int64_t i = 0; i < 3; ++i) {
            topk.emplace_back(scored[i].second);
        }
        std::sort(topk.begin(), topk.end());
        return topk;
    };

    std::vector<int64_t> covered(ids.begin(), ids.begin() + 5);
    auto covered_filter = std::make_shared<HalfRatioAllValidFilter>(covered);
    auto covered_result = index.value()->KnnSearch(query, 3, search_params, covered_filter);
    REQUIRE(covered_result.has_value());
    REQUIRE(covered_result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(std::stoull(covered_result.value()->GetStatistics({"mci_seed_count"})[0]) == 5);
    // Every valid point was scored by the seed phase, so the returned search skipped the
    // whole clique expansion. Auxiliary entry-point searches share the statistics object and
    // still contribute a few hops, which is why the hop count is not asserted to be zero.
    REQUIRE(covered_result.value()->GetStatistics({"mci_seed_saturated"})[0] == "true");
    const auto covered_hops = std::stoull(covered_result.value()->GetStatistics({"hops"})[0]);

    std::vector<int64_t> returned(covered_result.value()->GetIds(),
                                  covered_result.value()->GetIds() + 3);
    std::sort(returned.begin(), returned.end());
    REQUIRE(returned == exact_topk(5));

    // More valid labels than seeds: the seed list is a stride sample, so the expansion still
    // contributes and must not be skipped.
    std::vector<int64_t> partial(ids.begin(), ids.begin() + 40);
    auto partial_filter = std::make_shared<HalfRatioAllValidFilter>(partial);
    auto partial_result = index.value()->KnnSearch(query, 3, partial_params, partial_filter);
    REQUIRE(partial_result.has_value());
    REQUIRE(partial_result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(partial_result.value()->GetStatistics({"mci_seed_saturated"})[0] == "false");
    REQUIRE(std::stoull(partial_result.value()->GetStatistics({"hops"})[0]) > covered_hops);
}

namespace {

// Shared fixture: `total` vectors with a deterministic, non-degenerate layout.
std::vector<float>
make_test_vectors(int64_t total, int64_t dim) {
    std::vector<float> vectors(total * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 8);
        vectors[i * dim + 1] = static_cast<float>(i % 8);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 11);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 13);
    }
    return vectors;
}

}  // namespace

TEST_CASE("HGraph MCI coverage budget seeds every valid point when it fits the cap",
          "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 512;
    constexpr int64_t valid_count = 200;  // ceil(sqrt(512) * 0.1) == 3 with the ratio budget

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);
    auto vectors = make_test_vectors(total, dim);

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    std::vector<int64_t> valid(ids.begin(), ids.begin() + valid_count);
    auto filter = std::make_shared<HalfRatioAllValidFilter>(valid);

    // Default coverage budget: the seed list enumerates the whole valid set, the searcher skips
    // the expansion and the answer is the exact top-k.
    auto covered = index.value()->KnnSearch(
        query,
        10,
        R"({"hgraph":{"ef_search":512,"use_mci":true,"mci_seed_ratio":0.1,)"
        R"("hgraph_valid_ratio_threshold":1.0}})",
        filter);
    REQUIRE(covered.has_value());
    REQUIRE(std::stoull(covered.value()->GetStatistics({"mci_seed_count"})[0]) == valid_count);
    REQUIRE(covered.value()->GetStatistics({"mci_seed_saturated"})[0] == "true");

    std::vector<std::pair<float, int64_t>> scored;
    for (int64_t i = 0; i < valid_count; ++i) {
        float distance = 0.0F;
        for (int64_t d = 0; d < dim; ++d) {
            const float delta = vectors[static_cast<size_t>(i * dim + d)] - vectors[d];
            distance += delta * delta;
        }
        scored.emplace_back(distance, ids[i]);
    }
    std::sort(scored.begin(), scored.end());
    std::vector<int64_t> expected;
    for (int64_t i = 0; i < 10; ++i) {
        expected.emplace_back(scored[i].second);
    }
    std::sort(expected.begin(), expected.end());
    std::vector<int64_t> returned(covered.value()->GetIds(), covered.value()->GetIds() + 10);
    std::sort(returned.begin(), returned.end());
    REQUIRE(returned == expected);

    // With the coverage budget disabled the ratio budget applies, so the seed phase is partial.
    auto partial = index.value()->KnnSearch(
        query,
        10,
        R"({"hgraph":{"ef_search":512,"use_mci":true,"mci_seed_ratio":0.1,)"
        R"("mci_seed_coverage":0,"hgraph_valid_ratio_threshold":1.0}})",
        filter);
    REQUIRE(partial.has_value());
    REQUIRE(std::stoull(partial.value()->GetStatistics({"mci_seed_count"})[0]) < valid_count);
    REQUIRE(partial.value()->GetStatistics({"mci_seed_saturated"})[0] == "false");
}

TEST_CASE("HGraph MCI idle window aborts unproductive clique expansion", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 1024;
    constexpr int64_t valid_count = 512;

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);
    auto vectors = make_test_vectors(total, dim);

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    std::vector<int64_t> valid(ids.begin(), ids.begin() + valid_count);
    auto filter = std::make_shared<HalfRatioAllValidFilter>(valid);

    // Both runs use the ratio budget so that the expansion actually happens.
    auto full = index.value()->KnnSearch(
        query,
        10,
        R"({"hgraph":{"ef_search":1024,"use_mci":true,"mci_seed_ratio":0.1,"mci_seed_coverage":0,)"
        R"("mci_expansion_idle_window":0,"hgraph_valid_ratio_threshold":1.0}})",
        filter);
    auto idle = index.value()->KnnSearch(
        query,
        10,
        R"({"hgraph":{"ef_search":1024,"use_mci":true,"mci_seed_ratio":0.1,"mci_seed_coverage":0,)"
        R"("mci_expansion_idle_window":1,"hgraph_valid_ratio_threshold":1.0}})",
        filter);
    REQUIRE(full.has_value());
    REQUIRE(idle.has_value());

    const auto full_hops = std::stoull(full.value()->GetStatistics({"hops"})[0]);
    const auto idle_hops = std::stoull(idle.value()->GetStatistics({"hops"})[0]);
    REQUIRE(full_hops > 0);
    REQUIRE(idle_hops < full_hops);

    std::vector<int64_t> full_ids(full.value()->GetIds(), full.value()->GetIds() + 10);
    std::vector<int64_t> idle_ids(idle.value()->GetIds(), idle.value()->GetIds() + 10);
    const auto overlap = std::count_if(idle_ids.begin(), idle_ids.end(), [&](int64_t id) {
        return std::find(full_ids.begin(), full_ids.end(), id) != full_ids.end();
    });
    // The idle window is a heuristic: require most of the top-k to survive the earlier stop.
    REQUIRE(overlap >= 5);
}

TEST_CASE("HGraph MCI bitset sampling never claims coverage it did not scan", "[ut][hgraph][mci]") {
    // Regression: with total=6000 and a seed budget of 8 the bitset collector probes 4096 strided
    // positions, so `id_step == 1` must not be reported as full coverage -- id 3 is never probed.
    constexpr int64_t dim = 4;
    constexpr int64_t total = 6000;
    constexpr int64_t near_count = 8;  // ids 0..7 form one cluster, the rest is far away

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);
    std::vector<float> vectors(total * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        if (i < near_count) {
            vectors[i * dim + 1] = 0.1F * static_cast<float>(i);
        } else {
            vectors[i * dim] = 1000.0F + static_cast<float>(i % 17);
            vectors[i * dim + 1] = static_cast<float>(i % 23);
            vectors[i * dim + 2] = static_cast<float>(i % 29);
            vectors[i * dim + 3] = static_cast<float>(i % 31);
        }
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    // Query sits exactly on id 3, which is valid but is skipped by the strided bitset scan.
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + 3 * dim)->Owner(false);

    auto blacklist = vsag::Bitset::Make();
    for (int64_t i = 0; i < total; ++i) {
        if (i != 0 and i != 3) {
            blacklist->Set(ids[i]);
        }
    }

    // Coverage is disabled so the budget stays at ceil(sqrt(6000) * 0.012) == 1 seed, below
    // the two valid ids: the scan is cut short by the budget and must not claim coverage., below the two valid ids, so the collector
    // strides over the id space instead of enumerating it and must not claim coverage.
    auto result = index.value()->KnnSearch(
        query,
        1,
        R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":0.012,"mci_seed_coverage":0,)"
        R"("hgraph_valid_ratio_threshold":1.0}})",
        blacklist);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    // Only id 0 is probed, so the seed phase is incomplete and the expansion has to run.
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) == 1);
    // The load-bearing assertion: the collector only probed 4096 of 6000 ids, so it must not claim
    // full coverage.  Whether the following expansion rediscovers the skipped point depends on the
    // sampled clique build, so that is covered by the expansion tests instead of asserted here.
    REQUIRE(result.value()->GetStatistics({"mci_seed_saturated"})[0] == "false");
    REQUIRE(result.value()->GetDim() == 1);
}

TEST_CASE("HGraph MCI reports saturation for batched queries", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 64;

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);
    auto vectors = make_test_vectors(total, dim);

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    std::vector<float> queries(2 * dim, 0.0F);
    std::copy_n(vectors.begin(), dim, queries.begin());
    std::copy_n(vectors.begin() + dim, dim, queries.begin() + dim);
    auto query = vsag::Dataset::Make();
    query->NumElements(2)->Dim(dim)->Float32Vectors(queries.data())->Owner(false);

    std::vector<int64_t> covered(ids.begin(), ids.begin() + 5);
    auto filter = std::make_shared<HalfRatioAllValidFilter>(covered);

    auto result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 filter);
    REQUIRE(result.has_value());
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) == 5 * 2);
    // Aggregated with OR semantics over the sub-queries.
    REQUIRE(result.value()->GetStatistics({"mci_seed_saturated"})[0] == "true");
}

TEST_CASE("HGraph MCI bitset full scan reports saturation when it exactly fills the budget",
          "[ut][hgraph][mci]") {
    // total == sample_count, so the scan is a true full pass; the budget is filled by the last
    // probed id, which must still count as full coverage (the whole id space was visited).
    constexpr int64_t dim = 4;
    constexpr int64_t total = 4096;

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);
    std::vector<float> vectors(total * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>((i * 7) % 101);
        vectors[i * dim + 1] = static_cast<float>((i * 11) % 103);
        vectors[i * dim + 2] = static_cast<float>((i * 13) % 107);
        vectors[i * dim + 3] = static_cast<float>((i * 17) % 109);
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dim)
        ->Float32Vectors(vectors.data() + (total - 1) * dim)
        ->Owner(false);

    // Exactly four valid ids, the last of which is the final probed position.
    const std::vector<int64_t> keep{ids[0], ids[1], ids[2], ids[total - 1]};
    auto blacklist = vsag::Bitset::Make();
    for (auto id : ids) {
        if (std::find(keep.begin(), keep.end(), id) == keep.end()) {
            blacklist->Set(id);
        }
    }

    // ceil(sqrt(4096) * 0.0625) == 4 seeds, and sample_count == total == 4096.
    auto result = index.value()->KnnSearch(
        query,
        1,
        R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":0.0625,)"
        R"("hgraph_valid_ratio_threshold":1.0}})",
        blacklist);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) == 4);
    REQUIRE(result.value()->GetStatistics({"mci_seed_saturated"})[0] == "true");
    REQUIRE(result.value()->GetDim() == 1);
    REQUIRE(result.value()->GetIds()[0] == ids[total - 1]);
}

TEST_CASE("HGraph MCI enumerates a bitset filter and answers exactly", "[ut][hgraph][mci]") {
    // With a bitmap-style filter the per-id check is a bit test, so once the seed budget covers the
    // valid set the collector walks the id space once, collects every valid point and the seed phase
    // answers exactly without any clique expansion (and without the merged-marks array).
    constexpr int64_t dim = 4;
    constexpr int64_t total = 6000;
    constexpr int64_t near_count = 8;

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);
    std::vector<float> vectors(total * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        if (i < near_count) {
            vectors[i * dim + 1] = 0.1F * static_cast<float>(i);
        } else {
            vectors[i * dim] = 1000.0F + static_cast<float>(i % 17);
            vectors[i * dim + 1] = static_cast<float>(i % 23);
            vectors[i * dim + 2] = static_cast<float>(i % 29);
            vectors[i * dim + 3] = static_cast<float>(i % 31);
        }
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + 3 * dim)->Owner(false);

    auto blacklist = vsag::Bitset::Make();
    for (int64_t i = 0; i < total; ++i) {
        if (i != 0 and i != 3) {
            blacklist->Set(ids[i]);
        }
    }

    // ceil(sqrt(6000) * 0.1) == 8 seeds >= 2 valid ids, so the collector enumerates the id space.
    auto result =
        index.value()->KnnSearch(query,
                                 1,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":0.1,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 blacklist);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) == 2);
    REQUIRE(result.value()->GetStatistics({"mci_seed_saturated"})[0] == "true");
    // Bitset filters expose no merged bitmap, so the fast path stays off.
    REQUIRE_FALSE(
        vsag::JsonType::Parse(result.value()->GetStatistics())["mci_bitmap_fast_path"].GetBool());
    REQUIRE(result.value()->GetDim() == 1);
    REQUIRE(result.value()->GetIds()[0] == ids[3]);
}

TEST_CASE("HGraph per-route ef_search sets each route's search breadth", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 64;

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);

    std::vector<float> vectors(total * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    // 40 of 64 labels are valid and the seed budget stays ratio-based, so the partial seeds
    // force a real expansion whose breadth is exactly the route's ef.
    std::vector<int64_t> valid(ids.begin(), ids.begin() + 40);
    auto filter = std::make_shared<HalfRatioAllValidFilter>(valid);

    // MCI band: a huge shared ef must not widen the hybrid, which has its own breadth.
    const char* mci_routed =
        R"({"hgraph":{"ef_search":4096,"mci_ef_search":16,"use_mci":true,)"
        R"("mci_seed_ratio":1.0,"mci_seed_coverage":0,"hgraph_valid_ratio_threshold":1.0}})";
    const char* mci_reference = R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                R"("mci_seed_coverage":0,"hgraph_valid_ratio_threshold":1.0}})";
    auto mci_result = index.value()->KnnSearch(query, 3, mci_routed, filter);
    auto mci_baseline = index.value()->KnnSearch(query, 3, mci_reference, filter);
    REQUIRE(mci_result.has_value());
    REQUIRE(mci_baseline.has_value());
    REQUIRE(mci_result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(mci_baseline.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(mci_result.value()->GetStatistics({"mci_seed_saturated"})[0] == "false");
    // Same breadth, same deterministic walk: ignoring `mci_ef_search` would instead run a
    // 4096-wide expansion and explore strictly more hops.
    REQUIRE(mci_result.value()->GetStatistics({"hops"}) ==
            mci_baseline.value()->GetStatistics({"hops"}));

    // HGraph band: the plain walk follows `hgraph_ef_search` under the same rule.
    const char* hgraph_routed =
        R"({"hgraph":{"ef_search":4096,"hgraph_ef_search":16,"hgraph_valid_ratio_threshold":0.5}})";
    const char* hgraph_reference =
        R"({"hgraph":{"ef_search":16,"hgraph_valid_ratio_threshold":0.5}})";
    auto hgraph_result = index.value()->KnnSearch(query, 3, hgraph_routed, filter);
    auto hgraph_baseline = index.value()->KnnSearch(query, 3, hgraph_reference, filter);
    REQUIRE(hgraph_result.has_value());
    REQUIRE(hgraph_baseline.has_value());
    REQUIRE(hgraph_result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("hgraph")");
    REQUIRE(hgraph_result.value()->GetStatistics({"hops"}) ==
            hgraph_baseline.value()->GetStatistics({"hops"}));
}
