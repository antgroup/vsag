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
generate_hgraph_mci_params(int64_t dim) {
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
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
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
    REQUIRE(stats["mci_max_clique_size"].GetInt() <= 4);

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
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 mixed_filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 1);
    REQUIRE(result.value()->GetIds()[0] == ids[base_count + 1]);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");

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
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(stale_filter->CheckCount() == 0);
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

TEST_CASE("MCI builder treats clique_max only as a storage cap", "[ut][hgraph][mci]") {
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
    params.clique_max = 50;
    params.max_degree = 2;
    params.alpha = 2.1F;
    params.thread_count = 2;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE(std::any_of(
        cliques.begin(), cliques.end(), [](const auto& clique) { return clique.size() == 3; }));
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

TEST_CASE("MCI runner only saves pairwise-connected cliques", "[ut][hgraph][mci]") {
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
    for (uint32_t clique_id = 0; clique_id < clique_count; ++clique_id) {
        const auto& clique = cliques[clique_id];
        for (uint64_t i = 0; i < clique.size(); ++i) {
            for (uint64_t j = i + 1; j < clique.size(); ++j) {
                REQUIRE(adjacency.count(
                            {std::min(clique[i], clique[j]), std::max(clique[i], clique[j])}) == 1);
            }
        }
    }
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
