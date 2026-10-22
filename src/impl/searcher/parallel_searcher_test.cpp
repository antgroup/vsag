
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

#include "parallel_searcher.h"

#include <set>
#include <vector>

#include "searcher_test.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("ParallelSearcher matches BasicSearcher on a generic graph",
          "[ut][ParallelSearcher][regression]") {
    constexpr uint32_t base_size = 200;
    constexpr uint64_t dim = 32;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;

    constexpr const char* param_template = R"({{"type": "{}"}})";
    auto quantizer_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_template, "fp32")));
    auto io_param = IOParameter::GetIOParameterByJson(
        JsonType::Parse(fmt::format(param_template, "memory_io")));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
        quantizer_param, io_param, common);
    flatten->Train(base_vectors.data(), base_size);
    flatten->BatchInsertVector(base_vectors.data(), base_size, ids.data());

    auto pool = std::make_shared<VisitedListPool>(
        2, allocator.get(), flatten->TotalCount(), allocator.get());
    // Reserve enough workers for every requested parallelism in this regression.
    auto executor = std::make_shared<SafeThreadPool>(std::make_shared<DefaultThreadPool>(8));
    auto parallel = std::make_shared<ParallelSearcher>(common, executor);
    auto basic = std::make_shared<BasicSearcher>(common);

    InnerSearchParam search_param;
    search_param.ep = 0;
    search_param.ef = 40;
    search_param.topk = 20;
    search_param.parallel_search_thread_count = 4;

    auto run_search = [&](const auto& searcher, const GraphInterfacePtr& graph) {
        auto visited_list = pool->TakeOne();
        QueryContext* context = nullptr;
        auto result = searcher->Search(graph,
                                       flatten,
                                       visited_list,
                                       base_vectors.data(),
                                       search_param,
                                       LabelTablePtr{},
                                       context);
        pool->ReturnOne(visited_list);
        std::set<std::pair<float, InnerIdType>> values;
        while (not result->Empty()) {
            values.insert(result->Top());
            result->Pop();
        }
        return values;
    };

    // Repeated batches exercise reuse of the worker completion counter (#3225).
    // Compare each component explicitly so ARM failures retain the differing ID/distance.
    for (const auto& graph : {MakeRingGraph(base_size, 8), MakeIrregularGraph(base_size)}) {
        const auto expected = run_search(basic, graph);
        for (const int64_t threads : {1, 2, 4, 8}) {
            search_param.parallel_search_thread_count = threads;
            for (uint64_t repeat = 0; repeat < 16; ++repeat) {
                CAPTURE(threads, repeat, graph->MaximumDegree());
                const auto actual = run_search(parallel, graph);
                REQUIRE(actual.size() == expected.size());
                auto found = actual.begin();
                for (const auto& value : expected) {
                    CAPTURE(found->second, value.second, found->first, value.first);
                    REQUIRE(found->second == value.second);
                    REQUIRE(found->first == value.first);
                    ++found;
                }
            }
        }
    }
    search_param.parallel_search_thread_count = 4;

    SECTION("a missing executor is valid only for serial search") {
        const auto graph = MakeRingGraph(base_size, 8);
        auto no_executor = std::make_shared<ParallelSearcher>(common, nullptr);
        auto serial_param = search_param;
        serial_param.parallel_search_thread_count = 1;

        auto visited_list = pool->TakeOne();
        auto serial_result =
            no_executor->Search(graph, flatten, visited_list, base_vectors.data(), serial_param);
        pool->ReturnOne(visited_list);
        REQUIRE(not serial_result->Empty());

        for (const int64_t invalid_parallelism : {0, -1}) {
            auto invalid_param = search_param;
            invalid_param.parallel_search_thread_count = invalid_parallelism;
            visited_list = pool->TakeOne();
            REQUIRE_THROWS_AS(no_executor->Search(
                                  graph, flatten, visited_list, base_vectors.data(), invalid_param),
                              VsagException);
            pool->ReturnOne(visited_list);
        }

        visited_list = pool->TakeOne();
        auto missing_flatten =
            no_executor->Search(graph, nullptr, visited_list, base_vectors.data(), search_param);
        pool->ReturnOne(visited_list);
        REQUIRE(missing_flatten->Empty());

        visited_list = pool->TakeOne();
        auto missing_graph =
            no_executor->Search(nullptr, flatten, visited_list, base_vectors.data(), search_param);
        pool->ReturnOne(visited_list);
        REQUIRE(missing_graph->Empty());

        visited_list = pool->TakeOne();
        REQUIRE_THROWS_AS(
            no_executor->Search(graph, flatten, visited_list, base_vectors.data(), search_param),
            VsagException);
        pool->ReturnOne(visited_list);
    }

    auto visited_list = pool->TakeOne();
    auto empty_result =
        parallel->Search(nullptr, flatten, visited_list, base_vectors.data(), search_param);
    pool->ReturnOne(visited_list);
    REQUIRE(empty_result->Empty());
}

TEST_CASE("ParallelSearcher traverses through a non-finite-distance bridge",
          "[ut][ParallelSearcher][nonfinite]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 1;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;

    constexpr const char* param_temp = R"({{"type": "{}"}})";
    auto quantizer_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_temp, "fp32")));
    auto io_param =
        IOParameter::GetIOParameterByJson(JsonType::Parse(fmt::format(param_temp, "memory_io")));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
        quantizer_param, io_param, common);
    flatten->SetQuantizer(
        std::make_shared<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>>(1, allocator.get()));
    flatten->SetIO(std::make_unique<MemoryIO>(allocator.get()));
    const auto bridge_distance =
        GENERATE(std::numeric_limits<float>::max(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> vectors = {
        10.0F, bridge_distance, bridge_distance, bridge_distance, bridge_distance, 1.0F};
    std::vector<InnerIdType> ids = {0, 1, 2, 3, 4, 5};
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());

    auto graph = std::make_shared<MockGraphDataCell>(
        std::vector<std::vector<InnerIdType>>{{1}, {2}, {3}, {4}, {5}, {}});
    auto pool = std::make_shared<VisitedListPool>(1, allocator.get(), ids.size(), allocator.get());
    InnerSearchParam param;
    param.ep = 0;
    param.ef = 2;
    param.topk = 2;
    param.parallel_search_thread_count = 2;
    float query = 0.0F;
    auto vl = pool->TakeOne();
    auto result = ParallelSearcher(common, SafeThreadPool::FactoryDefaultThreadPool())
                      .Search(graph, flatten, vl, &query, param);
    pool->ReturnOne(vl);

    REQUIRE(result->Size() <= param.ef);
    bool found_target = false;
    while (not result->Empty()) {
        found_target = found_target or result->Top().second == 5;
        result->Pop();
    }
    REQUIRE(found_target);
}

TEST_CASE("ParallelSearcher publishes legacy statistics", "[ut][ParallelSearcher][statistics]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 1;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto quantizer_param =
        QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
    auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
        quantizer_param, io_param, common);
    const std::vector<float> vectors{1.0F, 2.0F};
    std::vector<InnerIdType> ids{0, 1};
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());
    auto graph =
        std::make_shared<MockGraphDataCell>(std::vector<std::vector<InnerIdType>>{{1}, {0}});
    auto pool = std::make_shared<VisitedListPool>(1, allocator.get(), ids.size(), allocator.get());
    InnerSearchParam param;
    param.ep = 0;
    param.ef = 2;
    param.topk = 2;
    param.parallel_search_thread_count = GENERATE(1, 2);
    SearchStatistics stats;
    QueryContext context;
    context.stats = &stats;
    float query = 0.0F;
    auto visited = pool->TakeOne();
    auto result = ParallelSearcher(common, SafeThreadPool::FactoryDefaultThreadPool())
                      .Search(graph, flatten, visited, &query, param, LabelTablePtr{}, &context);
    pool->ReturnOne(visited);
    REQUIRE(result->Size() == 2);
    CHECK(stats.dist_cmp.load() == 1);
    CHECK(stats.hops.load() == 2);
    CHECK(stats.distance_evaluations.load() == 2);
    CHECK(stats
              .distance_evaluations_by_phase[static_cast<uint64_t>(
                  DistanceEvaluationPhase::APPROXIMATE)]
              .load() == 2);
}

TEST_CASE("ParallelSearcher distance early stop matches BasicSearcher",
          "[ut][ParallelSearcher][distance_early_stop]") {
    constexpr uint32_t base_size = 200;
    constexpr uint64_t dim = 32;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto quantizer_param =
        QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
    auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
        quantizer_param, io_param, common);
    flatten->Train(base_vectors.data(), base_size);
    flatten->BatchInsertVector(base_vectors.data(), base_size, ids.data());

    auto pool = std::make_shared<VisitedListPool>(
        2, allocator.get(), flatten->TotalCount(), allocator.get());
    auto parallel =
        std::make_shared<ParallelSearcher>(common, SafeThreadPool::FactoryDefaultThreadPool());
    auto basic = std::make_shared<BasicSearcher>(common);

    struct Outcome {
        std::set<std::pair<float, InnerIdType>> values;
        uint32_t early_stops;
    };
    auto run_search = [&](const auto& searcher,
                          const GraphInterfacePtr& graph,
                          float alpha,
                          int64_t threads) {
        InnerSearchParam search_param;
        search_param.ep = 0;
        search_param.ef = 40;
        search_param.topk = 20;
        search_param.rerank_topk = 10;
        search_param.parallel_search_thread_count = threads;
        SearchStatistics stats;
        QueryContext context{.alloc = allocator.get(), .stats = &stats};
        context.distance_early_stop_alpha = alpha;
        auto visited_list = pool->TakeOne();
        const auto* query = base_vectors.data() + dim;
        auto result =
            alpha > 0.0F
                ? searcher->SearchWithDistanceEarlyStop(
                      graph, flatten, visited_list, query, search_param, LabelTablePtr{}, &context)
                : searcher->Search(
                      graph, flatten, visited_list, query, search_param, LabelTablePtr{}, &context);
        pool->ReturnOne(visited_list);
        Outcome outcome{{}, stats.early_stop_count.load()};
        while (not result->Empty()) {
            outcome.values.insert(result->Top());
            result->Pop();
        }
        return outcome;
    };

    uint32_t total_stops = 0;
    for (const auto& graph : {MakeRingGraph(base_size, 8), MakeIrregularGraph(base_size)}) {
        for (const float alpha : {1.0F, 1.05F, 1.5F, 1e30F}) {
            CAPTURE(alpha);
            const auto expected = run_search(basic, graph, alpha, 1);
            const auto actual = run_search(parallel, graph, alpha, 4);
            REQUIRE(actual.values == expected.values);
            REQUIRE(actual.early_stops == expected.early_stops);
            total_stops += actual.early_stops;
        }
        REQUIRE(run_search(parallel, graph, 1e30F, 4).values ==
                run_search(parallel, graph, 0.0F, 4).values);
    }
    REQUIRE(total_stops > 0);
}

namespace {
// Accepts the IDs from first_valid on and reports no valid vectors, so a search with skip_ratio 1
// never scores a rejected neighbour while expanding and defers it instead.
class AcceptFromFilter : public Filter {
public:
    explicit AcceptFromFilter(int64_t first_valid) : first_valid_(first_valid) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return id >= first_valid_;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return 0.0F;
    }

private:
    int64_t first_valid_;
};
}  // namespace

TEST_CASE("Distance early stop does not count rescued filter bridges as results",
          "[ut][BasicSearcher][ParallelSearcher][distance_early_stop]") {
    // Only 4 and 5 pass the filter, and they lie far from the query. Expanding a node always
    // scores its first unvisited neighbour, so the entry point's first neighbour 1 is a rejected
    // dead end; its other neighbours 2 and 3 are rejected bridges to 4 and 5. The traversal defers
    // the bridges, runs dry without a result and then rescues them as navigation-only candidates.
    // If a bridge were recorded as a result, alpha = 1 with k = 1 would end the search at the
    // first bridge, before any eligible point is reached.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 1;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto quantizer_param =
        QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
    auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
        quantizer_param, io_param, common);
    const std::vector<float> vectors{0.5F, 0.7F, 1.0F, 1.5F, 10.0F, 11.0F};
    std::vector<InnerIdType> ids{0, 1, 2, 3, 4, 5};
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());
    auto graph = std::make_shared<MockGraphDataCell>(
        std::vector<std::vector<InnerIdType>>{{1, 2, 3}, {}, {4}, {5}, {}, {}});
    auto pool = std::make_shared<VisitedListPool>(1, allocator.get(), ids.size(), allocator.get());
    auto filter = std::make_shared<AcceptFromFilter>(4);
    auto basic = std::make_shared<BasicSearcher>(common);
    auto parallel =
        std::make_shared<ParallelSearcher>(common, SafeThreadPool::FactoryDefaultThreadPool());

    auto run_search = [&](const auto& searcher,
                          int64_t threads,
                          uint64_t k,
                          uint64_t min_results,
                          float alpha) {
        InnerSearchParam search_param;
        search_param.ep = 0;
        search_param.ef = 4;
        search_param.topk = 4;
        search_param.rerank_topk = static_cast<int64_t>(k);
        search_param.min_results = min_results;
        search_param.skip_ratio = 1.0F;
        search_param.is_inner_id_allowed = filter;
        search_param.parallel_search_thread_count = threads;
        SearchStatistics stats;
        QueryContext context{.alloc = allocator.get(), .stats = &stats};
        context.distance_early_stop_alpha = alpha;
        auto visited_list = pool->TakeOne();
        const float query = 0.0F;
        auto result =
            alpha > 0.0F
                ? searcher->SearchWithDistanceEarlyStop(
                      graph, flatten, visited_list, &query, search_param, LabelTablePtr{}, &context)
                : searcher->Search(graph,
                                   flatten,
                                   visited_list,
                                   &query,
                                   search_param,
                                   LabelTablePtr{},
                                   &context);
        pool->ReturnOne(visited_list);
        std::set<InnerIdType> found;
        while (not result->Empty()) {
            found.insert(result->Top().second);
            result->Pop();
        }
        return found;
    };

    const std::set<InnerIdType> eligible{4, 5};
    for (const uint64_t k : {1, 2}) {
        CAPTURE(k);
        // Without the rescue no eligible point is reachable.
        REQUIRE(run_search(basic, 1, k, 0, 0.0F).empty());
        REQUIRE(run_search(parallel, 2, k, 0, 0.0F).empty());
        REQUIRE(run_search(basic, 1, k, k, 0.0F) == eligible);
        REQUIRE(run_search(parallel, 2, k, k, 0.0F) == eligible);
        REQUIRE(run_search(basic, 1, k, k, 1.0F) == eligible);
        REQUIRE(run_search(parallel, 2, k, k, 1.0F) == eligible);
    }
}
