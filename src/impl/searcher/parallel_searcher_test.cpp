
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

#include <future>
#include <set>
#include <vector>

#include "impl/thread_pool/ring_buffer_thread_pool.h"
#include "query_context.h"
#include "searcher_test.h"
#include "unittest.h"

using namespace vsag;

namespace {

class InlineSearchThreadPool : public ThreadPool {
public:
    std::future<void>
    Enqueue(std::function<void(void)> task) override {
        ++submissions;
        std::packaged_task<void()> packaged_task(std::move(task));
        auto result = packaged_task.get_future();
        packaged_task();
        return result;
    }

    void
    WaitUntilEmpty() override {
    }

    void
    SetQueueSizeLimit(std::uint64_t) override {
    }

    void
    SetPoolSize(std::uint64_t) override {
    }

    uint64_t submissions{0};
};

}  // namespace

TEST_CASE("ParallelSearcher matches BasicSearcher on a generic graph", "[ut][ParallelSearcher]") {
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
    auto parallel =
        std::make_shared<ParallelSearcher>(common, SafeThreadPool::FactoryDefaultThreadPool());
    auto basic = std::make_shared<BasicSearcher>(common);

    InnerSearchParam search_param;
    search_param.ep = 0;
    search_param.ef = 40;
    search_param.topk = 20;
    search_param.parallel_search_thread_count = 4;

    auto run_search =
        [&](const auto& searcher, const GraphInterfacePtr& graph, int expected_fallbacks = -1) {
            auto visited_list = pool->TakeOne();
            SearchMetrics metrics;
            QueryContext query_context;
            query_context.stats = &metrics.base;
            QueryContext* context = expected_fallbacks < 0 ? nullptr : &query_context;
            auto result = searcher->Search(graph,
                                           flatten,
                                           visited_list,
                                           base_vectors.data(),
                                           search_param,
                                           LabelTablePtr{},
                                           context);
            pool->ReturnOne(visited_list);
            if (expected_fallbacks >= 0) {
                REQUIRE(metrics.base.ToJson()["parallel_search_fallback_count"].GetUint64() ==
                        static_cast<uint64_t>(expected_fallbacks));
                REQUIRE(std::move(metrics).Snapshot().parallel_search_fallback_count ==
                        static_cast<uint64_t>(expected_fallbacks));
            }
            std::set<std::pair<float, InnerIdType>> values;
            while (not result->Empty()) {
                values.insert(result->Top());
                result->Pop();
            }
            return values;
        };

    for (const auto& graph : {MakeRingGraph(base_size, 8), MakeIrregularGraph(base_size)}) {
        REQUIRE(run_search(parallel, graph, 0) == run_search(basic, graph));
        REQUIRE(run_search(parallel, graph) == run_search(basic, graph));
    }

    SECTION("same-pool worker queries do not submit polling helpers inline") {
        search_param.parallel_search_thread_count = 2;
        for (const uint64_t workers : {1, 2}) {
            auto raw_pool = std::make_shared<RingBufferThreadPool>(workers);
            auto safe_pool = std::make_shared<SafeThreadPool>(raw_pool);
            auto nested_searcher = std::make_shared<ParallelSearcher>(common, safe_pool);
            for (const auto& graph : {MakeRingGraph(base_size, 8), MakeIrregularGraph(base_size)}) {
                const auto expected = run_search(basic, graph);
                std::set<std::pair<float, InnerIdType>> actual;
                bool requires_inline = false;
                // Submit directly through the underlying pool: SafeThreadPool must detect
                // native worker identity without relying on its own enqueue wrapper.
                auto query = raw_pool->Enqueue([&]() {
                    requires_inline = safe_pool->RequiresInlineExecution();
                    actual = run_search(nested_searcher, graph, 1);
                });
                // The external test runner supplies the deadlock timeout. Always join this
                // future before any captured fixture data can leave scope.
                query.get();
                REQUIRE(requires_inline);
                REQUIRE(actual == expected);
            }
        }
    }

    SECTION("an unknown synchronous executor does not receive polling helpers") {
        search_param.parallel_search_thread_count = 2;
        auto inline_pool = std::make_shared<InlineSearchThreadPool>();
        auto safe_pool = std::make_shared<SafeThreadPool>(inline_pool);
        auto inline_searcher = std::make_shared<ParallelSearcher>(common, safe_pool);
        REQUIRE(safe_pool->RequiresInlineExecution());
        for (const auto& graph : {MakeRingGraph(base_size, 8), MakeIrregularGraph(base_size)}) {
            REQUIRE(run_search(inline_searcher, graph, 1) == run_search(basic, graph));
            search_param.parallel_search_thread_count = 1;
            REQUIRE(run_search(inline_searcher, graph, 0) == run_search(basic, graph));
            search_param.parallel_search_thread_count = 2;
            REQUIRE(run_search(inline_searcher, graph) == run_search(basic, graph));
        }
        REQUIRE(inline_pool->submissions == 0);
    }

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
