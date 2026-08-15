
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

#include "basic_searcher.h"

#include <set>
#include <vector>

#include "datacell/flatten_interface.h"
#include "datacell/graph_datacell_parameter.h"
#include "impl/filter/black_list_filter.h"
#include "impl/filter/iterator_filter.h"
#include "searcher_test.h"
#include "unittest.h"
#include "utils/visited_list.h"
using namespace vsag;

TEST_CASE("BasicSearcher supports KNN, range, filters, and empty data cells",
          "[ut][BasicSearcher]") {
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
    auto flatten =
        std::make_shared<FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            quantizer_param, io_param, common);
    flatten->SetQuantizer(
        std::make_shared<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>>(1, allocator.get()));
    flatten->SetIO(std::make_unique<MemoryIO>(allocator.get()));
    std::vector<float> vectors = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    std::vector<InnerIdType> ids = {0, 1, 2, 3, 4, 5};
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());

    auto graph = std::make_shared<MockGraphDataCell>(
        std::vector<std::vector<InnerIdType>>{{1, 2, 3, 4, 5}, {}, {}, {}, {}, {}});
    auto pool = std::make_shared<VisitedListPool>(1, allocator.get(), ids.size(), allocator.get());
    BasicSearcher searcher(common);
    float query = 0.0F;

    auto search = [&](InnerSearchMode mode, const FilterPtr& filter) {
        InnerSearchParam param;
        param.ep = 0;
        param.ef = ids.size();
        param.topk = 3;
        param.radius = 4.0F;
        param.search_mode = mode;
        param.is_inner_id_allowed = filter;
        auto vl = pool->TakeOne();
        QueryContext* ctx = nullptr;
        auto result = searcher.Search(graph, flatten, vl, &query, param, LabelTablePtr{}, ctx);
        pool->ReturnOne(vl);
        std::set<InnerIdType> result_ids;
        while (not result->Empty()) {
            result_ids.insert(result->Top().second);
            result->Pop();
        }
        return result_ids;
    };

    REQUIRE(search(KNN_SEARCH, nullptr) == std::set<InnerIdType>{0, 1, 2});
    REQUIRE(search(RANGE_SEARCH, nullptr) == std::set<InnerIdType>{0, 1, 2});

    auto filter =
        std::make_shared<BlackListFilter>([](LabelType id) -> bool { return id % 2 == 0; });
    REQUIRE(search(KNN_SEARCH, filter) == std::set<InnerIdType>{1, 3, 5});
    REQUIRE(search(RANGE_SEARCH, filter) == std::set<InnerIdType>{1});

    InnerSearchParam param;
    auto vl = pool->TakeOne();
    QueryContext* ctx = nullptr;
    REQUIRE(searcher.Search(graph, nullptr, vl, &query, param, LabelTablePtr{}, ctx)->Empty());
    REQUIRE(searcher.Search(nullptr, flatten, vl, &query, param, LabelTablePtr{}, ctx)->Empty());
    pool->ReturnOne(vl);
}

TEST_CASE("BasicSearcher returns valid heaps from all search entry points with missing inputs",
          "[ut][BasicSearcher][search_impl]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    BasicSearcher searcher(common);

    InnerSearchParam param;
    auto flatten_result =
        searcher.Search(nullptr, nullptr, nullptr, nullptr, param, LabelTablePtr{}, nullptr);
    REQUIRE(flatten_result != nullptr);
    REQUIRE(flatten_result->Empty());

    auto iterator_result = searcher.Search(nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           param,
                                           static_cast<IteratorFilterContext*>(nullptr),
                                           nullptr);
    REQUIRE(iterator_result != nullptr);
    REQUIRE(iterator_result->Empty());

    FlattenDistanceProvider distance_provider(nullptr, nullptr);
    auto graph_result =
        searcher.Search(nullptr, distance_provider, nullptr, param, nullptr, nullptr);
    REQUIRE(graph_result != nullptr);
    REQUIRE(graph_result->Empty());
}

TEST_CASE("Search with stored vector ID", "[ut][BasicSearcher][DistanceProvider]") {
    constexpr uint64_t dim = 8;
    constexpr InnerIdType count = 4;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;

    constexpr const char* param_temp = R"({{"type": "{}"}})";
    auto fp32_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_temp, "fp32")));
    auto io_param =
        IOParameter::GetIOParameterByJson(JsonType::Parse(fmt::format(param_temp, "memory_io")));
    auto flatten =
        std::make_shared<FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            fp32_param, io_param, common);

    const std::vector<float> vectors = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    };
    flatten->Train(vectors.data(), count);
    flatten->Resize(count);
    InnerIdType insert_ids[count] = {0, 1, 2, 3};
    flatten->BatchInsertVector(vectors.data(), count, insert_ids);

    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = count - 1;
    auto graph = GraphInterface::MakeInstance(graph_param, common);
    graph->Resize(count);
    graph->InsertNeighborsById(0, Vector<InnerIdType>({1}, allocator.get()));
    graph->InsertNeighborsById(1, Vector<InnerIdType>({0, 2}, allocator.get()));
    graph->InsertNeighborsById(2, Vector<InnerIdType>({1, 3}, allocator.get()));
    graph->InsertNeighborsById(3, Vector<InnerIdType>({2}, allocator.get()));

    FlattenIdDistanceProvider provider(flatten, 0);
    REQUIRE(provider.QueryDistance(1) == 1.0F);
    REQUIRE(provider.PairwiseDistance(1, 3) == 4.0F);
    REQUIRE_THROWS(provider.FactoryComputerById(0));
    provider.Prefetch(1);

    std::vector<InnerIdType> ids = {0, 1, 2, 3};
    std::vector<float> distances(count);
    provider.BatchQueryDistance(distances.data(), ids.data(), count);
    REQUIRE(distances == std::vector<float>{0.0F, 1.0F, 4.0F, 9.0F});

    InnerSearchParam search_param;
    search_param.ep = 0;
    search_param.ef = count;
    search_param.topk = count;
    search_param.search_mode = KNN_SEARCH;
    auto pool = std::make_shared<VisitedListPool>(1, allocator.get(), count, allocator.get());
    auto vl = pool->TakeOne();
    BasicSearcher searcher(common);
    auto result = searcher.Search(graph, provider, vl, search_param, nullptr, nullptr);
    pool->ReturnOne(vl);
    REQUIRE(result->Size() == count);
}

TEST_CASE("Optimize SQ4", "[ut][BasicOptimizer]") {
    // avoid too much slow task logs
    fixtures::logger::LoggerReplacer _;
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    // data attr
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    uint32_t base_size = 1000;
    uint64_t dim = 128;
    auto quantizer_type = GENERATE("fp32", "sq4_uniform");

    // build and search attr
    uint32_t ef_search = 300;
    uint32_t k = ef_search;
    InnerIdType fixed_entry_point_id = 0;

    // data preparation
    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    // vector data cell
    constexpr const char* param_temp = R"({{"type": "{}"}})";
    auto quantizer_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_temp, quantizer_type)));
    auto io_param =
        IOParameter::GetIOParameterByJson(JsonType::Parse(fmt::format(param_temp, "memory_io")));
    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;

    FlattenInterfacePtr vector_data_cell;
    if (quantizer_type == std::string("sq4_uniform")) {
        vector_data_cell = std::make_shared<
            FlattenDataCell<SQ4UniformQuantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            quantizer_param, io_param, common);
    } else {
        vector_data_cell = std::make_shared<
            FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            quantizer_param, io_param, common);
    }

    vector_data_cell->Train(base_vectors.data(), base_size);
    vector_data_cell->BatchInsertVector(base_vectors.data(), base_size, ids.data());

    auto graph_data_cell = MakeRingGraph(base_size, 8);

    // pool
    auto init_size = 10;
    auto pool = std::make_shared<VisitedListPool>(
        init_size, allocator.get(), vector_data_cell->TotalCount(), allocator.get());

    // search param
    InnerSearchParam search_param;
    search_param.ep = fixed_entry_point_id;
    search_param.ef = ef_search;
    search_param.topk = k;

    // init searcher
    auto searcher = std::make_shared<BasicSearcher>(common);

    // searcher-optimizer
    searcher->SetMockParameters(graph_data_cell, vector_data_cell, pool, search_param, dim, 1000);
    SearchStatistics stats;
    auto loss_before = searcher->MockRun(stats);
    auto optimizer_searcher = std::make_shared<Optimizer<BasicSearcher>>(common);
    optimizer_searcher->RegisterParameter(RuntimeParameter(PREFETCH_DEPTH_CODE, 1, 3, 1));
    optimizer_searcher->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_CODE, 1, 3, 1));
    optimizer_searcher->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_VISIT, 1, 3, 1));
    float end2end_improvement = optimizer_searcher->Optimize(searcher);
    auto loss_after = searcher->MockRun(stats);
}

TEST_CASE("BasicSearcher duplicate threshold keeps nearest owner",
          "[ut][BasicSearcher][duplicate]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 2;
    common.allocator_ = allocator;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;

    constexpr const char* param_temp = R"({{"type": "{}"}})";
    auto quantizer_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_temp, "fp32")));
    auto io_param =
        IOParameter::GetIOParameterByJson(JsonType::Parse(fmt::format(param_temp, "memory_io")));

    auto vector_data_cell = std::make_shared<
        FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
        quantizer_param, io_param, common);
    vector_data_cell->SetQuantizer(
        std::make_shared<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(2, allocator.get()));
    vector_data_cell->SetIO(std::make_unique<MemoryIO>(allocator.get()));

    std::vector<float> base_vectors = {0.0F, 0.0F, 0.3F, 0.0F};
    std::vector<InnerIdType> ids = {0, 1};
    vector_data_cell->Train(base_vectors.data(), ids.size());
    vector_data_cell->BatchInsertVector(base_vectors.data(), ids.size(), ids.data());

    auto graph_data_cell =
        std::make_shared<MockGraphDataCell>(std::vector<std::vector<InnerIdType>>{{1}, {0}});
    auto pool = std::make_shared<VisitedListPool>(
        1, allocator.get(), vector_data_cell->TotalCount(), allocator.get());
    auto searcher = std::make_shared<BasicSearcher>(common);

    auto run_search = [&](const std::vector<float>& query,
                          float threshold,
                          InnerIdType duplicate_query_id =
                              std::numeric_limits<InnerIdType>::max()) {
        InnerSearchParam search_param;
        search_param.ep = 0;
        search_param.ef = 2;
        search_param.topk = 2;
        search_param.find_duplicate = true;
        search_param.duplicate_query_id = duplicate_query_id;
        search_param.duplicate_distance_threshold = threshold;
        auto vl = pool->TakeOne();
        QueryContext* ctx = nullptr;
        auto result = searcher->Search(graph_data_cell,
                                       vector_data_cell,
                                       vl,
                                       query.data(),
                                       search_param,
                                       LabelTablePtr{},
                                       ctx);
        REQUIRE(result->Size() == 2);
        pool->ReturnOne(vl);
        return search_param.duplicate_id;
    };

    REQUIRE(run_search({0.12F, 0.0F}, 0.01F) == -1);
    REQUIRE(run_search({0.12F, 0.0F}, 0.02F) == 0);
    REQUIRE(run_search({0.3F, 0.0F}, 0.0F) == 1);
    REQUIRE(run_search({0.3F, 0.0F}, 0.0F, 0) == -1);
    REQUIRE(run_search({0.3F, 0.0F}, 0.0F, 1) == 1);
}

TEST_CASE("BasicSearcher iterator drain path handles sign and lower_bound correctly",
          "[ut][BasicSearcher][iterator]") {
    // Regression test for the iterator drain path fix:
    //   1. candidate_set Push must use -cur_dist (negative) in drain path.
    //   2. lower_bound must be initialized from top_candidates->Top() after
    //      the drain loop, not left at float::max.
    //   3. top_candidates must be trimmed to ef after drain, with discarded
    //      nodes preserved in iter_ctx for future calls.
    //
    // This test verifies the drain path runs without crash, that returned
    // distances are non-negative (the sign convention invariant), and that
    // multiple iterator calls produce a non-empty accumulated result set.

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    uint64_t dim = 32;
    uint32_t base_size = 200;
    InnerIdType fixed_entry_point_id = 0;

    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    auto graph_data_cell = MakeRingGraph(base_size, 8);

    constexpr const char* param_temp = R"({{"type": "{}"}})";
    auto fp32_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_temp, "fp32")));
    auto io_param =
        IOParameter::GetIOParameterByJson(JsonType::Parse(fmt::format(param_temp, "memory_io")));
    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;

    auto vector_data_cell = std::make_shared<
        FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
        fp32_param, io_param, common);
    vector_data_cell->SetQuantizer(
        std::make_shared<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(dim, allocator.get()));
    vector_data_cell->SetIO(std::make_unique<MemoryIO>(allocator.get()));
    vector_data_cell->Train(base_vectors.data(), base_size);
    vector_data_cell->BatchInsertVector(base_vectors.data(), base_size, ids.data());

    auto pool = std::make_shared<VisitedListPool>(
        1, allocator.get(), vector_data_cell->TotalCount(), allocator.get());

    auto searcher = std::make_shared<BasicSearcher>(common);

    const uint32_t ef = 8;
    const uint32_t topk = 3;
    auto query = base_vectors.data();

    auto* iter_ctx = new IteratorFilterContext();
    iter_ctx->init(vector_data_cell->TotalCount(), ef, allocator.get());

    // First call: exercises the normal entry path
    {
        InnerSearchParam param;
        param.ep = fixed_entry_point_id;
        param.ef = ef;
        param.topk = topk;

        auto vl = pool->TakeOne();
        QueryContext* ctx = nullptr;
        auto result =
            searcher->Search(graph_data_cell, vector_data_cell, vl, query, param, iter_ctx, ctx);
        pool->ReturnOne(vl);

        REQUIRE(result != nullptr);
        // Verify distance sign invariant: top_candidates stores positive distances.
        while (result->Size() > 0) {
            auto [dist, id] = result->Top();
            REQUIRE(dist >= 0.0F);
            REQUIRE(id < base_size);
            result->Pop();
        }
    }

    iter_ctx->SetOFFFirstUsed();

    // Second call: exercises the drain path (the code path fixed by this PR).
    // Must not crash, must return valid results with positive distances.
    {
        InnerSearchParam param;
        param.ep = fixed_entry_point_id;
        param.ef = ef;
        param.topk = topk;

        auto vl = pool->TakeOne();
        QueryContext* ctx = nullptr;
        auto result =
            searcher->Search(graph_data_cell, vector_data_cell, vl, query, param, iter_ctx, ctx);
        pool->ReturnOne(vl);

        REQUIRE(result != nullptr);
        uint32_t count = 0;
        while (result->Size() > 0) {
            auto [dist, id] = result->Top();
            REQUIRE(dist >= 0.0F);  // sign invariant
            REQUIRE(id < base_size);
            count++;
            result->Pop();
        }
        // The drain path must produce at least one candidate (otherwise the
        // fix was not exercised — iter_ctx should have stored some nodes).
        REQUIRE(count > 0);
    }

    delete iter_ctx;
}

TEST_CASE("BasicSearcher traverses through a non-finite-distance bridge",
          "[ut][BasicSearcher][nonfinite]") {
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
    auto flatten =
        std::make_shared<FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
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
    float query = 0.0F;
    auto vl = pool->TakeOne();
    QueryContext* ctx = nullptr;
    auto result =
        BasicSearcher(common).Search(graph, flatten, vl, &query, param, LabelTablePtr{}, ctx);
    pool->ReturnOne(vl);

    REQUIRE(result->Size() <= param.ef);
    bool found_target = false;
    while (not result->Empty()) {
        found_target = found_target or result->Top().second == 5;
        result->Pop();
    }
    REQUIRE(found_target);
}
