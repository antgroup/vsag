
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

#include "algorithm/inner_index_interface.h"
#include "datacell/flatten_interface.h"
#include "searcher_test.h"
#include "unittest.h"
#include "utils/visited_list.h"
using namespace vsag;

TEST_CASE("Basic Usage for GraphDataCell (adapter of hnsw)", "[ut][GraphDataCell]") {
    uint32_t M = 32;
    uint32_t data_size = 1000;
    uint32_t ef_construction = 100;
    uint64_t default_max_element = 1;
    uint64_t dim = 960;
    auto vectors = fixtures::generate_vectors(data_size, dim);
    std::vector<int64_t> ids(data_size);
    std::iota(ids.begin(), ids.end(), 0);

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto io = std::make_shared<MemoryIO>(allocator.get());
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   default_max_element,
                                                   allocator.get(),
                                                   M / 2,
                                                   ef_construction,
                                                   Options::Instance().block_size_limit());
    alg_hnsw->init_memory_space();
    for (int64_t i = 0; i < data_size; ++i) {
        auto successful_insert =
            alg_hnsw->addPoint((const void*)(vectors.data() + i * dim), ids[i]);
        REQUIRE(successful_insert == true);
    }

    GraphInterfacePtr graph = std::make_shared<AdaptGraphDataCell>(alg_hnsw);

    for (uint32_t i = 0; i < data_size; i++) {
        auto neighbor_size = graph->GetNeighborSize(i);
        Vector<InnerIdType> neighbor_ids(neighbor_size, allocator.get());
        graph->GetNeighbors(i, neighbor_ids);

        auto* data = alg_hnsw->get_linklist0(i);
        REQUIRE(neighbor_size == alg_hnsw->getListCount((hnswlib::linklistsizeint*)data));

        for (uint32_t j = 0; j < neighbor_size; j++) {
            REQUIRE(neighbor_ids[j] == *(data + j + 1));
        }
    }
}

TEST_CASE("Search with HNSW", "[ut][BasicSearcher]") {
    // data attr
    uint32_t base_size = 1000;
    uint32_t query_size = 100;
    uint64_t dim = 128;

    // build and search attr
    uint32_t M = 16;
    uint32_t ef_construction = 100;
    uint32_t ef_search = 300;
    uint32_t k = ef_search;
    InnerIdType fixed_entry_point_id = 0;
    uint64_t default_max_element = 1;

    // data preparation
    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    // hnswlib build
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto io = std::make_shared<MemoryIO>(allocator.get());
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   default_max_element,
                                                   allocator.get(),
                                                   M / 2,
                                                   ef_construction,
                                                   Options::Instance().block_size_limit());
    alg_hnsw->init_memory_space();
    for (int64_t i = 0; i < base_size; ++i) {
        auto successful_insert =
            alg_hnsw->addPoint((const void*)(base_vectors.data() + i * dim), ids[i]);
        REQUIRE(successful_insert == true);
    }

    // graph data cell
    auto graph_data_cell = std::make_shared<AdaptGraphDataCell>(alg_hnsw);

    // vector data cell
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

    auto init_size = 10;
    auto pool = std::make_shared<VisitedListPool>(
        init_size, allocator.get(), vector_data_cell->TotalCount(), allocator.get());

    auto exception_func = [&](const InnerSearchParam& search_param) -> void {
        // init searcher
        auto searcher = std::make_shared<BasicSearcher>(common);
        {
            // search with empty graph_data_cell
            auto vl = pool->TakeOne();
            QueryContext* qctx = nullptr;
            auto failed_without_vector = searcher->Search(
                graph_data_cell, nullptr, vl, base_vectors.data(), search_param, nullptr, qctx);
            pool->ReturnOne(vl);
            REQUIRE(failed_without_vector->Size() == 0);
        }
        {
            // search with empty vector_data_cell
            auto vl = pool->TakeOne();
            QueryContext* qctx = nullptr;
            auto failed_without_graph = searcher->Search(
                nullptr, vector_data_cell, vl, base_vectors.data(), search_param, nullptr, qctx);
            pool->ReturnOne(vl);
            REQUIRE(failed_without_graph->Size() == 0);
        }
    };

    auto filter_func = [](LabelType id) -> bool { return id % 2 == 0; };
    float range = 0.1F;
    auto f = std::make_shared<BlackListFilter>(filter_func);

    // search param
    InnerSearchParam search_param_temp;
    search_param_temp.ep = fixed_entry_point_id;
    search_param_temp.ef = ef_search;
    search_param_temp.topk = k;
    search_param_temp.is_inner_id_allowed = nullptr;
    search_param_temp.radius = range;

    std::vector<InnerSearchParam> params(4);
    params[0] = search_param_temp;
    params[1] = search_param_temp;
    params[1].is_inner_id_allowed = f;
    params[2] = search_param_temp;
    params[2].search_mode = RANGE_SEARCH;
    params[3] = params[2];
    params[3].is_inner_id_allowed = f;

    for (const auto& search_param : params) {
        exception_func(search_param);
        auto searcher = std::make_shared<BasicSearcher>(common);
        for (int i = 0; i < query_size; i++) {
            std::unordered_set<InnerIdType> valid_set, set;
            auto vl = pool->TakeOne();
            QueryContext* ctx = nullptr;
            auto result = searcher->Search(graph_data_cell,
                                           vector_data_cell,
                                           vl,
                                           base_vectors.data() + i * dim,
                                           search_param,
                                           (LabelTablePtr) nullptr,
                                           ctx);
            pool->ReturnOne(vl);
            auto result_size = result->Size();
            for (int j = 0; j < result_size; j++) {
                set.insert(result->Top().second);
                result->Pop();
            }
            if (search_param.search_mode == KNN_SEARCH) {
                auto valid_result =
                    alg_hnsw->searchBaseLayerST<false, false>(fixed_entry_point_id,
                                                              base_vectors.data() + i * dim,
                                                              ef_search,
                                                              search_param.is_inner_id_allowed);
                REQUIRE(result_size == valid_result.size());
                for (int j = 0; j < result_size; j++) {
                    valid_set.insert(valid_result.top().second);
                    valid_result.pop();
                }
            } else if (search_param.search_mode == RANGE_SEARCH) {
                auto valid_result =
                    alg_hnsw->searchBaseLayerST<false, false>(fixed_entry_point_id,
                                                              base_vectors.data() + i * dim,
                                                              range,
                                                              ef_search,
                                                              search_param.is_inner_id_allowed);
                REQUIRE(result_size == valid_result.size());
                for (int j = 0; j < result_size; j++) {
                    valid_set.insert(valid_result.top().second);
                    valid_result.pop();
                }
            }

            for (auto id : set) {
                REQUIRE(valid_set.count(id) > 0);
            }
            for (auto id : valid_set) {
                REQUIRE(set.count(id) > 0);
            }
        }
    }
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
    uint32_t M = 16;
    uint32_t ef_construction = 100;
    uint32_t ef_search = 300;
    uint32_t k = ef_search;
    InnerIdType fixed_entry_point_id = 0;
    uint64_t default_max_element = 1;

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

    // hnswlib build
    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto io = std::make_shared<MemoryIO>(allocator.get());
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   default_max_element,
                                                   allocator.get(),
                                                   M / 2,
                                                   ef_construction,
                                                   Options::Instance().block_size_limit());
    alg_hnsw->init_memory_space();

    for (int64_t i = 0; i < base_size; ++i) {
        alg_hnsw->addPoint((const void*)(base_vectors.data() + i * dim), ids[i]);
    }

    // graph data cell
    auto graph_data_cell = std::make_shared<AdaptGraphDataCell>(alg_hnsw);

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
    uint32_t M = 16;
    uint32_t ef_construction = 100;
    InnerIdType fixed_entry_point_id = 0;
    uint64_t default_max_element = 1;

    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   default_max_element,
                                                   allocator.get(),
                                                   M / 2,
                                                   ef_construction,
                                                   Options::Instance().block_size_limit());
    alg_hnsw->init_memory_space();
    for (int64_t i = 0; i < static_cast<int64_t>(base_size); ++i) {
        alg_hnsw->addPoint((const void*)(base_vectors.data() + i * dim),
                           static_cast<hnswlib::LabelType>(ids[i]));
    }

    auto graph_data_cell = std::make_shared<AdaptGraphDataCell>(alg_hnsw);

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
