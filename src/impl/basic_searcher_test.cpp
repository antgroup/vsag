
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

#include <catch2/generators/catch_generators.hpp>

#include "algorithm/hnswlib/hnswalg.h"
#include "algorithm/hnswlib/space_l2.h"
#include "basic_optimizer.h"
#include "catch2/catch_template_test_macros.hpp"
#include "data_cell/flatten_datacell.h"
#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/scalar_quantization/sq4_uniform_quantizer.h"
#include "test_logger.h"
#include "utils/visited_list.h"

using namespace vsag;

class AdaptGraphDataCell : public GraphInterface {
public:
    AdaptGraphDataCell(std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw) : alg_hnsw_(alg_hnsw){};

    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override {
        return;
    };

    void
    Resize(InnerIdType new_size) override {
        return;
    };

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override {
        int* data = (int*)alg_hnsw_->get_linklist0(id);
        uint32_t size = alg_hnsw_->getListCount((hnswlib::linklistsizeint*)data);
        neighbor_ids.resize(size);
        for (uint32_t i = 0; i < size; i++) {
            neighbor_ids[i] = *(data + i + 1);
        }
    }

    uint32_t
    GetNeighborSize(InnerIdType id) const override {
        int* data = (int*)alg_hnsw_->get_linklist0(id);
        return alg_hnsw_->getListCount((hnswlib::linklistsizeint*)data);
    }

    void
    Prefetch(InnerIdType id, InnerIdType neighbor_i) override {
        int* data = (int*)alg_hnsw_->get_linklist0(id);
        vsag::Prefetch(data + neighbor_i + 1);
    }

    InnerIdType
    MaximumDegree() const override {
        return alg_hnsw_->getMaxDegree();
    }

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw_;
};

TEST_CASE("Basic Usage for GraphDataCell (adapter of hnsw)", "[ut][GraphDataCell]") {
    uint32_t M = 32;
    uint32_t data_size = 1000;
    uint32_t ef_construction = 100;
    uint64_t DEFAULT_MAX_ELEMENT = 1;
    uint64_t dim = 960;
    auto vectors = fixtures::generate_vectors(data_size, dim);
    std::vector<int64_t> ids(data_size);
    std::iota(ids.begin(), ids.end(), 0);

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto io = std::make_shared<MemoryIO>(allocator.get());
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   DEFAULT_MAX_ELEMENT,
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

        int* data = (int*)alg_hnsw->get_linklist0(i);
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
    uint64_t dim = 960;

    // build and search attr
    uint32_t M = 32;
    uint32_t ef_construction = 100;
    uint32_t ef_search = 300;
    uint32_t k = ef_search;
    InnerIdType fixed_entry_point_id = 0;
    uint64_t DEFAULT_MAX_ELEMENT = 1;

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
                                                   DEFAULT_MAX_ELEMENT,
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
        JsonType::parse(fmt::format(param_temp, "fp32")));
    auto io_param =
        IOParameter::GetIOParameterByJson(JsonType::parse(fmt::format(param_temp, "memory_io")));
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
            auto failed_without_vector =
                searcher->Search(graph_data_cell, nullptr, vl, base_vectors.data(), search_param);
            pool->ReturnOne(vl);
            REQUIRE(failed_without_vector->Size() == 0);
        }
        {
            // search with empty vector_data_cell
            auto vl = pool->TakeOne();
            auto failed_without_graph =
                searcher->Search(nullptr, vector_data_cell, vl, base_vectors.data(), search_param);
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
            auto result = searcher->Search(
                graph_data_cell, vector_data_cell, vl, base_vectors.data() + i * dim, search_param);
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
    uint32_t base_size = 10000;
    uint64_t dim = 960;
    auto quantizer_type = GENERATE("fp32", "sq4_uniform");

    // build and search attr
    uint32_t M = 32;
    uint32_t ef_construction = 100;
    uint32_t ef_search = 300;
    uint32_t k = ef_search;
    InnerIdType fixed_entry_point_id = 0;
    uint64_t DEFAULT_MAX_ELEMENT = 1;

    // data preparation
    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    // vector data cell
    constexpr const char* param_temp = R"({{"type": "{}"}})";
    auto quantizer_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::parse(fmt::format(param_temp, quantizer_type)));
    auto io_param =
        IOParameter::GetIOParameterByJson(JsonType::parse(fmt::format(param_temp, "memory_io")));
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
                                                   DEFAULT_MAX_ELEMENT,
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
    auto loss_before = searcher->MockRun();
    auto optimizer_searcher = std::make_shared<Optimizer<BasicSearcher>>(common);
    optimizer_searcher->RegisterParameter(RuntimeParameter(PREFETCH_DEPTH_CODE, 1, 10, 1));
    optimizer_searcher->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_CODE, 1, 10, 1));
    optimizer_searcher->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_VISIT, 1, 10, 1));
    float end2end_improvement = optimizer_searcher->Optimize(searcher);
    auto loss_after = searcher->MockRun();
}
