
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

#include "searcher_test.h"

using namespace vsag;

TEST_CASE("Parallel search with HNSW", "[ut][ParallelSearcher][concurrent]") {
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

    auto thread_pool = SafeThreadPool::FactoryDefaultThreadPool();
    thread_pool->SetPoolSize(16);

    auto exception_func = [&](const InnerSearchParam& search_param) -> void {
        // init searcher
        auto searcher = std::make_shared<ParallelSearcher>(common, thread_pool);
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
    search_param_temp.use_muti_threads_for_one_query = true;
    search_param_temp.level_0 = true;

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
        auto searcher = std::make_shared<ParallelSearcher>(common, thread_pool);
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

            uint32_t res_in_valid_num = 0;
            uint32_t valid_in_res_num = 0;

            for (auto id : set) {
                if (valid_set.count(id) > 0)
                    res_in_valid_num++;
            }

            for (auto id : valid_set) {
                if (set.count(id) > 0)
                    valid_in_res_num++;
            }
            REQUIRE(res_in_valid_num == valid_in_res_num);
        }
    }
}
