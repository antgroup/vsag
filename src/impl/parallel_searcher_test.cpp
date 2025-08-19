
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

TEST_CASE("Parallel search with HNSW", "[ut][BasicSearcher]") {
    //data attr
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
        auto searcher = std::make_shared<ParallelSearcher>(common);
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
    float range = 0.35F;
    auto f = std::make_shared<BlackListFilter>(filter_func);
    double latency[20] = {0.0};

    std::vector<int> num_th = {1,2,4,8,12,16};

    for(int th = 0; th < 6; th++){
        std::cout << "========================" << num_th[th] << " threads===========================" << std::endl;

        // search param
        InnerSearchParam search_param_temp;
        search_param_temp.ep = fixed_entry_point_id;
        search_param_temp.ef = ef_search;
        search_param_temp.topk = k;
        search_param_temp.is_inner_id_allowed = nullptr;
        search_param_temp.radius = range;
        search_param_temp.parallel_search_thread_count_per_query = num_th[th];
        search_param_temp.latency = latency;

        std::vector<InnerSearchParam> params(4);
        params[0] = search_param_temp;
        params[1] = search_param_temp;
        params[1].is_inner_id_allowed = f;
        params[2] = search_param_temp;
        params[2].search_mode = RANGE_SEARCH;
        params[3] = params[2];
        params[3].is_inner_id_allowed = f;

        auto search_pool = SafeThreadPool::FactoryDefaultThreadPool();
        search_pool->SetPoolSize(search_param_temp.parallel_search_thread_count_per_query);

        uint32_t a = 0, b = 0, c = 0;
        uint32_t result_size1 = 300;

        double valid_latency = 0.0, blatency = 0.0;

        for (const auto& search_param : params) {
            valid_latency = 0.0;
            blatency = 0.0;
            for(int i = 0; i < 20; i++)
                latency[i] = 0.0;
            exception_func(search_param);
            auto searcher = std::make_shared<ParallelSearcher>(common, nullptr, search_pool);
            for (int i = 0; i < query_size; i++) {
                std::unordered_set<InnerIdType> valid_set, set;
                auto vl = pool->TakeOne();

                auto start_time = std::chrono::high_resolution_clock::now();
                //auto result = searcher->Search(graph_data_cell, vector_data_cell, vl, base_vectors.data() + i * dim, search_param);
                auto result = searcher->Search_plus(graph_data_cell, vector_data_cell, vl, base_vectors.data() + i * dim, search_param);
                auto end_time = std::chrono::high_resolution_clock::now();

                auto search_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                blatency += search_time_us;
                pool->ReturnOne(vl);
                auto result_size = result->Size();
                for (int j = 0; j < result_size; j++) {
                    set.insert(result->Top().second);
                    result->Pop();
                }
                if (search_param.search_mode == KNN_SEARCH) {
                    auto start_time = std::chrono::high_resolution_clock::now();
                    auto valid_result =
                        alg_hnsw->searchBaseLayerST<false, false>(fixed_entry_point_id,
                                                                base_vectors.data() + i * dim,
                                                                ef_search,
                                                                search_param.is_inner_id_allowed);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto search_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                    valid_latency += search_time_us;                                              
                    REQUIRE(result_size == valid_result.size());
                    for (int j = 0; j < result_size; j++) {
                        valid_set.insert(valid_result.top().second);
                        valid_result.pop();
                    }
                } else if (search_param.search_mode == RANGE_SEARCH) {
                    auto start_time = std::chrono::high_resolution_clock::now();
                    auto valid_result =
                        alg_hnsw->searchBaseLayerST<false, false>(fixed_entry_point_id,
                                                                base_vectors.data() + i * dim,
                                                                range,
                                                                ef_search,
                                                                search_param.is_inner_id_allowed);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto search_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                    valid_latency += search_time_us;
                    if(result_size != valid_result.size())
                        c++;
                    //std::cout << "valid_result.size() = " << valid_result.size() << std::endl;
                    for (int j = 0; j < result_size; j++) {
                        valid_set.insert(valid_result.top().second);
                        valid_result.pop();
                    }
                }
                
                for (auto id : set) {
                    if(valid_set.count(id) == 0)
                        a++;
                }
                for (auto id : valid_set) {
                    if(set.count(id) == 0)
                        b++;
                }
            }

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "latency[11] = " << latency[11] / query_size << std::endl;
            std::cout << "并行平均延迟 = " << blatency / query_size / 1000<< std::endl;
            std::cout << "标准平均延迟 = " << valid_latency / query_size / 1000<< std::endl;
            // std::cout << "latency[0] = " << latency[0] / query_size / 1000 << std::endl;
            // std::cout << "latency[1] = " << latency[1] / query_size / 1000 << std::endl;
            // std::cout << "latency[2] = " << latency[2] / query_size / 1000 << std::endl;
            // std::cout << "latency[3] = " << latency[3] / query_size / 1000 << std::endl;
            // std::cout << "latency[4] = " << latency[4] / query_size / 1000 << std::endl;
            // std::cout << "latency[5] = " << latency[5] / query_size / 1000 << std::endl;
            // std::cout << "latency[6] = " << latency[6] / query_size / 1000 << std::endl;
            // std::cout << "latency[7] = " << latency[7] / query_size / 1000 << std::endl;
            // std::cout << "latency[8] = " << latency[8] / query_size / 1000 << std::endl;
            // std::cout << "latency[9] = " << latency[9] / query_size / 1000 << std::endl;
            // std::cout << "latency[10] = " << latency[10] / query_size / 1000 << std::endl;
            // std::cout << "latency[11] = " << latency[11] / query_size << std::endl;
        }
        std::cout << "a = " << a << std::endl;
        std::cout << "b = " << b << std::endl;
        std::cout << "c = " << c << std::endl;

    }

}

TEST_CASE("Parallel search with HNSW used basic_searcher", "[ut][BasicSearcher]") {
    //data attr
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
        auto searcher = std::make_shared<ParallelSearcher>(common);
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
    float range = 0.35F;
    auto f = std::make_shared<BlackListFilter>(filter_func);

    double latency[20] = {0.0};

    // search param
    InnerSearchParam search_param_temp;
    search_param_temp.ep = fixed_entry_point_id;
    search_param_temp.ef = ef_search;
    search_param_temp.topk = k;
    search_param_temp.is_inner_id_allowed = nullptr;
    search_param_temp.radius = range;
    search_param_temp.parallel_search_thread_count_per_query = 1;
    search_param_temp.latency = latency;

    std::vector<InnerSearchParam> params(4);
    params[0] = search_param_temp;
    params[1] = search_param_temp;
    params[1].is_inner_id_allowed = f;
    params[2] = search_param_temp;
    params[2].search_mode = RANGE_SEARCH;
    params[3] = params[2];
    params[3].is_inner_id_allowed = f;

    auto search_pool = SafeThreadPool::FactoryDefaultThreadPool();
    search_pool->SetPoolSize(search_param_temp.parallel_search_thread_count_per_query);

    double valid_latency = 0.0, parallel_latency = 0.0, basic_latency = 0.0;

    for (const auto& search_param : params) {
        valid_latency = 0.0;
        parallel_latency = 0.0;
        basic_latency = 0.0;
        for(int i = 0; i < 20; i++)
            latency[i] = 0.0;
        exception_func(search_param);
        auto parallel_searcher = std::make_shared<ParallelSearcher>(common, nullptr, search_pool);
        auto basic_searcher = std::make_shared<BasicSearcher>(common);

        for (int i = 0; i < query_size; i++) {
            std::unordered_set<InnerIdType> valid_set, basic_set;
            auto pvl = pool->TakeOne();
            auto bvl = pool->TakeOne();

            auto pstart_time = std::chrono::high_resolution_clock::now();
            auto parallel_result = parallel_searcher->Search(graph_data_cell, vector_data_cell, pvl, base_vectors.data() + i * dim, search_param);
            auto pend_time = std::chrono::high_resolution_clock::now();

            auto psearch_time_us = std::chrono::duration_cast<std::chrono::microseconds>(pend_time - pstart_time).count();
            parallel_latency += psearch_time_us;
            pool->ReturnOne(pvl);
            auto result_size = parallel_result->Size();
            for (int j = 0; j < result_size; j++) {
                basic_set.insert(parallel_result->Top().second);
                parallel_result->Pop();
            }

            auto bstart_time = std::chrono::high_resolution_clock::now();
            auto basic_result = basic_searcher->Search(graph_data_cell, vector_data_cell, bvl, base_vectors.data() + i * dim, search_param);
            auto bend_time = std::chrono::high_resolution_clock::now();

            auto bsearch_time_us = std::chrono::duration_cast<std::chrono::microseconds>(bend_time - bstart_time).count();
            basic_latency += bsearch_time_us;
            pool->ReturnOne(bvl);
            
            if (search_param.search_mode == KNN_SEARCH) {
                auto start_time = std::chrono::high_resolution_clock::now();
                auto valid_result =
                    alg_hnsw->searchBaseLayerST<false, false>(fixed_entry_point_id,
                                                              base_vectors.data() + i * dim,
                                                              ef_search,
                                                              search_param.is_inner_id_allowed);
                auto end_time = std::chrono::high_resolution_clock::now();
                auto search_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                valid_latency += search_time_us;                                              
                REQUIRE(result_size == valid_result.size());
                for (int j = 0; j < result_size; j++) {
                    valid_set.insert(valid_result.top().second);
                    valid_result.pop();
                }
            } else if (search_param.search_mode == RANGE_SEARCH) {
                auto start_time = std::chrono::high_resolution_clock::now();
                auto valid_result =
                    alg_hnsw->searchBaseLayerST<false, false>(fixed_entry_point_id,
                                                              base_vectors.data() + i * dim,
                                                              range,
                                                              ef_search,
                                                              search_param.is_inner_id_allowed);
                auto end_time = std::chrono::high_resolution_clock::now();
                auto search_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                valid_latency += search_time_us;
                REQUIRE(result_size == valid_result.size());
                for (int j = 0; j < result_size; j++) {
                    valid_set.insert(valid_result.top().second);
                    valid_result.pop();
                }
            }
            
            for (auto id : basic_set) {
                REQUIRE(valid_set.count(id) > 0);
            }
            for (auto id : valid_set) {
                REQUIRE(basic_set.count(id) > 0);
            }
        }
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "并行平均延迟 = " << parallel_latency / query_size / 1000 << std::endl;
        std::cout << "基础平均延迟 = " << basic_latency / query_size / 1000 << std::endl;
        std::cout << "标准平均延迟 = " << valid_latency / query_size / 1000 << std::endl;
        std::cout << "latency[0] = " << latency[0] / query_size / 1000 << std::endl;
        std::cout << "latency[1] = " << latency[1] / query_size / 1000 << std::endl;
        std::cout << "latency[2] = " << latency[2] / query_size / 1000 << std::endl;
        std::cout << "latency[3] = " << latency[3] / query_size / 1000 << std::endl;
        std::cout << "latency[4] = " << latency[4] / query_size / 1000 << std::endl;
        std::cout << "latency[5] = " << latency[5] / query_size / 1000 << std::endl;
        std::cout << "latency[6] = " << latency[6] / query_size / 1000 << std::endl;
        std::cout << "latency[7] = " << latency[7] / query_size / 1000 << std::endl;
        std::cout << "latency[8] = " << latency[8] / query_size / 1000 << std::endl;
        std::cout << "latency[9] = " << latency[9] / query_size / 1000 << std::endl;
        std::cout << "latency[10] = " << latency[10] / query_size / 1000 << std::endl;
    }

}
