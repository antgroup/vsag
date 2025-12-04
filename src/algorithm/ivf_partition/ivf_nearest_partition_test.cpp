
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

#include "ivf_nearest_partition.h"

#include <catch2/catch_test_macros.hpp>

#include "algorithm/inner_index_interface.h"
#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/inner_search_param.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "storage/serialization_template_test.h"

using namespace vsag;

TEST_CASE("IVF Nearest Partition Basic Test", "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto thread_pool = SafeThreadPool::FactoryDefaultThreadPool();
    std::vector<SafeThreadPoolPtr> pools{thread_pool, nullptr};
    int64_t dim = 128;
    int64_t bucket_count = 20;
    for (auto& tp : pools) {
        IndexCommonParam param;
        param.dim_ = 128;
        param.metric_ = MetricType::METRIC_TYPE_L2SQR;
        param.allocator_ = allocator;
        param.thread_pool_ = tp;

        IVFPartitionStrategyParametersPtr strategy_param =
            std::make_shared<IVFPartitionStrategyParameters>();
        auto partition = std::make_unique<IVFNearestPartition>(bucket_count, param, strategy_param);

        auto dataset = Dataset::Make();
        int64_t data_count = 1000L;
        auto vec = fixtures::generate_vectors(data_count, dim, true, 95);
        dataset->Float32Vectors(vec.data())->Dim(dim)->NumElements(data_count)->Owner(false);

        partition->Train(dataset);
        Statistics stats;
        auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1, stats);
        REQUIRE(class_result.size() == data_count);

        auto index = partition->route_index_ptr_;
        std::string route_search_param = R"(
        {
            "hgraph": {
                "ef_search": 20
            }
        }
        )";
        FilterPtr filter = nullptr;
        for (int64_t i = 0; i < data_count; ++i) {
            auto query = Dataset::Make();
            query->Dim(dim)->Float32Vectors(vec.data() + i * dim)->NumElements(1)->Owner(false);
            auto result = index->KnnSearch(query, 1, route_search_param, filter);
            auto id = result->GetIds()[0];
            REQUIRE(id == class_result[i]);
        }
    }
}

TEST_CASE("IVF Nearest Partition Serialize Test", "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t dim = 128;
    int64_t bucket_count = 20;
    IndexCommonParam param;
    param.dim_ = 128;
    param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    param.allocator_ = allocator;
    IVFPartitionStrategyParametersPtr strategy_param =
        std::make_shared<IVFPartitionStrategyParameters>();
    auto partition = std::make_unique<IVFNearestPartition>(bucket_count, param, strategy_param);

    auto dataset = Dataset::Make();
    int64_t data_count = 1000L;
    auto vec = fixtures::generate_vectors(data_count, dim, true, 95);
    dataset->Float32Vectors(vec.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    partition->Train(dataset);
    Statistics stats;
    auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1, stats);
    REQUIRE(class_result.size() == data_count);

    auto partition2 = std::make_unique<IVFNearestPartition>(bucket_count, param, strategy_param);
    test_serializion(*partition, *partition2);

    auto index = partition2->route_index_ptr_;
    std::string route_search_param = R"(
    {
        "hgraph": {
            "ef_search": 20
        }
    }
    )";
    FilterPtr filter = nullptr;

    for (int64_t i = 0; i < data_count; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim)->Float32Vectors(vec.data() + i * dim)->NumElements(1)->Owner(false);
        auto result = index->KnnSearch(query, 1, route_search_param, filter);
        auto id = result->GetIds()[0];
        REQUIRE(id == class_result[i]);
    }
}
