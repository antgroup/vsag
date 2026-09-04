
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

#include <algorithm>
#include <vector>

#include "algorithm/inner_index_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/inner_search_param.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"
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
        auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1, nullptr);
        REQUIRE(class_result.size() == data_count);

        auto index = partition->route_index_ptr_;
        // Match ClassifyDatas so this checks its routing rather than HGraph search breadth.
        std::string route_search_param = R"(
        {
            "hgraph": {
                "ef_search": 10
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
    auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1, nullptr);
    REQUIRE(class_result.size() == data_count);

    auto partition2 = std::make_unique<IVFNearestPartition>(bucket_count, param, strategy_param);
    test_serializion(*partition, *partition2);

    auto restored_class_result = partition2->ClassifyDatas(vec.data(), data_count, 1, nullptr);
    REQUIRE(restored_class_result == class_result);
}

TEST_CASE("IVF Nearest Partition Routing Statistics Test", "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto thread_pool = SafeThreadPool::FactoryDefaultThreadPool();
    std::vector<SafeThreadPoolPtr> pools{thread_pool, nullptr};
    int64_t dim = 128;
    int64_t bucket_count = 20;
    for (auto& tp : pools) {
        IndexCommonParam param;
        param.dim_ = dim;
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
        auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1, nullptr);

        // The search path accumulates routing statistics through a non-null QueryContext. The
        // statistics parsing must produce the same bucket assignment and non-zero routing stats
        // (the parse now runs outside the reduce lock).
        SearchStatistics stats;
        QueryContext ctx;
        ctx.stats = &stats;
        auto stats_result = partition->ClassifyDatas(vec.data(), data_count, 1, &ctx);
        REQUIRE(stats_result == class_result);
        REQUIRE(stats.dist_cmp.load() > 0);
        auto dumped = JsonType::Parse(stats.Dump());
        REQUIRE(dumped["distance_evaluations_by_phase"]["routing"].GetUint64() > 0);
    }
}

TEST_CASE("IVF Nearest Partition Centroid Scan Test", "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 16;
    constexpr int64_t data_count = 256;
    constexpr BucketIdType buckets_per_data = 4;
    const std::vector<MetricType> metrics{
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE};

    for (const auto metric : metrics) {
        IndexCommonParam common_param;
        common_param.dim_ = dim;
        common_param.metric_ = metric;
        common_param.allocator_ = allocator;

        auto strategy_param = std::make_shared<IVFPartitionStrategyParameters>();
        strategy_param->use_route_graph = false;
        auto partition =
            std::make_unique<IVFNearestPartition>(bucket_count, common_param, strategy_param);
        REQUIRE(partition->route_index_ptr_ == nullptr);

        const auto untrained_result =
            partition->ClassifyDatas(nullptr, 1, buckets_per_data, nullptr);
        REQUIRE(untrained_result == Vector<BucketIdType>(buckets_per_data,
                                                         static_cast<BucketIdType>(-1),
                                                         allocator.get()));
        auto untrained_restored =
            std::make_unique<IVFNearestPartition>(bucket_count, common_param, strategy_param);
        test_serializion(*partition, *untrained_restored);
        REQUIRE(untrained_restored->ClassifyDatas(nullptr, 1, buckets_per_data, nullptr) ==
                untrained_result);

        auto vectors = fixtures::generate_vectors(data_count, dim, true, 95);
        auto dataset = Dataset::Make();
        dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);
        partition->Train(dataset);
        REQUIRE(partition->route_index_ptr_ == nullptr);

        auto actual =
            partition->ClassifyDatas(vectors.data(), data_count, buckets_per_data, nullptr);
        REQUIRE(actual.size() == static_cast<uint64_t>(data_count * buckets_per_data));

        Vector<float> centroid(dim, allocator.get());
        Vector<float> normalized_query(dim, allocator.get());
        Vector<std::pair<float, BucketIdType>> expected(bucket_count, allocator.get());
        for (int64_t i = 0; i < data_count; ++i) {
            const auto* query = vectors.data() + i * dim;
            if (metric == MetricType::METRIC_TYPE_COSINE) {
                Normalize(query, normalized_query.data(), dim);
                query = normalized_query.data();
            }
            for (BucketIdType b = 0; b < bucket_count; ++b) {
                partition->GetCentroid(b, centroid);
                float distance = 0.0F;
                if (metric == MetricType::METRIC_TYPE_L2SQR) {
                    distance = FP32ComputeL2Sqr(query, centroid.data(), dim);
                } else {
                    distance = 1.0F - FP32ComputeIP(query, centroid.data(), dim);
                }
                expected[b] = {distance, b};
            }
            std::sort(expected.begin(), expected.end());
            for (BucketIdType j = 0; j < buckets_per_data; ++j) {
                REQUIRE(actual[i * buckets_per_data + j] == expected[j].second);
            }
        }

        auto graph_config_param = std::make_shared<IVFPartitionStrategyParameters>();
        graph_config_param->use_route_graph = true;
        auto restored =
            std::make_unique<IVFNearestPartition>(bucket_count, common_param, graph_config_param);
        test_serializion(*partition, *restored);
        REQUIRE(restored->route_index_ptr_ == nullptr);
        REQUIRE(restored->ClassifyDatas(vectors.data(), data_count, buckets_per_data, nullptr) ==
                actual);
    }
}

TEST_CASE("IVF Nearest Partition Route Graph Layout Cross Config Test",
          "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 16;
    constexpr int64_t data_count = 256;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    auto graph_param = std::make_shared<IVFPartitionStrategyParameters>();
    graph_param->use_route_graph = true;
    auto graph_partition =
        std::make_unique<IVFNearestPartition>(bucket_count, common_param, graph_param);

    auto vectors = fixtures::generate_vectors(data_count, dim, true, 95);
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);
    graph_partition->Train(dataset);
    auto expected = graph_partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);

    auto scan_config_param = std::make_shared<IVFPartitionStrategyParameters>();
    scan_config_param->use_route_graph = false;
    auto restored =
        std::make_unique<IVFNearestPartition>(bucket_count, common_param, scan_config_param);
    REQUIRE(restored->route_index_ptr_ == nullptr);
    test_serializion(*graph_partition, *restored);
    REQUIRE(restored->route_index_ptr_ != nullptr);
    REQUIRE(restored->ClassifyDatas(vectors.data(), data_count, 1, nullptr) == expected);
}
