
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

#include "gno_imi_partition.h"

#include <catch2/catch_test_macros.hpp>

#include "algorithm/ivf_parameter.h"
#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/basic_searcher.h"

using namespace vsag;

TEST_CASE("GNO-IMI Partition Basic Test", "[ut][GNOIMIPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t dim = 128;
    IndexCommonParam param;
    param.dim_ = 128;
    param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    param.allocator_ = allocator;
    auto param_str = R"({
        "partition_strategy_type": "gno_imi",
        "ivf_train_type": "kmeans", 
        "gno_imi": {
            "first_order_buckets_count": 10,
            "second_order_buckets_count": 10
        }
    })";
    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto strategy_param = std::make_shared<vsag::IVFPartitionStrategyParameters>();
    strategy_param->FromJson(param_json);
    auto partition = std::make_unique<GNOIMIPartition>(param, strategy_param);

    auto dataset = Dataset::Make();
    int64_t data_count = 10000L;
    auto vec = fixtures::generate_vectors(data_count, dim, true, 95);
    dataset->Float32Vectors(vec.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    partition->Train(dataset);
    auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1);
    REQUIRE(class_result.size() == data_count);

    param_str = R"(
    {
        "ivf": {
            "scan_buckets_count": 1,
            "first_order_scan_ratio": 0.1
        }
    })";
    auto search_param = IVFSearchParameters::FromJson(param_str);
    InnerSearchParam inner_search_param;
    inner_search_param.scan_bucket_size = search_param.scan_buckets_count;
    inner_search_param.first_order_scan_ratio = search_param.first_order_scan_ratio;
    REQUIRE(inner_search_param.scan_bucket_size == 1);
    REQUIRE(inner_search_param.first_order_scan_ratio == 0.1f);
    size_t match_count = 0;
    for (int64_t i = 0; i < data_count; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim)->Float32Vectors(vec.data() + i * dim)->NumElements(1)->Owner(false);
        auto result =
            partition->ClassifyDatasForSearch(vec.data() + i * dim, 1, inner_search_param);
        auto id = result[0];
        if (id == class_result[i]) {
            match_count++;
        }
    }
    std::cout << "match count(first_order_scan_ratio=0.1): " << match_count << std::endl;

    inner_search_param.first_order_scan_ratio = 0.2f;
    match_count = 0;
    for (int64_t i = 0; i < data_count; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim)->Float32Vectors(vec.data() + i * dim)->NumElements(1)->Owner(false);
        auto result =
            partition->ClassifyDatasForSearch(vec.data() + i * dim, 1, inner_search_param);
        auto id = result[0];
        if (id == class_result[i]) {
            match_count++;
        }
    }
    std::cout << "match count(first_order_scan_ratio=0.2): " << match_count << std::endl;

    inner_search_param.first_order_scan_ratio = 1.0f;
    match_count = 0;
    for (int64_t i = 0; i < data_count; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim)->Float32Vectors(vec.data() + i * dim)->NumElements(1)->Owner(false);
        auto result =
            partition->ClassifyDatasForSearch(vec.data() + i * dim, 1, inner_search_param);
        auto id = result[0];
        // REQUIRE(id == class_result[i]);
        if (id == class_result[i]) {
            match_count++;
        }
    }
    REQUIRE(match_count > 9990);
    std::cout << "match count(first_order_scan_ratio=1.0): " << match_count << std::endl;
}

TEST_CASE("GNO-IMI Partition Serialize Test", "[ut][GNOIMIPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t dim = 128;
    IndexCommonParam param;
    param.dim_ = 128;
    param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    param.allocator_ = allocator;
    auto param_str = R"({
        "partition_strategy_type": "gno_imi",
        "ivf_train_type": "kmeans", 
        "gno_imi": {
            "first_order_buckets_count": 10,
            "second_order_buckets_count": 10
        }
    })";
    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto strategy_param = std::make_shared<vsag::IVFPartitionStrategyParameters>();
    strategy_param->FromJson(param_json);
    auto partition = std::make_unique<GNOIMIPartition>(param, strategy_param);

    auto dataset = Dataset::Make();
    int64_t data_count = 10000L;
    auto vec = fixtures::generate_vectors(data_count, dim, true, 95);
    dataset->Float32Vectors(vec.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    partition->Train(dataset);
    auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1);
    REQUIRE(class_result.size() == data_count);

    auto dir = fixtures::TempDir("serialize");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path, std::ios::out | std::ios::binary);
    IOStreamWriter writer(outfile);
    partition->Serialize(writer);
    outfile.close();
    partition = std::make_unique<GNOIMIPartition>(param, strategy_param);

    std::ifstream infile(path, std::ios::in | std::ios::binary);
    IOStreamReader reader(infile);
    partition->Deserialize(reader);
    infile.close();

    param_str = R"(
    {
        "ivf": {
            "scan_buckets_count": 1,
            "first_order_scan_ratio": 1.0
        }
    })";

    auto search_param = IVFSearchParameters::FromJson(param_str);
    InnerSearchParam inner_search_param;
    inner_search_param.scan_bucket_size = search_param.scan_buckets_count;
    inner_search_param.first_order_scan_ratio = search_param.first_order_scan_ratio;

    size_t match_count = 0;
    FilterPtr filter = nullptr;
    for (int64_t i = 0; i < data_count; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim)->Float32Vectors(vec.data() + i * dim)->NumElements(1)->Owner(false);
        auto result =
            partition->ClassifyDatasForSearch(vec.data() + i * dim, 1, inner_search_param);
        auto id = result[0];
        if (id == class_result[i]) {
            match_count++;
        }
    }
    REQUIRE(match_count > 9990);
    std::cout << "match count(first_order_scan_ratio=1.0): " << match_count << std::endl;
}
