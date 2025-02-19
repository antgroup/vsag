
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

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "ivf_nearest_partition.h"
#include "safe_allocator.h"

using namespace vsag;

TEST_CASE("IVF Nearest Classify Basic Test", "[ut][IVFNearestClassify]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t dim = 128;
    int64_t bucket_count = 20;
    auto classify = std::make_unique<IVFNearestPartition>(
        allocator.get(), bucket_count, dim, IVFNearestPartitionTrainerType::KMeansTrainer);

    auto dataset = Dataset::Make();
    int64_t data_count = 1000L;
    auto vec = fixtures::generate_vectors(data_count, dim, true, 95);
    dataset->Float32Vectors(vec.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    classify->Train(dataset);
    auto class_result = classify->ClassifyDatas(vec.data(), data_count, 1);
    REQUIRE(class_result.size() == data_count);

    auto index = classify->route_index_ptr_;
    std::string route_search_param = R"(
    {{
        "hnsw": {{
            "ef_search": 20
        }}
    }}
    )";

    for (int64_t i = 0; i < data_count; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim)->Float32Vectors(vec.data() + i * dim)->NumElements(1)->Owner(false);
        auto result = index->KnnSearch(query, 1, route_search_param);
        REQUIRE(result.has_value());
        auto id = result.value()->GetIds()[0];
        REQUIRE(id == class_result[i]);
    }
}

TEST_CASE("IVF Nearest Classify Serialize Test", "[ut][IVFNearestClassify]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t dim = 128;
    int64_t bucket_count = 20;
    auto classify = std::make_unique<IVFNearestPartition>(
        allocator.get(), bucket_count, dim, IVFNearestPartitionTrainerType::KMeansTrainer);

    auto dataset = Dataset::Make();
    int64_t data_count = 1000L;
    auto vec = fixtures::generate_vectors(data_count, dim, true, 95);
    dataset->Float32Vectors(vec.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    classify->Train(dataset);
    auto class_result = classify->ClassifyDatas(vec.data(), data_count, 1);
    REQUIRE(class_result.size() == data_count);

    auto dir = fixtures::TempDir("serialize");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path, std::ios::out | std::ios::binary);
    IOStreamWriter writer(outfile);
    classify->Serialize(writer);
    outfile.close();
    classify = std::make_unique<IVFNearestPartition>(
        allocator.get(), bucket_count, dim, IVFNearestPartitionTrainerType::KMeansTrainer);

    std::ifstream infile(path, std::ios::in | std::ios::binary);
    IOStreamReader reader(infile);
    classify->Deserialize(reader);
    infile.close();

    auto index = classify->route_index_ptr_;
    std::string route_search_param = R"(
    {{
        "hnsw": {{
            "ef_search": 20
        }}
    }}
    )";

    for (int64_t i = 0; i < data_count; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim)->Float32Vectors(vec.data() + i * dim)->NumElements(1)->Owner(false);
        auto result = index->KnnSearch(query, 1, route_search_param);
        REQUIRE(result.has_value());
        auto id = result.value()->GetIds()[0];
        REQUIRE(id == class_result[i]);
    }
}
