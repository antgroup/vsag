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
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "vsag/vsag.h"

namespace vsag {

static DatasetPtr
CreateFP16Dataset(int64_t dim, int64_t count, const std::vector<float>& fp32_data) {
    auto* fp16_data = new uint16_t[dim * count];
    for (int64_t i = 0; i < dim * count; ++i) {
        fp16_data[i] = generic::FloatToFP16(fp32_data[i]);
    }

    auto* ids = new int64_t[count];
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
    }

    auto dataset = Dataset::Make();
    dataset->NumElements(count)->Dim(dim)->Ids(ids)->Float16Vectors(fp16_data)->Owner(true);
    return dataset;
}

static DatasetPtr
CreateBF16Dataset(int64_t dim, int64_t count, const std::vector<float>& fp32_data) {
    auto* bf16_data = new uint16_t[dim * count];
    for (int64_t i = 0; i < dim * count; ++i) {
        bf16_data[i] = generic::FloatToBF16(fp32_data[i]);
    }

    auto* ids = new int64_t[count];
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
    }

    auto dataset = Dataset::Make();
    dataset->NumElements(count)->Dim(dim)->Ids(ids)->Float16Vectors(bf16_data)->Owner(true);
    return dataset;
}

static std::vector<float>
GenerateRandomFP32Vectors(int64_t dim, int64_t count) {
    std::vector<float> data(dim * count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) {
        v = dist(rng);
    }
    return data;
}

TEST_CASE("FP16 Dataset API Test", "[ut][dataset][fp16]") {
    int64_t dim = 128;
    int64_t count = 100;

    std::vector<uint16_t> data(dim * count, 0x3C00);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
    }

    auto dataset = Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float16Vectors(data.data())
        ->Owner(false);

    REQUIRE(dataset->GetNumElements() == count);
    REQUIRE(dataset->GetDim() == dim);
    REQUIRE(dataset->GetFloat16Vectors() != nullptr);
    REQUIRE(dataset->GetFloat16Vectors()[0] == 0x3C00);
}

TEST_CASE("HGraph with FP16 Test", "[ft][hgraph][fp16]") {
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 100;

    auto param = R"(
    {
        "dtype": "float16",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp16",
            "max_degree": 16,
            "ef_construction": 50
        }
    }
    )";

    auto index = vsag::Factory::CreateIndex("hgraph", param).value();

    auto fp32_data = GenerateRandomFP32Vectors(TEST_DIM, TEST_COUNT);
    auto base = CreateFP16Dataset(TEST_DIM, TEST_COUNT, fp32_data);
    auto build_result = index->Build(base);

    REQUIRE(build_result.has_value());
    auto failed_ids = build_result.value();
    INFO("Failed IDs count: " << failed_ids.size());
    REQUIRE(index->GetNumElements() == TEST_COUNT - failed_ids.size());

    if (index->GetNumElements() > 0) {
        std::vector<uint16_t> query_fp16(TEST_DIM);
        for (int64_t i = 0; i < TEST_DIM; ++i) {
            query_fp16[i] = generic::FloatToFP16(fp32_data[i]);
        }
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(TEST_DIM)->Float16Vectors(query_fp16.data())->Owner(false);

        auto search_param = R"({"hgraph": {"ef_search": 50}})";
        auto search_result = index->KnnSearch(query, 10, search_param);
        REQUIRE(search_result.has_value());
    }
}

TEST_CASE("HGraph with BF16 Test", "[ft][hgraph][bf16]") {
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 100;

    auto param = R"(
    {
        "dtype": "bfloat16",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "bf16",
            "max_degree": 16,
            "ef_construction": 50
        }
    }
    )";

    auto index = vsag::Factory::CreateIndex("hgraph", param).value();

    auto fp32_data = GenerateRandomFP32Vectors(TEST_DIM, TEST_COUNT);
    auto base = CreateBF16Dataset(TEST_DIM, TEST_COUNT, fp32_data);
    auto build_result = index->Build(base);

    REQUIRE(build_result.has_value());
    auto failed_ids = build_result.value();
    INFO("Failed IDs count: " << failed_ids.size());
    REQUIRE(index->GetNumElements() == TEST_COUNT - failed_ids.size());

    if (index->GetNumElements() > 0) {
        std::vector<uint16_t> query_bf16(TEST_DIM);
        for (int64_t i = 0; i < TEST_DIM; ++i) {
            query_bf16[i] = generic::FloatToBF16(fp32_data[i]);
        }
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(TEST_DIM)->Float16Vectors(query_bf16.data())->Owner(false);

        auto search_param = R"({"hgraph": {"ef_search": 50}})";
        auto search_result = index->KnnSearch(query, 10, search_param);
        REQUIRE(search_result.has_value());
    }
}

}  // namespace vsag
