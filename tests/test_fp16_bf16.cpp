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

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "json_types.h"
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

TEST_CASE("HGraph typed native distance contract", "[ft][hgraph][distance_contract]") {
    for (const auto& dtype :
         {std::string("float16"), std::string("bfloat16"), std::string("int8")}) {
        const std::string quant = dtype == "float16"    ? "fp16"
                                  : dtype == "bfloat16" ? "bf16"
                                                        : "int8";
        const std::string params =
            "{\"dtype\":\"" + dtype +
            "\",\"metric_type\":\"l2\",\"dim\":32,\"index_param\":{\"base_quantization_type\":\"" +
            quant + "\",\"max_degree\":16,\"ef_construction\":50}}";
        auto made = vsag::Factory::CreateIndex("hgraph", params);
        REQUIRE(made.has_value());
        auto index = made.value();
        REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID));
        REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_BATCH_CALC_DISTANCE_BY_ID));
        uint16_t half[64] = {};
        int8_t ints[64] = {};
        float wrong[32] = {};
        int64_t labels[] = {11, 22};
        auto base = vsag::Dataset::Make()->Owner(false)->Dim(32)->NumElements(2)->Ids(labels);
        if (dtype == "int8")
            base->Int8Vectors(ints);
        else
            base->Float16Vectors(half);
        REQUIRE(index->Build(base).has_value());
        auto single = vsag::Dataset::Make()->Owner(false)->Dim(32)->NumElements(1);
        if (dtype == "int8")
            single->Int8Vectors(ints);
        else
            single->Float16Vectors(half);
        auto distance = index->CalcDistanceById(single, 11);
        REQUIRE(distance.has_value());
        REQUIRE(distance.value() == 0);
        REQUIRE(!index->CalcDistanceById(wrong, 11).has_value());
        int64_t candidates[] = {11, 999, 22, 999};
        for (int64_t topk : {-1, 2}) {
            auto batch = index->CalcDistancesById(base, candidates, 2, true, topk);
            REQUIRE(batch.has_value());
            REQUIRE(batch.value()->GetNumElements() == 2);
            REQUIRE(batch.value()->GetDistances()[0] == 0);
            REQUIRE(batch.value()->GetDistances()[1] == -1);
            REQUIRE(batch.value()->GetDistances()[2] == 0);
            REQUIRE(batch.value()->GetDistances()[3] == -1);
        }
    }
}

TEST_CASE("HGraph analyzer preserves native half queries and batch strides",
          "[ft][hgraph][fp16][bf16][analyzer]") {
    const std::string dtype = GENERATE("float32", "float16", "bfloat16");
    const std::string quant = GENERATE("sq8", "rabitq", "tq");
    CAPTURE(dtype, quant);
    constexpr int64_t dim = 33;
    constexpr int64_t count = 256;
    constexpr int64_t topk = 5;
    std::vector<float> floats(count * dim);
    std::vector<uint16_t> halves(count * dim);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 1000);
    std::mt19937 random(20261112);
    for (uint64_t i = 0; i < floats.size(); ++i) {
        const auto value = (1 + random() % 128) / 129.0F;
        const bool bf16 = dtype == "bfloat16";
        halves[i] = bf16 ? generic::FloatToBF16(value) : generic::FloatToFP16(value);
        floats[i] = bf16 ? generic::BF16ToFloat(halves[i]) : generic::FP16ToFloat(halves[i]);
    }
    const auto input = [&](uint64_t row, uint64_t size) {
        auto data =
            Dataset::Make()->NumElements(size)->Dim(dim)->Owner(false)->Ids(ids.data() + row);
        if (dtype == "float32") {
            data->Float32Vectors(floats.data() + row * dim);
        } else {
            data->Float16Vectors(halves.data() + row * dim);
        }
        return data;
    };
    const std::string transform = quant == "tq" ? R"(,"tq_chain":"mrle,rabitq","mrle_dim":24)" : "";
    const auto parameters = std::string(R"({"dtype":")") + dtype +
                            R"(","metric_type":"l2","dim":33,"index_param":{"max_degree":16,
        "ef_construction":64,"build_thread_count":1,"base_quantization_type":")" +
                            quant + R"(","rabitq_bits_per_dim_base":3)" + transform + R"(,
        "precise_quantization_type":"sq8","use_reorder":true,"support_duplicate":true,
        "graph_storage_type":"compressed"}})";
    auto made = Factory::CreateIndex("hgraph", parameters);
    REQUIRE(made.has_value());
    auto index = made.value();
    auto built = index->Build(input(0, count));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    const auto stats = JsonType::Parse(index->GetStats());
    for (const auto* key : {"avg_distance_base", "recall_base", "quantization_bias_ratio"}) {
        REQUIRE(std::isfinite(stats[key].GetFloat()));
    }
    REQUIRE(stats["recall_base"].GetFloat() > 0.5F);

    SearchRequest request;
    request.query_ = input(0, 2);
    request.topk_ = topk;
    request.params_str_ = R"({"hgraph":{"ef_search":100}})";
    const auto batch = JsonType::Parse(index->AnalyzeIndexBySearch(request));
    REQUIRE(std::isfinite(batch["quantization_bias_ratio_query"].GetFloat()));
    float mean_distance = 0;
    float mean_recall = 0;
    for (uint64_t row = 0; row < 2; ++row) {
        request.query_ = input(row, 1);
        const auto single = JsonType::Parse(index->AnalyzeIndexBySearch(request));
        mean_distance += single["avg_distance_query"].GetFloat() / 2;
        mean_recall += single["recall_query"].GetFloat() / 2;
        auto distances = index->CalcDistancesById(request.query_, ids.data(), count, true);
        REQUIRE(distances.has_value());
        const auto* values = distances.value()->GetDistances();
        std::vector<float> sorted(values, values + count);
        std::sort(sorted.begin(), sorted.end());
        const auto reference = std::accumulate(sorted.begin(), sorted.begin() + topk, 0.0F) / topk;
        REQUIRE(single["avg_distance_query"].GetFloat() == Catch::Approx(reference));
    }
    REQUIRE(batch["avg_distance_query"].GetFloat() == Catch::Approx(mean_distance));
    REQUIRE(batch["recall_query"].GetFloat() == Catch::Approx(mean_recall));
    if (dtype != "float32") {
        request.query_ =
            Dataset::Make()->Dim(dim)->NumElements(1)->Owner(false)->Float32Vectors(floats.data());
        REQUIRE_THROWS(index->AnalyzeIndexBySearch(request));
    }
}

TEST_CASE("HGraph analyzer preserves INT8 queries and batch strides",
          "[ft][hgraph][int8][analyzer]") {
    constexpr int64_t dim = 33;
    constexpr int64_t count = 128;
    constexpr int64_t topk = 5;
    std::vector<int8_t> vectors(count * dim);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 1000);
    std::mt19937 random(20261112);
    for (auto& value : vectors) {
        value = static_cast<int8_t>(static_cast<int>(random() % 23) - 11);
    }
    const auto input = [&](uint64_t row, uint64_t size) {
        return Dataset::Make()
            ->NumElements(size)
            ->Dim(dim)
            ->Owner(false)
            ->Ids(ids.data() + row)
            ->Int8Vectors(vectors.data() + row * dim);
    };
    auto made = Factory::CreateIndex("hgraph", R"({"dtype":"int8","metric_type":"l2","dim":33,
        "index_param":{"max_degree":16,"ef_construction":64,"build_thread_count":1,
        "base_quantization_type":"int8","use_reorder":false,"graph_storage_type":"compressed"}})");
    REQUIRE(made.has_value());
    auto index = made.value();
    auto built = index->Build(input(0, count));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    const auto stats = JsonType::Parse(index->GetStats());
    REQUIRE(std::isfinite(stats["avg_distance_base"].GetFloat()));
    REQUIRE(stats["recall_base"].GetFloat() > 0.5F);

    SearchRequest request;
    request.query_ = input(0, 2);
    request.topk_ = topk;
    request.params_str_ = R"({"hgraph":{"ef_search":100}})";
    const auto batch = JsonType::Parse(index->AnalyzeIndexBySearch(request));
    REQUIRE(batch["recall_query"].GetFloat() > 0.5F);
    float mean_distance = 0;
    float mean_recall = 0;
    for (uint64_t row = 0; row < 2; ++row) {
        request.query_ = input(row, 1);
        const auto single = JsonType::Parse(index->AnalyzeIndexBySearch(request));
        auto distances = index->CalcDistancesById(request.query_, ids.data(), count, true);
        REQUIRE(distances.has_value());
        const auto* values = distances.value()->GetDistances();
        std::vector<float> sorted(values, values + count);
        std::sort(sorted.begin(), sorted.end());
        const auto reference = std::accumulate(sorted.begin(), sorted.begin() + topk, 0.0F) / topk;
        REQUIRE(single["avg_distance_query"].GetFloat() == Catch::Approx(reference));
        mean_distance += reference / 2;
        mean_recall += single["recall_query"].GetFloat() / 2;
    }
    REQUIRE(batch["avg_distance_query"].GetFloat() == Catch::Approx(mean_distance));
    REQUIRE(batch["recall_query"].GetFloat() == Catch::Approx(mean_recall));
    std::vector<float> wrong(dim, 0.0F);
    request.query_ =
        Dataset::Make()->Dim(dim)->NumElements(1)->Owner(false)->Float32Vectors(wrong.data());
    REQUIRE_THROWS(index->AnalyzeIndexBySearch(request));
}

TEST_CASE("HGraph fused RaBitQ analysis preserves native half samples",
          "[ft][hgraph][fp16][bf16][analyzer][fused]") {
    const std::string dtype = GENERATE("float32", "float16", "bfloat16");
    CAPTURE(dtype);
    constexpr int64_t dim = 64;
    constexpr int64_t count = 8;
    std::vector<float> floats(count * dim);
    std::vector<uint16_t> halves(count * dim);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 1000);
    std::mt19937 random(20261112);
    for (uint64_t i = 0; i < floats.size(); ++i) {
        const auto value = static_cast<float>(1 + random() % 64) / 64.0F;
        const bool bf16 = dtype == "bfloat16";
        halves[i] = bf16 ? generic::FloatToBF16(value) : generic::FloatToFP16(value);
        floats[i] = bf16 ? generic::BF16ToFloat(halves[i]) : generic::FP16ToFloat(halves[i]);
    }
    const auto input = [&](uint64_t row, uint64_t size) {
        auto data =
            Dataset::Make()->NumElements(size)->Dim(dim)->Owner(false)->Ids(ids.data() + row);
        if (dtype == "float32") {
            data->Float32Vectors(floats.data() + row * dim);
        } else {
            data->Float16Vectors(halves.data() + row * dim);
        }
        return data;
    };
    const auto parameters = std::string(R"({"dtype":")") + dtype +
                            R"(","metric_type":"l2","dim":64,"index_param":{
        "base_quantization_type":"rabitq","precise_quantization_type":"rabitq",
        "base_io_type":"memory_io","base_supplement_io_type":"memory_io",
        "rabitq_bits_per_dim_base":3,"rabitq_bits_per_dim_precise":5,"rabitq_use_fht":true,
        "graph_io_type":"memory_io","graph_storage_type":"flat","graph_type":"nsw",
        "max_degree":8,"ef_construction":32,"build_thread_count":1,
        "use_reorder":true,"reorder_source":"base","store_raw_vector":false,
        "rabitq_fused_datacell":true}})";
    auto made = Factory::CreateIndex("hgraph", parameters);
    REQUIRE(made.has_value());
    auto index = made.value();
    auto built = index->Build(input(0, count));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());

    // Each training point is a centroid, so decoded samples retain their input values.
    // GetStats samples every row and includes every id in its top-k for this small index.
    float reference_distance = 0.0F;
    for (uint64_t row = 0; row < count; ++row) {
        auto distances = index->CalcDistancesById(input(row, 1), ids.data(), count, true);
        REQUIRE(distances.has_value());
        reference_distance += std::accumulate(distances.value()->GetDistances(),
                                              distances.value()->GetDistances() + count,
                                              0.0F) /
                              static_cast<float>(count * count);
    }
    REQUIRE(reference_distance > 0.0F);
    const auto stats = JsonType::Parse(index->GetStats());
    REQUIRE(stats["recall_base"].GetFloat() > 0.5F);
    REQUIRE(stats["avg_distance_base"].GetFloat() ==
            Catch::Approx(reference_distance).epsilon(1e-4).margin(1e-4));
    REQUIRE(std::isfinite(stats["quantization_bias_ratio"].GetFloat()));
}

}  // namespace vsag
