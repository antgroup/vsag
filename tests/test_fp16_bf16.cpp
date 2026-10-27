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

TEST_CASE("IVF with FP16 build and search", "[ft][ivf][fp16]") {
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 100;

    auto param = R"(
    {
        "dtype": "float16",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp16",
            "buckets_count": 4,
            "train_sample_count": 512,
            "use_route_graph": false
        }
    }
    )";

    auto index = vsag::Factory::CreateIndex("ivf", param).value();
    REQUIRE(index->GetNumElements() == 0);

    auto fp32_data = GenerateRandomFP32Vectors(TEST_DIM, TEST_COUNT);
    auto base = CreateFP16Dataset(TEST_DIM, TEST_COUNT, fp32_data);
    auto build_result = index->Build(base);

    REQUIRE(build_result.has_value());
    auto failed_ids = build_result.value();
    REQUIRE(index->GetNumElements() == TEST_COUNT - failed_ids.size());

    if (index->GetNumElements() > 0) {
        auto query_fp32 = GenerateRandomFP32Vectors(TEST_DIM, 1);
        auto query = CreateFP16Dataset(TEST_DIM, 1, query_fp32);

        auto search_param = R"({"ivf": {"scan_buckets_count": 4}})";
        auto search_result = index->KnnSearch(query, 10, search_param);
        REQUIRE(search_result.has_value());
        auto result = search_result.value();
        REQUIRE(result->GetDim() == 10);
        REQUIRE(result->GetNumElements() == 1);
    }
}

TEST_CASE("IVF with BF16 build and search", "[ft][ivf][bf16]") {
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 100;

    auto param = R"(
    {
        "dtype": "bfloat16",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "bf16",
            "buckets_count": 4,
            "train_sample_count": 512,
            "use_route_graph": false
        }
    }
    )";

    auto index = vsag::Factory::CreateIndex("ivf", param).value();
    REQUIRE(index->GetNumElements() == 0);

    auto fp32_data = GenerateRandomFP32Vectors(TEST_DIM, TEST_COUNT);
    auto base = CreateBF16Dataset(TEST_DIM, TEST_COUNT, fp32_data);
    auto build_result = index->Build(base);

    REQUIRE(build_result.has_value());
    auto failed_ids = build_result.value();
    REQUIRE(index->GetNumElements() == TEST_COUNT - failed_ids.size());

    if (index->GetNumElements() > 0) {
        auto query_fp32 = GenerateRandomFP32Vectors(TEST_DIM, 1);
        auto query = CreateBF16Dataset(TEST_DIM, 1, query_fp32);

        auto search_param = R"({"ivf": {"scan_buckets_count": 4}})";
        auto search_result = index->KnnSearch(query, 10, search_param);
        REQUIRE(search_result.has_value());
    }
}

TEST_CASE("BruteForce with FP16 build and search", "[ft][bruteforce][fp16]") {
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 100;

    auto param = R"(
    {
        "dtype": "float16",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp16"
        }
    }
    )";

    auto index = vsag::Factory::CreateIndex("brute_force", param).value();
    REQUIRE(index->GetNumElements() == 0);

    auto fp32_data = GenerateRandomFP32Vectors(TEST_DIM, TEST_COUNT);
    auto base = CreateFP16Dataset(TEST_DIM, TEST_COUNT, fp32_data);
    auto build_result = index->Build(base);

    REQUIRE(build_result.has_value());
    auto failed_ids = build_result.value();
    REQUIRE(index->GetNumElements() == TEST_COUNT - failed_ids.size());

    if (index->GetNumElements() > 0) {
        auto query_fp32 = GenerateRandomFP32Vectors(TEST_DIM, 1);
        auto query = CreateFP16Dataset(TEST_DIM, 1, query_fp32);

        auto search_param = "{}";
        auto search_result = index->KnnSearch(query, 10, search_param);
        REQUIRE(search_result.has_value());
    }
}

TEST_CASE("BruteForce with BF16 build and search", "[ft][bruteforce][bf16]") {
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 100;

    auto param = R"(
    {
        "dtype": "bfloat16",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "bf16"
        }
    }
    )";

    auto index = vsag::Factory::CreateIndex("brute_force", param).value();
    REQUIRE(index->GetNumElements() == 0);

    auto fp32_data = GenerateRandomFP32Vectors(TEST_DIM, TEST_COUNT);
    auto base = CreateBF16Dataset(TEST_DIM, TEST_COUNT, fp32_data);
    auto build_result = index->Build(base);

    REQUIRE(build_result.has_value());
    auto failed_ids = build_result.value();
    REQUIRE(index->GetNumElements() == TEST_COUNT - failed_ids.size());

    if (index->GetNumElements() > 0) {
        auto query_fp32 = GenerateRandomFP32Vectors(TEST_DIM, 1);
        auto query = CreateBF16Dataset(TEST_DIM, 1, query_fp32);

        auto search_param = "{}";
        auto search_result = index->KnnSearch(query, 10, search_param);
        REQUIRE(search_result.has_value());
    }
}

TEST_CASE("Half input numerical regression", "[ft][fp16][bf16][half_regression]") {
    constexpr int64_t dim = 33;
    constexpr int64_t count = 64;
    for (const auto& index_name : {std::string("ivf"), std::string("brute_force")}) {
        for (const auto& dtype : {std::string("float16"), std::string("bfloat16")}) {
            for (const auto& quant :
                 {std::string("fp32"), std::string("fp16"), std::string("bf16")}) {
                for (const bool route_graph : {false, true}) {
                    if (route_graph && index_name != "ivf")
                        continue;
                    INFO(index_name << " dtype=" << dtype << " quantizer=" << quant
                                    << " route_graph=" << route_graph);
                    std::string index_param = "\"base_quantization_type\":\"" + quant + "\"";
                    if (index_name == "ivf") {
                        index_param += std::string(
                                           ",\"buckets_count\":4,\"train_sample_count\":512,\"use_"
                                           "route_graph\":") +
                                       (route_graph ? "true" : "false");
                        if (route_graph)
                            index_param += ",\"graph_build_threshold\":1";
                    }
                    const std::string params =
                        "{\"dtype\":\"" + dtype +
                        "\",\"metric_type\":\"l2\",\"dim\":33,\"index_param\":{" + index_param +
                        "}}";
                    auto made = Factory::CreateIndex(index_name, params);
                    REQUIRE(made.has_value());
                    auto index = made.value();
                    std::vector<float> values(dim * count);
                    for (int64_t row = 0; row < count; ++row) {
                        for (int64_t col = 0; col < dim; ++col) {
                            // Exactly representable in both formats; asymmetric rows expose bad strides.
                            values[row * dim + col] =
                                static_cast<float>((row * 7 + col * 11) % 61 - 30) / 8.0F;
                        }
                        values[row * dim] = static_cast<float>(row) / 8.0F;
                    }
                    auto base = dtype == "float16" ? CreateFP16Dataset(dim, count, values)
                                                   : CreateBF16Dataset(dim, count, values);
                    base->NumElements(count - 1);
                    auto built = index->Build(base);
                    REQUIRE(built.has_value());
                    REQUIRE(built.value().empty());
                    int64_t last_id = count - 1;
                    auto extra =
                        Dataset::Make()
                            ->Owner(false)
                            ->Dim(dim)
                            ->NumElements(1)
                            ->Ids(&last_id)
                            ->Float16Vectors(base->GetFloat16Vectors() + (count - 1) * dim);
                    auto added = index->Add(extra);
                    REQUIRE(added.has_value());
                    REQUIRE(added.value().empty());
                    REQUIRE(index->GetNumElements() == count);
                    const std::string search_param =
                        index_name == "ivf" ? "{\"ivf\":{\"scan_buckets_count\":4}}" : "{}";
                    auto check = [&](const auto& target) {
                        for (int64_t row : {int64_t(0), int64_t(17), count - 1}) {
                            auto query =
                                Dataset::Make()
                                    ->Owner(false)
                                    ->Dim(dim)
                                    ->NumElements(1)
                                    ->Float16Vectors(base->GetFloat16Vectors() + row * dim);
                            auto found = target->KnnSearch(query, count, search_param);
                            REQUIRE(found.has_value());
                            REQUIRE(found.value()->GetDim() == count);
                            REQUIRE(found.value()->GetIds()[0] == row);
                            REQUIRE(std::abs(found.value()->GetDistances()[0]) < 0.001F);
                            std::vector<bool> seen(count, false);
                            for (int64_t k = 0; k < count; ++k) {
                                const auto id = found.value()->GetIds()[k];
                                REQUIRE(id >= 0);
                                REQUIRE(id < count);
                                REQUIRE_FALSE(seen[id]);
                                seen[id] = true;
                                float expected = 0;
                                for (int64_t col = 0; col < dim; ++col) {
                                    const float delta =
                                        values[row * dim + col] - values[id * dim + col];
                                    expected += delta * delta;
                                }
                                REQUIRE(std::abs(found.value()->GetDistances()[k] - expected) <
                                        0.02F);
                            }
                        }
                    };
                    check(index);
                    if (index_name == "ivf" && route_graph) {
                        REQUIRE(index->RebuildIVFBucketGraphs().has_value());
                        check(index);
                    }
                    auto serialized = index->Serialize();
                    REQUIRE(serialized.has_value());
                    auto restored = Factory::CreateIndex(index_name, params);
                    REQUIRE(restored.has_value());
                    REQUIRE(restored.value()->Deserialize(serialized.value()).has_value());
                    check(restored.value());
                }
            }
        }
    }
}

TEST_CASE("Half input missing representation is rejected", "[ft][fp16][bf16][half_regression]") {
    for (const auto& name : {std::string("ivf"), std::string("brute_force")}) {
        for (const auto& dtype : {std::string("float16"), std::string("bfloat16")}) {
            const std::string params = "{\"dtype\":\"" + dtype +
                                       "\",\"metric_type\":\"l2\",\"dim\":33,\"index_param\":{"
                                       "\"base_quantization_type\":\"fp32\"}}";
            auto made = Factory::CreateIndex(name, params);
            REQUIRE(made.has_value());
            std::vector<float> wrong(33 * 64, 1.0F);
            std::vector<int64_t> ids(64);
            for (int64_t i = 0; i < 64; ++i) ids[i] = i;
            auto base = Dataset::Make()
                            ->Owner(false)
                            ->Dim(33)
                            ->NumElements(64)
                            ->Ids(ids.data())
                            ->Float32Vectors(wrong.data());
            REQUIRE_FALSE(made.value()->Build(base).has_value());
            auto valid = dtype == "float16" ? CreateFP16Dataset(33, 64, wrong)
                                            : CreateBF16Dataset(33, 64, wrong);
            REQUIRE(made.value()->Build(valid).has_value());
            REQUIRE_FALSE(made.value()->Add(base).has_value());
            if (name == "brute_force") {
                base->NumElements(1);
                REQUIRE_FALSE(made.value()->UpdateVector(0, base).has_value());
            }
        }
    }
}

}  // namespace vsag
