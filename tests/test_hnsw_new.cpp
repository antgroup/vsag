
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

#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <limits>

#include "simd/simd.h"
#include "test_index.h"
#include "vsag/vsag.h"

TEST_CASE_PERSISTENT_FIXTURE(fixtures::TestIndex, "HNSW Build & ContinueAdd Test", "[ft][hnsw]") {
    auto dims = fixtures::get_common_used_dims(3);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    for (auto& dim : dims) {
        auto param = fixtures::generate_hnsw_build_parameters_string(metric_type, dim);
        auto index = TestFactory(name, param, true);
        TestBuildIndex(index, dim);
        TestContinueAdd(index, dim);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::TestIndex, "HNSW Float General Test", "[ft][hnsw]") {
    auto dims = fixtures::get_common_used_dims(3);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    for (auto& dim : dims) {
        auto build_param = fixtures::generate_hnsw_build_parameters_string(metric_type, dim);
        FastGeneralTest(name, build_param, search_parameters, metric_type, dim);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::TestIndex,
                             "HNSW Factory Test With Exceptions",
                             "[ft][hnsw]") {
    auto name = "hnsw";
    SECTION("Empty parameters") {
        auto param = "{}";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("No dim param") {
        auto param = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "hnsw": {{
                "max_degree": 64,
                "ef_construction": 500
            }}
        }})";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid param") {
        auto metric = GENERATE("", "l4", "inner_product", "cosin", "hamming");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "{}",
            "dim": 23,
            "hnsw": {{
                "max_degree": 64,
                "ef_construction": 500
            }}
        }})";
        auto param = fmt::format(param_tmp, metric);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid datatype param") {
        auto datatype = GENERATE("fp32", "uint8_t", "binary", "", "float");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "{}",
            "metric_type": "l2",
            "dim": 23,
            "hnsw": {{
                "max_degree": 64,
                "ef_construction": 500
            }}
        }})";
        auto param = fmt::format(param_tmp, datatype);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }
    // TODO(lht)dim check
    /*
    SECTION("Invalid dim param") {
        auto dim = GENERATE(-1, std::numeric_limits<uint64_t>::max(), 0, 8.6);
        auto param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "dim": {},
            "hnsw": {{
                "max_degree": 64,
                "ef_construction": 500
            }}
        }})";
        auto param = fmt::format(param_tmp, dim);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }
    */

    SECTION("Miss hnsw param") {
        auto param = GENERATE(
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "hnsw": {{
                    "ef_construction": 500
                }}
            }})",
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "hnsw": {{
                    "max_degree": 64,
                }}
            }})");
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid hnsw param max_degree") {
        auto max_degree = GENERATE(-1, 0, 256, 3);
        // TODO(LHT): test for float param
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "hnsw": {{
                    "max_degree": {},
                    "ef_construction": 500
                }}
            }})";
        auto param = fmt::format(param_temp, max_degree);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid hnsw param ef_construction") {
        auto ef_construction = GENERATE(-1, 0, 100000, 31);
        // TODO(LHT): test for float param
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "hnsw": {{
                    "max_degree": 32,
                    "ef_construction": {}
                }}
            }})";
        auto param = fmt::format(param_temp, ef_construction);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("No match name and parameters") {
        auto new_name = GENERATE("diskann", "hgraph", "hsnw", "", "hnswlib");
        auto param =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 135,
                "hnsw": {{
                    "max_degree": 32,
                    "ef_construction": 500
                }}
            }})";
        REQUIRE_THROWS(TestFactory(new_name, param, false));
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::TestIndex,
                             "HNSW Build & Add Test With Exceptions",
                             "[ft][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kINFO);
    auto dims = fixtures::get_common_used_dims(3);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    std::string metric = metric_type;
    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    for (auto& dim : dims) {
        auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
        auto build_param = fixtures::generate_hnsw_build_parameters_string(metric_type, dim);
        auto index = FastGeneralTest(
            name, build_param, search_parameters, metric_type, dim, IndexStatus::Factory);
        SaveIndex(key, index, IndexStatus::Factory);
    }

    SECTION("Invalid dim for build") {
        auto incorrect_dim = GENERATE(-1, 20, 32131);
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto [index, status] = LoadIndex(key);
            std::vector<int64_t> ids(1);
            std::vector<float> vectors(dim);
            if (incorrect_dim > 0) {
                vectors.resize(incorrect_dim);
            }

            auto dataset = vsag::Dataset::Make();
            dataset->Dim(incorrect_dim)
                ->NumElements(1)
                ->Ids(ids.data())
                ->Float32Vectors(vectors.data())
                ->Owner(false);
            TestBuildIndex(index, dataset, false);
        }
    }

    SECTION("Invalid dim for add") {
        auto incorrect_dim = GENERATE(-1, 20, 32131);
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto [index, status] = LoadIndex(key);
            std::vector<int64_t> ids(1);
            std::vector<float> vectors(dim);
            if (incorrect_dim > 0) {
                vectors.resize(incorrect_dim);
            }

            auto dataset = vsag::Dataset::Make();
            dataset->Dim(incorrect_dim)
                ->NumElements(1)
                ->Ids(ids.data())
                ->Float32Vectors(vectors.data())
                ->Owner(false);
            TestAddIndex(index, dataset, false);
        }
    }
    // TODO(LHT): bugfix
    /*
    SECTION("Invalid data=nullptr for build & add") {
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto [index, status] = LoadIndex(key);
            std::vector<int64_t> ids(1);
            auto dataset = vsag::Dataset::Make();
            dataset->Dim(dim)
                ->NumElements(1)
                ->Ids(ids.data())
                ->Float32Vectors(nullptr)
                ->Owner(false);
            TestAddIndex(index, dataset, false);
            TestBuildIndex(index, dataset, false);
        }
    }
     */
    // TODO(LHT): bugfix
    /*
    SECTION("Invalid data is short for build & add") {
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto [index, status] = LoadIndex(key);
            std::vector<int64_t> ids(3);
            std::vector<float> base(2 * dim);
            auto dataset = vsag::Dataset::Make();
            dataset->Dim(dim)
                ->NumElements(3)
                ->Ids(ids.data())
                ->Float32Vectors(base.data())
                ->Owner(false);
            TestAddIndex(index, dataset, false);
            TestBuildIndex(index, dataset, false);
        }
    }
    */

    // TODO(LHT): bugfix
    /*
    SECTION("Invalid id=nullptr for build & add") {
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto [index, status] = LoadIndex(key);
            std::vector<float> vectors(dim);
            auto dataset = vsag::Dataset::Make();
            dataset->Dim(dim)
                ->NumElements(1)
                ->Ids(nullptr)
                ->Float32Vectors(vectors.data())
                ->Owner(false);
            TestAddIndex(index, dataset, false);
            TestBuildIndex(index, dataset, false);
        }
    }*/

    // TODO(LHT): bugfix
    /*
    SECTION("Invalid id is short for build & add") {
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto [index, status] = LoadIndex(key);
            std::vector<int64_t> ids(2);
            std::vector<float> base(3 * dim);
            auto dataset = vsag::Dataset::Make();
            dataset->Dim(dim)
                ->NumElements(3)
                ->Ids(ids.data())
                ->Float32Vectors(base.data())
                ->Owner(false);
            TestAddIndex(index, dataset, false);
            TestBuildIndex(index, dataset, false);
        }
    }
     */
}

TEST_CASE_METHOD(fixtures::TestIndex,
                 "HNSW KnnSearch & RangeSearch Test With Exceptions",
                 "[ft][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kINFO);
    auto dims = fixtures::get_common_used_dims(3);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    std::string metric = metric_type;
    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    auto top_k = 10;
    float range = 0.01;
    float recall = 0.99;

    for (auto& dim : dims) {
        auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);

        auto build_param = fixtures::generate_hnsw_build_parameters_string(metric_type, dim);
        auto index = FastGeneralTest(
            name, build_param, search_parameters, metric_type, dim, IndexStatus::Build);
        SaveIndex(key, index, IndexStatus::Build);
    }

    SECTION("Miss search_param or wrong key") {
        auto new_param = GENERATE("",
                                  "{}",
                                  R"({
                                        "hsnw": {
                                            "ef_search": 100
                                        }
                                    })",
                                  R"({
                                        "hnsw": {
                                            "efsearch": 100
                                        }
                                    })");
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto dataset_key = KeyGen(dim, dataset_base_count);
            auto [index, status] = LoadIndex(key);
            TestKnnSearch(index, GetDataset(dataset_key), new_param, top_k, recall, false);
            TestRangeSearch(index, GetDataset(dataset_key), new_param, range, recall, -1, false);
        }
    }

    SECTION("Invalid search_param ef_search value") {
        auto ef_search = GENERATE(-1, 0, 37182903, -100);  // TODO(LHT): float param
        constexpr const char* param_temp =
            R"({{
                "hnsw": {{
                    "ef_search": {}
                }}
            }})";
        auto new_param = fmt::format(param_temp, ef_search);
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto dataset_key = KeyGen(dim, dataset_base_count);
            auto [index, status] = LoadIndex(key);
            TestKnnSearch(index, GetDataset(dataset_key), new_param, top_k, recall, false);
            TestRangeSearch(index, GetDataset(dataset_key), new_param, range, recall, -1, false);
        }
    }

    SECTION("Invalid topk value for knn search") {
        auto topk = GENERATE(-1, 0, -100);
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto dataset_key = KeyGen(dim, dataset_base_count);
            auto [index, status] = LoadIndex(key);
            TestKnnSearch(index, GetDataset(dataset_key), search_parameters, topk, recall, false);
        }
    }

    SECTION("Invalid radius value for range search") {
        float radius = GENERATE(-1, -100);
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto dataset_key = KeyGen(dim, dataset_base_count);
            auto [index, status] = LoadIndex(key);
            TestRangeSearch(
                index, GetDataset(dataset_key), search_parameters, radius, recall, -1, false);
        }
    }

    SECTION("Invalid query dataset for search (Invalid query dim)") {
        auto incorrect_dim = GENERATE(-1, 20, 32131);
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto dataset_key = KeyGen(dim, dataset_base_count);
            auto [index, status] = LoadIndex(key);

            std::vector<int64_t> ids(1);
            std::vector<float> vectors(dim);
            if (incorrect_dim > 0) {
                vectors.resize(incorrect_dim);
            }

            auto dataset = vsag::Dataset::Make();
            dataset->Dim(incorrect_dim)
                ->NumElements(1)
                ->Ids(ids.data())
                ->Float32Vectors(vectors.data())
                ->Owner(false);

            auto knn_result = index->KnnSearch(dataset, top_k, search_parameters);
            REQUIRE_FALSE(knn_result.has_value());
            auto range_result = index->RangeSearch(dataset, range, search_parameters);
            REQUIRE_FALSE(range_result.has_value());
        }
    }
    // TODO(LHT): bugfix
    /*
    SECTION("Invalid query dataset for search (Invalid query=nullptr)") {
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto [index, status] = LoadIndex(key);
            std::vector<int64_t> ids(1);
            auto dataset = vsag::Dataset::Make();
            dataset->Dim(dim)
                ->NumElements(1)
                ->Ids(ids.data())
                ->Float32Vectors(nullptr)
                ->Owner(false);
            auto knn_result = index->KnnSearch(dataset, top_k, search_parameters);
            REQUIRE_FALSE(knn_result.has_value());
            auto range_result = index->RangeSearch(dataset, range, search_parameters);
            REQUIRE_FALSE(range_result.has_value());
        }
    }
     */
    // TODO(LHT): bugfix
    /*
    SECTION("Invalid query dataset for search (query is shorter than dim)") {
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto [index, status] = LoadIndex(key);
            std::vector<float> query(dim - 1);
            auto dataset = vsag::Dataset::Make();
            dataset->Dim(dim)
                ->NumElements(1)
                ->Ids(nullptr)
                ->Float32Vectors(query.data())
                ->Owner(false);
            auto knn_result = index->KnnSearch(dataset, top_k, search_parameters);
            REQUIRE_FALSE(knn_result.has_value());
            auto range_result = index->RangeSearch(dataset, range, search_parameters);
            REQUIRE_FALSE(range_result.has_value());
        }
    }
     */

    SECTION("Invalid query dataset for search (query counts != 1)") {
        auto elements = GENERATE(3, 10, -1, 0);
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto [index, status] = LoadIndex(key);
            std::vector<float> query(dim);
            auto dataset = vsag::Dataset::Make();
            dataset->Dim(dim)
                ->NumElements(elements)
                ->Ids(nullptr)
                ->Float32Vectors(query.data())
                ->Owner(false);
            auto knn_result = index->KnnSearch(dataset, top_k, search_parameters);
            REQUIRE_FALSE(knn_result.has_value());
            auto range_result = index->RangeSearch(dataset, range, search_parameters);
            REQUIRE_FALSE(range_result.has_value());
        }
    }

    SECTION("Success limited_size for range search") {
        float new_range = 1000;
        auto limited_size = GENERATE(3, 10, -1, -1000);
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto dataset_key = KeyGen(dim, dataset_base_count);
            auto [index, status] = LoadIndex(key);
            TestRangeSearch(index,
                            GetDataset(dataset_key),
                            search_parameters,
                            new_range,
                            recall,
                            limited_size,
                            true);
        }
    }

    SECTION("Invalid limited_size for range search") {
        auto limited_size = GENERATE(0);
        for (auto& dim : dims) {
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto dataset_key = KeyGen(dim, dataset_base_count);
            auto [index, status] = LoadIndex(key);
            TestRangeSearch(index,
                            GetDataset(dataset_key),
                            search_parameters,
                            range,
                            recall,
                            limited_size,
                            false);
        }
    }
}

TEST_CASE_METHOD(fixtures::TestIndex,
                 "HNSW Serialize & Deserialize Test With Exceptions",
                 "[ft][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kINFO);
    auto dims = fixtures::get_common_used_dims(3);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    const std::string name = "hnsw";
    std::string metric = metric_type;
    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    })";

    for (auto& dim : dims) {
        auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);

        auto build_param = fixtures::generate_hnsw_build_parameters_string(metric_type, dim);
        auto index = FastGeneralTest(
            name, build_param, search_parameters, metric_type, dim, IndexStatus::Build);
        SaveIndex(key, index, IndexStatus::Build);
    }
    // TODO(LHT): bugfix
    /*
    SECTION("Invalid file for Serialize & Deserialize (shorter file)") {
        fixtures::TempDir dir("hnsw");
        for (auto& dim : dims) {
            auto build_param = fixtures::generate_hnsw_build_parameters_string(metric_type, dim);
            auto key = KeyGenIndex(dim, dataset_base_count, "hnsw_" + metric);
            auto dataset_key = KeyGen(dim, dataset_base_count);
            auto [index, status] = LoadIndex(key);
            auto filename = dir.GenerateRandomFile();
            TestSerializeFile(index, filename, true);
            auto size = fixtures::GetFileSize(filename);
            std::filesystem::resize_file(filename, size - 10);

            TestDeserializeFile(filename, name, build_param, false);
        }
    }
     */
}
/*

TEST_CASE("HNSW Filtering Test", "[ft][hnsw]") {
    spdlog::set_level(spdlog::level::debug);

    int dim = 17;
    int max_elements = 1000;
    int max_degree = 16;
    int ef_construction = 100;
    int ef_search = 1000;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int64_t i = 0; i < max_elements; i++) {
        ids[i] = max_elements - i - 1;
    }
    for (int64_t i = 0; i < dim * max_elements; ++i) {
        data[i] = distrib_real(rng);
    }

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(dim)->NumElements(max_elements)->Ids(ids)->Float32Vectors(data);
    hnsw->Build(dataset);

    REQUIRE(hnsw->GetNumElements() == max_elements);

    // Query the elements for themselves and measure recall 1@1
    float correct_knn = 0.0f;
    float recall_knn = 0.0f;
    float correct_range = 0.0f;
    float recall_range = 0.0f;
    for (int i = 0; i < max_elements; i++) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(data + i * dim)->Owner(false);
        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        float radius = 9.87f;
        int64_t k = 10;

        vsag::BitsetPtr filter = vsag::Bitset::Random(max_elements);
        int64_t num_deleted = filter->Count();

        if (auto result = hnsw->RangeSearch(query, radius, parameters.dump(), filter);
            result.has_value()) {
            REQUIRE(result.value()->GetDim() == max_elements - num_deleted);
            for (int64_t j = 0; j < result.value()->GetDim(); ++j) {
                // deleted ids NOT in result
                REQUIRE(filter->Test(result.value()->GetIds()[j]) == false);
            }
        } else {
            std::cerr << "failed to range search on index: internalError" << std::endl;
            exit(-1);
        }

        if (auto result = hnsw->KnnSearch(query, k, parameters.dump(), filter);
            result.has_value()) {
            REQUIRE(result.has_value());
            for (int64_t j = 0; j < result.value()->GetDim(); ++j) {
                // deleted ids NOT in result
                REQUIRE(filter->Test(result.value()->GetIds()[j]) == false);
            }
        } else {
            std::cerr << "failed to knn search on index: internalError" << std::endl;
            exit(-1);
        }

        vsag::BitsetPtr ones = vsag::Bitset::Make();
        for (int64_t j = 0; j < max_elements; ++j) {
            ones->Set(j, true);
        }
        if (auto result = hnsw->RangeSearch(query, radius, parameters.dump(), ones);
            result.has_value()) {
            REQUIRE(result.value()->GetDim() == 0);
            REQUIRE(result.value()->GetDistances() == nullptr);
            REQUIRE(result.value()->GetIds() == nullptr);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to range search on index: internalError" << std::endl;
            exit(-1);
        }

        if (auto result = hnsw->KnnSearch(query, k, parameters.dump(), ones); result.has_value()) {
            REQUIRE(result.value()->GetDim() == 0);
            REQUIRE(result.value()->GetDistances() == nullptr);
            REQUIRE(result.value()->GetIds() == nullptr);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to knn search on index: internalError" << std::endl;
            exit(-1);
        }

        vsag::BitsetPtr zeros = vsag::Bitset::Make();

        if (auto result = hnsw->KnnSearch(query, k, parameters.dump(), zeros); result.has_value()) {
            correct_knn += vsag::knn_search_recall(data,
                                                   ids,
                                                   max_elements,
                                                   data + i * dim,
                                                   dim,
                                                   result.value()->GetIds(),
                                                   result.value()->GetDim());
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to knn search on index: internalError" << std::endl;
            exit(-1);
        }

        if (auto result = hnsw->RangeSearch(query, radius, parameters.dump(), zeros);
            result.has_value()) {
            if (result.value()->GetNumElements() == 1) {
                if (result.value()->GetDim() != 0 && result.value()->GetIds()[0] == ids[i]) {
                    correct_range++;
                }
            }
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to range search on index: internalError" << std::endl;
            exit(-1);
        }
    }
    recall_knn = correct_knn / max_elements;
    recall_range = correct_range / max_elements;

    REQUIRE(recall_range == 1);
    REQUIRE(recall_knn == 1);
}

TEST_CASE("HNSW small dimension", "[ft][hnsw]") {
    spdlog::set_level(spdlog::level::debug);

    int dim = 3;
    int max_elements = 1000;
    int max_degree = 24;
    int ef_construction = 100;
    int ef_search = 100;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(dim)->NumElements(max_elements)->Ids(ids)->Float32Vectors(data);
    hnsw->Add(dataset);
    return;

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(data + i * dim)->Owner(false);
        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result.value()->GetIds()[0] == i) {
                correct++;
            }
            REQUIRE(result.value()->GetDim() == k);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}

TEST_CASE("HNSW Random Id", "[ft][hnsw]") {
    spdlog::set_level(spdlog::level::debug);

    int dim = 128;
    int max_elements = 1000;
    int max_degree = 64;
    int ef_construction = 200;
    int ef_search = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib_real;
    std::uniform_int_distribution<> ids_random(0, max_elements - 1);
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = ids_random(rng);
        if (i == 1 || i == 2) {
            ids[i] = std::numeric_limits<int64_t>::max();
        } else if (i == 3 || i == 4) {
            ids[i] = std::numeric_limits<int64_t>::min();
        } else if (i == 5 || i == 6) {
            ids[i] = 1;
        } else if (i == 7 || i == 8) {
            ids[i] = -1;
        } else if (i == 9 || i == 10) {
            ids[i] = 0;
        }
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(dim)->NumElements(max_elements)->Ids(ids)->Float32Vectors(data);
    auto failed_ids = hnsw->Build(dataset);

    float rate = hnsw->GetNumElements() / (float)max_elements;
    // 1 - 1 / e
    REQUIRE((rate > 0.60 && rate < 0.65));

    REQUIRE(failed_ids->size() + hnsw->GetNumElements() == max_elements);

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    std::set<int64_t> unique_ids;
    for (int i = 0; i < max_elements; i++) {
        if (unique_ids.find(ids[i]) != unique_ids.end()) {
            continue;
        }
        unique_ids.insert(ids[i]);
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(data + i * dim)->Owner(false);
        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result.value()->GetIds()[0] == ids[i]) {
                correct++;
            }
            REQUIRE(result.value()->GetDim() == k);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    float recall = correct / hnsw->GetNumElements();
    REQUIRE(recall == 1);
}

TEST_CASE("pq infer knn search time recall", "[ft][hnsw]") {
    spdlog::set_level(spdlog::level::debug);

    int dim = 128;
    int max_elements = 1000;
    int max_degree = 64;
    int ef_construction = 200;
    int ef_search = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"use_static", true},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(dim)->NumElements(max_elements)->Ids(ids)->Float32Vectors(data);
    hnsw->Build(dataset);

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(data + i * dim)->Owner(false);
        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result.value()->GetIds()[0] == i) {
                correct++;
            }
            REQUIRE(result.value()->GetDim() == k);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}

TEST_CASE("hnsw serialize", "[ft][hnsw]") {
    spdlog::set_level(spdlog::level::debug);

    int dim = 128;
    int max_elements = 1000;
    int max_degree = 64;
    int ef_construction = 200;
    int ef_search = 200;
    auto metric_type = GENERATE("cosine", "l2");
    auto use_static = GENERATE(true, false);
    // Initing index
    nlohmann::json hnsw_parameters{{"max_degree", max_degree},
                                   {"ef_construction", ef_construction},
                                   {"use_static", use_static},
                                   {"use_conjugate_graph", true}};
    nlohmann::json index_parameters{{"dtype", "float32"},
                                    {"metric_type", metric_type},
                                    {"dim", dim},
                                    {"hnsw", hnsw_parameters}};

    if (metric_type == std::string("cosine") && use_static) {
        return;  // static hnsw only support the metric type of l2.
    }
    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(dim)->NumElements(max_elements)->Ids(ids)->Float32Vectors(data);
    hnsw->Build(dataset);

    // Serialize(single-file)
    {
        if (auto bs = hnsw->Serialize(); bs.has_value()) {
            hnsw = nullptr;
            auto keys = bs->GetKeys();
            std::vector<uint64_t> offsets;

            std::ofstream file(tmp_dir + "hnsw.index", std::ios::binary);
            uint64_t offset = 0;
            for (auto key : keys) {
                // [len][data...][len][data...]...
                vsag::Binary b = bs->Get(key);
                writeBinaryPOD(file, b.size);
                file.write((const char*)b.data.get(), b.size);
                offsets.push_back(offset);
                offset += sizeof(b.size) + b.size;
            }
            // footer
            for (uint64_t i = 0; i < keys.size(); ++i) {
                // [len][key...][offset][len][key...][offset]...
                const auto& key = keys[i];
                int64_t len = key.length();
                writeBinaryPOD(file, len);
                file.write(key.c_str(), key.length());
                writeBinaryPOD(file, offsets[i]);
            }
            // [num_keys][footer_offset]$
            writeBinaryPOD(file, keys.size());
            writeBinaryPOD(file, offset);
            file.close();
        } else if (bs.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY) {
            std::cerr << "no enough memory to serialize index" << std::endl;
        }
    }

    // Deserialize(binaryset)
    {
        std::ifstream file(tmp_dir + "hnsw.index", std::ios::in);
        file.seekg(-sizeof(uint64_t) * 2, std::ios::end);
        uint64_t num_keys, footer_offset;
        readBinaryPOD(file, num_keys);
        readBinaryPOD(file, footer_offset);
        // std::cout << "num_keys: " << num_keys << std::endl;
        // std::cout << "footer_offset: " << footer_offset << std::endl;
        file.seekg(footer_offset, std::ios::beg);

        std::vector<std::string> keys;
        std::vector<uint64_t> offsets;
        for (uint64_t i = 0; i < num_keys; ++i) {
            int64_t key_len;
            readBinaryPOD(file, key_len);
            // std::cout << "key_len: " << key_len << std::endl;
            char key_buf[key_len + 1];
            memset(key_buf, 0, key_len + 1);
            file.read(key_buf, key_len);
            // std::cout << "key: " << key_buf << std::endl;
            keys.push_back(key_buf);

            uint64_t offset;
            readBinaryPOD(file, offset);
            // std::cout << "offset: " << offset << std::endl;
            offsets.push_back(offset);
        }

        vsag::BinarySet bs;
        for (uint64_t i = 0; i < num_keys; ++i) {
            file.seekg(offsets[i], std::ios::beg);
            vsag::Binary b;
            readBinaryPOD(file, b.size);
            // std::cout << "len: " << b.size << std::endl;
            b.data.reset(new int8_t[b.size]);
            file.read((char*)b.data.get(), b.size);
            bs.Set(keys[i], b);
        }

        if (auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
            index.has_value()) {
            hnsw = index.value();
        } else {
            std::cout << "Build HNSW Error" << std::endl;
            return;
        }
        hnsw->Deserialize(bs);
    }

    // Deserialize(readerset)
    {
        std::ifstream file(tmp_dir + "hnsw.index", std::ios::in);
        file.seekg(-sizeof(uint64_t) * 2, std::ios::end);
        uint64_t num_keys, footer_offset;
        readBinaryPOD(file, num_keys);
        readBinaryPOD(file, footer_offset);
        // std::cout << "num_keys: " << num_keys << std::endl;
        // std::cout << "footer_offset: " << footer_offset << std::endl;
        file.seekg(footer_offset, std::ios::beg);

        std::vector<std::string> keys;
        std::vector<uint64_t> offsets;
        for (uint64_t i = 0; i < num_keys; ++i) {
            int64_t key_len;
            readBinaryPOD(file, key_len);
            // std::cout << "key_len: " << key_len << std::endl;
            char key_buf[key_len + 1];
            memset(key_buf, 0, key_len + 1);
            file.read(key_buf, key_len);
            // std::cout << "key: " << key_buf << std::endl;
            keys.push_back(key_buf);

            uint64_t offset;
            readBinaryPOD(file, offset);
            // std::cout << "offset: " << offset << std::endl;
            offsets.push_back(offset);
        }

        vsag::ReaderSet rs;
        for (uint64_t i = 0; i < num_keys; ++i) {
            int64_t size = 0;
            if (i + 1 == num_keys) {
                size = footer_offset;
            } else {
                size = offsets[i + 1];
            }
            size -= (offsets[i] + sizeof(uint64_t));
            auto file_reader = vsag::Factory::CreateLocalFileReader(
                tmp_dir + "hnsw.index", offsets[i] + sizeof(uint64_t), size);
            rs.Set(keys[i], file_reader);
        }

        if (auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
            index.has_value()) {
            hnsw = index.value();
        } else {
            std::cout << "Build HNSW Error" << std::endl;
            return;
        }
        hnsw->Deserialize(rs);
    }

    // Query the elements for themselves and measure recall 1@10
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(data + i * dim)->Owner(false);
        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            correct += vsag::knn_search_recall(data,
                                               ids,
                                               max_elements,
                                               data + i * dim,
                                               dim,
                                               result.value()->GetIds(),
                                               result.value()->GetDim());
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to perform search on index" << std::endl;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}
 */
