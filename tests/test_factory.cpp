
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

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <iostream>

#ifndef VSAG_MOCKIMPL_TEST
#include "functest.h"
#include "json_wrapper.h"
#endif
#include "vsag/vsag.h"

TEST_CASE("Test Factory", "[ft][factory]") {
    int dim = 16;
    int max_elements = 1000;
    int max_degree = 16;
    int ef_construction = 100;
    int ef_search = 100;

    auto index_parameters = R"(
        {
            "dim": 16,
            "dtype": "float32",
            "hnsw": {
                "ef_construction": 100,
                "ef_search": 100,
                "max_degree": 16
            },
            "metric_type": "l2"
        }
    )";
    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters);
    REQUIRE(index.has_value());
    hnsw = index.value();
    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib_real;
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

        std::string parameters = R"(
            {
                "hnsw": {
                    "ef_search": 100
                }
            }
        )";
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters); result.has_value()) {
            if (result.value()->GetIds()[0] == i) {
                correct++;
            }
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to search on index: internalError" << std::endl;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}

#ifndef VSAG_MOCKIMPL_TEST

TEST_CASE("check correct build parameters", "[ft][factory]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto json_string = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 512,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        },
        "diskann": {
            "max_degree": 16,
            "ef_construction": 200,
            "pq_dims": 32,
            "pq_sample_rate": 0.5
        }
    }
    )";
    auto res = vsag::check_diskann_hnsw_build_parameters(json_string);
    REQUIRE(res.has_value());
}

TEST_CASE("check incorrect build parameters", "[ft][factory]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto json_string = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 512,
        "hnsw": {
            "max_degree": 16
        },
        "diskann": {
            "max_degree": 16,
            "ef_construction": 200,
            "pq_dims": 32,
            "pq_sample_rate": 0.5
        }
    }
    )";
    auto res = vsag::check_diskann_hnsw_build_parameters(json_string);
    REQUIRE_FALSE(res.has_value());
    REQUIRE(res.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("check correct search parameters", "[ft][factory]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto json_string = R"(
        {
            "hnsw": {
                "ef_search": 100
            },
            "diskann": {
                "ef_search": 200,
                "beam_search": 4,
                "io_limit": 200,
                "use_reorder": true
           }
        }
        )";
    auto res = vsag::check_diskann_hnsw_search_parameters(json_string);
    REQUIRE(res.has_value());
}

TEST_CASE("check incorrect search parameters", "[ft][factory]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto json_string = R"(
        {
            "hhhhhhhhhhhhhh": {
                "ef_search": 100
            },
            "diskann": {
                "ef_search": 200,
                "beam_search": 4,
                "io_limit": 200,
                "use_reorder": true
           }
        }
        )";
    auto res = vsag::check_diskann_hnsw_search_parameters(json_string);
    REQUIRE_FALSE(res.has_value());
    REQUIRE(res.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("generate build parameters", "[ft][factory]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto metric_type = GENERATE("l2", "IP");
    auto num_elements = GENERATE(1'000'000,
                                 2'000'000,
                                 3'000'000,
                                 4'000'000,
                                 5'000'000,
                                 6'000'000,
                                 7'000'000,
                                 8'000'000,
                                 9'000'000,
                                 10'000'000,
                                 11'000'000);
    auto dim = GENERATE(32, 64, 96, 128, 256, 512, 768, 1024, 1536, 2048, 4096);

    auto parameters = vsag::generate_build_parameters(metric_type, num_elements, dim);

    REQUIRE(parameters.has_value());
    auto json = vsag::JsonWrapper::Parse(parameters.value());
    REQUIRE(json["dim"].GetInt() == dim);
    REQUIRE(json["diskann"]["pq_dims"].GetInt() == dim / 4);
}

TEST_CASE("estimate search cost", "[ft][factory]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    constexpr auto search_parameters_json = R"(
    {{
        "hnsw": {{
            "ef_search": {}
        }}
    }}
    )";

    SECTION("small dataset") {
        auto dim = 128;
        auto ef_search = 100;
        auto data_num = 1'000'000;
        auto search_parameters = fmt::format(search_parameters_json, ef_search);
        auto result = vsag::estimate_search_time("hnsw", data_num, dim, search_parameters);
        REQUIRE(result.has_value());
        REQUIRE(fixtures::time_t(result.value()) == 1);
    }

    SECTION("large dataset") {
        auto dim = 512;
        auto ef_search = 300;
        auto data_num = 10'000'000;
        auto search_parameters = fmt::format(search_parameters_json, ef_search);
        auto result = vsag::estimate_search_time("hnsw", data_num, dim, search_parameters);
        REQUIRE(result.has_value());
        REQUIRE(fixtures::time_t(result.value()) == 24);
    }

    SECTION("unsupported index operation") {
        auto dim = 512;
        auto ef_search = 300;
        auto data_num = 10'000'000;
        auto search_parameters = fmt::format(search_parameters_json, ef_search);
        auto result = vsag::estimate_search_time("diskann", data_num, dim, search_parameters);
        REQUIRE(result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
    }

    SECTION("invalid argument") {
        auto dim = 512;
        auto data_num = 10'000'000;
        auto search_parameters = fmt::format(search_parameters_json, "\"hhhhh\"");
        auto result = vsag::estimate_search_time("hnsw", data_num, dim, search_parameters);
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("generate build parameters with invalid num_elements", "[ft][factory]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto metric_type = GENERATE("l2", "IP");
    auto num_elements = GENERATE(-1'000'000, -1, 0, 17'000'001, 1'000'000'000);
    int64_t dim = 128;

    auto parameters = vsag::generate_build_parameters(metric_type, num_elements, dim);

    REQUIRE(not parameters.has_value());
    REQUIRE(parameters.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("generate build parameters with invalid dim", "[ft][factory]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto metric_type = GENERATE("l2", "IP");
    int64_t num_elements = 1'000'000;
    int64_t dim = GENERATE(1, 3, 42, 61, 90);

    auto parameters = vsag::generate_build_parameters(metric_type, num_elements, dim);

    REQUIRE(not parameters.has_value());
    REQUIRE(parameters.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("build index with generated_build_parameters", "[ft][factory][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    int64_t num_vectors = 1000;
    int64_t dim = 64;

    auto params = vsag::generate_build_parameters("l2", num_vectors, dim);
    REQUIRE(params.has_value());
    auto index_opt = vsag::Factory::CreateIndex("hnsw", params.value());
    REQUIRE(index_opt.has_value());
    auto index = index_opt.value();

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    REQUIRE(index->Build(base).has_value());

    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        },
        "diskann": {
            "ef_search": 100,
            "beam_search": 4,
            "io_limit": 100,
            "use_reorder": false
        }
    }
    )";

    int64_t correct = 0;
    for (int64_t i = 0; i < num_vectors; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + i * dim)->Owner(false);
        auto result_opt = index->KnnSearch(query, 10, search_parameters);
        REQUIRE(result_opt.has_value());
        auto result = result_opt.value();
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            if (i == result->GetIds()[j]) {
                ++correct;
                break;
            }
        }
    }

    float recall = 1.0 * correct / num_vectors;
    std::cout << "recall: " << recall << std::endl;
    REQUIRE(recall > 0.95);
}

#endif
