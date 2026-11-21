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

#include <vsag/vsag_eval_c_api.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "eval_config.h"

namespace {

using nlohmann::json;

constexpr int64_t DIM = 8;
constexpr int64_t BASE_COUNT = 200;
constexpr int64_t QUERY_COUNT = 10;
constexpr int64_t TOP_K = 5;

const char* const BRUTE_FORCE_PARAM =
    R"({"dim":8,"dtype":"float32","metric_type":"l2","index_param":{}})";

struct ErrorDeleter {
    void
    operator()(vsag_error_t* error) const {
        vsag_error_destroy(error);
    }
};
using ErrorPtr = std::unique_ptr<vsag_error_t, ErrorDeleter>;

struct DatasetDeleter {
    void
    operator()(vsag_dataset_t* dataset) const {
        vsag_dataset_destroy(dataset);
    }
};
using DatasetPtr = std::unique_ptr<vsag_dataset_t, DatasetDeleter>;

struct IndexDeleter {
    void
    operator()(vsag_index_t* index) const {
        vsag_index_destroy(index);
    }
};
using IndexPtr = std::unique_ptr<vsag_index_t, IndexDeleter>;

struct StringDeleter {
    void
    operator()(char* text) const {
        vsag_string_destroy(text);
    }
};
using StringPtr = std::unique_ptr<char, StringDeleter>;

DatasetPtr
MakeDataset(int64_t count, const std::vector<float>& vectors, const std::vector<int64_t>& ids) {
    vsag_dataset_t* raw = nullptr;
    ErrorPtr error(vsag_dataset_create(&raw));
    REQUIRE(error == nullptr);
    DatasetPtr dataset(raw);
    REQUIRE(ErrorPtr(vsag_dataset_set_num_elements(raw, count)) == nullptr);
    REQUIRE(ErrorPtr(vsag_dataset_set_dim(raw, DIM)) == nullptr);
    if (not ids.empty()) {
        REQUIRE(ErrorPtr(vsag_dataset_set_ids(raw, ids.data())) == nullptr);
    }
    REQUIRE(ErrorPtr(vsag_dataset_set_float32_vectors(raw, vectors.data())) == nullptr);
    return dataset;
}

struct Fixture {
    Fixture() {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0F, 1.0F);
        base.resize(BASE_COUNT * DIM);
        queries.resize(QUERY_COUNT * DIM);
        for (auto& value : base) {
            value = dist(rng);
        }
        for (auto& value : queries) {
            value = dist(rng);
        }
        ids.resize(BASE_COUNT);
        for (int64_t i = 0; i < BASE_COUNT; ++i) {
            ids[i] = i;
        }
        // Exact neighbours: the queries are copies of the first base vectors.
        for (int64_t q = 0; q < QUERY_COUNT; ++q) {
            for (int64_t d = 0; d < DIM; ++d) {
                queries[q * DIM + d] = base[q * DIM + d];
            }
        }
        base_dataset = MakeDataset(BASE_COUNT, base, ids);
        query_dataset = MakeDataset(QUERY_COUNT, queries, {});
        for (int64_t q = 0; q < QUERY_COUNT; ++q) {
            for (int64_t k = 0; k < TOP_K; ++k) {
                gt.push_back(k == 0 ? q : (q + k) % BASE_COUNT);
            }
        }
        vsag_index_t* raw = nullptr;
        ErrorPtr error(vsag_index_create("brute_force", BRUTE_FORCE_PARAM, &raw));
        REQUIRE(error == nullptr);
        index.reset(raw);
    }

    std::vector<float> base;
    std::vector<float> queries;
    std::vector<int64_t> ids;
    std::vector<int64_t> gt;
    DatasetPtr base_dataset;
    DatasetPtr query_dataset;
    IndexPtr index;
};

json
TakeJson(char* text) {
    StringPtr owned(text);
    REQUIRE(owned != nullptr);
    return json::parse(owned.get());
}

}  // namespace

TEST_CASE("EvalConfig::FromJson reads fields and rejects invalid input", "[eval][config]") {
    auto config = vsag::eval::EvalConfig::FromJson(json{{"index_name", "hgraph"},
                                                        {"build_param", {{"dim", 4}}},
                                                        {"search_param", "{\"a\":1}"},
                                                        {"top_k", 7},
                                                        {"radius", 0.25},
                                                        {"search_query_count", 11},
                                                        {"enable_recall", false},
                                                        {"num_threads_searching", 3}});
    REQUIRE(config.index_name == "hgraph");
    REQUIRE(json::parse(config.build_param) == json{{"dim", 4}});
    REQUIRE(config.search_param == "{\"a\":1}");
    REQUIRE(config.top_k == 7);
    REQUIRE(config.radius == 0.25F);
    REQUIRE(config.search_query_count == 11);
    REQUIRE_FALSE(config.enable_recall);
    REQUIRE(config.num_threads_searching == 3);

    REQUIRE_THROWS_AS(vsag::eval::EvalConfig::FromJson(json::array()), std::invalid_argument);
    REQUIRE_THROWS_AS(vsag::eval::EvalConfig::FromJson(json{{"no_such_key", 1}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(vsag::eval::EvalConfig::FromJson(json{{"top_k", "ten"}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(vsag::eval::EvalConfig::FromJson(json{{"top_k", 1LL << 40}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(vsag::eval::EvalConfig::FromJson(json{{"search_query_count", -1}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(vsag::eval::EvalConfig::FromJson(json{{"enable_qps", 1}}),
                      std::invalid_argument);
}

TEST_CASE("vsag_eval_build and vsag_eval_search measure an in-memory index", "[eval][c_api]") {
    Fixture fixture;
    char* text = nullptr;

    ErrorPtr error(vsag_eval_build(fixture.index.get(),
                                   fixture.base_dataset.get(),
                                   "l2",
                                   R"({"index_name":"brute_force"})",
                                   &text));
    REQUIRE(error == nullptr);
    auto build = TakeJson(text);
    REQUIRE(build["action"] == "build");
    REQUIRE(build["dataset_info"]["base_count"] == BASE_COUNT);
    REQUIRE(build["tps"].get<double>() > 0);

    text = nullptr;
    error.reset(vsag_eval_search(fixture.index.get(),
                                 fixture.base_dataset.get(),
                                 fixture.query_dataset.get(),
                                 fixture.gt.data(),
                                 TOP_K,
                                 "l2",
                                 R"({"top_k":5,"search_param":"{}","search_query_count":20})",
                                 &text));
    REQUIRE(error == nullptr);
    auto search = TakeJson(text);
    REQUIRE(search["action"] == "search");
    REQUIRE(search["qps"].get<double>() > 0);
    REQUIRE(search.contains("recall_avg"));
    REQUIRE(search.contains("latency_detail(ms)"));
}

TEST_CASE("vsag_eval_search supports id based recall without a base dataset", "[eval][c_api]") {
    Fixture fixture;
    char* text = nullptr;
    ErrorPtr error(
        vsag_eval_build(fixture.index.get(), fixture.base_dataset.get(), "l2", nullptr, &text));
    REQUIRE(error == nullptr);
    StringPtr build(text);

    text = nullptr;
    error.reset(vsag_eval_search(
        fixture.index.get(),
        nullptr,
        fixture.query_dataset.get(),
        fixture.gt.data(),
        TOP_K,
        "l2",
        R"({"top_k":1,"search_param":"{}","search_query_count":10,"use_id_based_recall":true})",
        &text));
    REQUIRE(error == nullptr);
    auto search = TakeJson(text);
    REQUIRE(search["recall_avg"].get<double>() > 0.0);
}

TEST_CASE("vsag_eval functions report invalid arguments", "[eval][c_api]") {
    Fixture fixture;
    char* text = nullptr;

    ErrorPtr error(vsag_eval_build(nullptr, fixture.base_dataset.get(), "l2", nullptr, &text));
    REQUIRE(error != nullptr);
    REQUIRE(vsag_error_code(error.get()) == VSAG_INVALID_ARGUMENT);

    error.reset(
        vsag_eval_build(fixture.index.get(), fixture.base_dataset.get(), "l2", "{not json", &text));
    REQUIRE(error != nullptr);
    REQUIRE(vsag_error_code(error.get()) == VSAG_INVALID_ARGUMENT);

    error.reset(vsag_eval_build(
        fixture.index.get(), fixture.base_dataset.get(), "l2", R"({"bogus":1})", &text));
    REQUIRE(error != nullptr);
    REQUIRE(vsag_error_code(error.get()) == VSAG_INVALID_ARGUMENT);

    error.reset(vsag_eval_build(
        fixture.index.get(), fixture.base_dataset.get(), "hamming", nullptr, &text));
    REQUIRE(error != nullptr);

    error.reset(vsag_eval_search(fixture.index.get(),
                                 fixture.base_dataset.get(),
                                 fixture.query_dataset.get(),
                                 fixture.gt.data(),
                                 TOP_K,
                                 "l2",
                                 R"({"top_k":6,"search_param":"{}"})",
                                 &text));
    REQUIRE(error != nullptr);
    REQUIRE(vsag_error_code(error.get()) == VSAG_INVALID_ARGUMENT);
}

TEST_CASE("vsag_eval_set_num_threads_building returns the previous value", "[eval][c_api]") {
    int32_t original = 0;
    ErrorPtr error(vsag_eval_set_num_threads_building(3, &original));
    REQUIRE(error == nullptr);

    int32_t previous = 0;
    error.reset(vsag_eval_set_num_threads_building(original, &previous));
    REQUIRE(error == nullptr);
    REQUIRE(previous == 3);

    error.reset(vsag_eval_set_num_threads_building(0, nullptr));
    REQUIRE(error != nullptr);
    REQUIRE(vsag_error_code(error.get()) == VSAG_INVALID_ARGUMENT);
}
