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

#include "autotune.h"

#include <H5Cpp.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "autotune_internal.h"
#include "eval_dataset.h"
#include "vsag/options.h"
#include "vsag/vsag.h"

namespace {

using vsag::autotune::JsonType;

std::string
temp_path(const std::string& stem, const std::string& extension = "") {
    static std::atomic<uint64_t> serial{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return (std::filesystem::temp_directory_path() /
            (stem + "-" + std::to_string(now) + "-" + std::to_string(serial++) + extension))
        .string();
}

class ScopedPath {
public:
    explicit ScopedPath(std::string path) : path_(std::move(path)) {
    }

    ~ScopedPath() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::string&
    Get() const {
        return path_;
    }

private:
    std::string path_;
};

class ScopedBlockSizeLimit {
public:
    explicit ScopedBlockSizeLimit(uint64_t value)
        : original_(vsag::Options::Instance().block_size_limit()) {
        vsag::Options::Instance().set_block_size_limit(value);
    }

    ~ScopedBlockSizeLimit() {
        vsag::Options::Instance().set_block_size_limit(original_);
    }

private:
    uint64_t original_;
};

class ScopedOpenMpThreads {
public:
    explicit ScopedOpenMpThreads(int value) : original_(omp_get_max_threads()) {
        omp_set_num_threads(value);
    }

    ~ScopedOpenMpThreads() {
        omp_set_num_threads(original_);
    }

private:
    int original_;
};

uint64_t
count_index_artifacts(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        return 0;
    }
    uint64_t count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".index") {
            ++count;
        }
    }
    return count;
}

std::vector<JsonType>
load_json_reports(const std::string& path) {
    std::vector<JsonType> reports;
    if (!std::filesystem::exists(path)) {
        return reports;
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") {
            continue;
        }
        std::ifstream input(entry.path());
        JsonType report;
        input >> report;
        reports.emplace_back(std::move(report));
    }
    return reports;
}

float
l2(const std::vector<float>& train,
   const std::vector<float>& test,
   int64_t base,
   int64_t query,
   int64_t dim) {
    float distance = 0.0F;
    for (int64_t i = 0; i < dim; ++i) {
        const auto difference = train[base * dim + i] - test[query * dim + i];
        distance += difference * difference;
    }
    return std::sqrt(distance);
}

void
write_dataset(const std::string& path) {
    constexpr int64_t BASE_COUNT = 64;
    constexpr int64_t QUERY_COUNT = 6;
    constexpr int64_t DIM = 8;
    constexpr int64_t GROUND_TRUTH_K = 6;
    std::vector<float> train(BASE_COUNT * DIM);
    std::vector<float> test(QUERY_COUNT * DIM);
    for (int64_t i = 0; i < BASE_COUNT; ++i) {
        for (int64_t j = 0; j < DIM; ++j) {
            train[i * DIM + j] = static_cast<float>((i * 17 + j * 13) % 101) / 101.0F;
        }
    }
    for (int64_t i = 0; i < QUERY_COUNT; ++i) {
        for (int64_t j = 0; j < DIM; ++j) {
            test[i * DIM + j] = train[((i * 7) % BASE_COUNT) * DIM + j];
        }
    }

    std::vector<int64_t> neighbors(QUERY_COUNT * GROUND_TRUTH_K);
    std::vector<float> distances(QUERY_COUNT * GROUND_TRUTH_K);
    for (int64_t query = 0; query < QUERY_COUNT; ++query) {
        std::vector<std::pair<float, int64_t>> ranked;
        for (int64_t base = 0; base < BASE_COUNT; ++base) {
            ranked.emplace_back(l2(train, test, base, query, DIM), base);
        }
        std::sort(ranked.begin(), ranked.end());
        for (int64_t k = 0; k < GROUND_TRUTH_K; ++k) {
            neighbors[query * GROUND_TRUTH_K + k] = ranked[k].second;
            distances[query * GROUND_TRUTH_K + k] = ranked[k].first;
        }
    }

    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    auto distance = file.createAttribute("distance", string_type, H5::DataSpace(H5S_SCALAR));
    std::string metric = "euclidean";
    distance.write(string_type, metric);
    const auto write_matrix = [&](const std::string& name,
                                  const auto* data,
                                  const H5::DataType& type,
                                  int64_t rows,
                                  int64_t columns) {
        hsize_t dimensions[2] = {static_cast<hsize_t>(rows), static_cast<hsize_t>(columns)};
        H5::DataSpace space(2, dimensions);
        auto dataset = file.createDataSet(name, type, space);
        dataset.write(data, type);
    };
    write_matrix("/train", train.data(), H5::PredType::NATIVE_FLOAT, BASE_COUNT, DIM);
    write_matrix("/test", test.data(), H5::PredType::NATIVE_FLOAT, QUERY_COUNT, DIM);
    write_matrix(
        "/neighbors", neighbors.data(), H5::PredType::NATIVE_INT64, QUERY_COUNT, GROUND_TRUTH_K);
    write_matrix(
        "/distances", distances.data(), H5::PredType::NATIVE_FLOAT, QUERY_COUNT, GROUND_TRUTH_K);
}

JsonType
request(const std::string& dataset, const std::string& workspace) {
    return {{"version", 1},
            {"data_path", dataset},
            {"indexes",
             JsonType::array({{{"name", "hgraph"},
                               {"create_params",
                                {{"index_param",
                                  {{"base_quantization_type", "fp32"},
                                   {"max_degree", 8},
                                   {"ef_construction", 40},
                                   {"build_thread_count", 2}}}}},
                               {"search_params", {{"hgraph", {{"ef_search", {8, 16}}}}}}},
                              {{"name", "ivf"},
                               {"create_params",
                                {{"index_param",
                                  {{"base_quantization_type", "fp32"},
                                   {"buckets_count", 4},
                                   {"thread_count", 2}}}}},
                               {"search_params", {{"ivf", {{"scan_buckets_count", {1, 4}}}}}}}})},
            {"workload", {{"top_k", 3}, {"concurrency", 2}}},
            {"constraints", {{"recall_at_k", 0.0}, {"build_seconds", 1000.0}}},
            {"objective", {{"metric", "latency_avg_ms"}}},
            {"tuning_config",
             {{"workspace_path", workspace}, {"keep_intermediate", true}, {"max_trials", 4}}}};
}

struct MemoryFixture {
    static constexpr int64_t BASE_COUNT = 64;
    static constexpr int64_t QUERY_COUNT = 6;
    static constexpr int64_t DIM = 8;
    static constexpr int64_t GROUND_TRUTH_K = 6;

    MemoryFixture()
        : base_ids(BASE_COUNT),
          train(BASE_COUNT * DIM),
          test(QUERY_COUNT * DIM),
          neighbors(QUERY_COUNT * GROUND_TRUTH_K),
          distances(QUERY_COUNT * GROUND_TRUTH_K) {
        for (int64_t i = 0; i < BASE_COUNT; ++i) {
            base_ids[i] = 1001 + i * 17;
            for (int64_t j = 0; j < DIM; ++j) {
                train[i * DIM + j] = static_cast<float>((i * 17 + j * 13) % 101) / 101.0F;
            }
        }
        for (int64_t i = 0; i < QUERY_COUNT; ++i) {
            for (int64_t j = 0; j < DIM; ++j) {
                test[i * DIM + j] = train[((i * 7) % BASE_COUNT) * DIM + j];
            }
        }
        for (int64_t query = 0; query < QUERY_COUNT; ++query) {
            std::vector<std::pair<float, int64_t>> ranked;
            for (int64_t row = 0; row < BASE_COUNT; ++row) {
                ranked.emplace_back(l2(train, test, row, query, DIM), row);
            }
            std::sort(ranked.begin(), ranked.end());
            for (int64_t k = 0; k < GROUND_TRUTH_K; ++k) {
                neighbors[query * GROUND_TRUTH_K + k] = base_ids[ranked[k].second];
                distances[query * GROUND_TRUTH_K + k] = ranked[k].first;
            }
        }
        base = vsag::Dataset::Make()
                   ->NumElements(BASE_COUNT)
                   ->Dim(DIM)
                   ->Ids(base_ids.data())
                   ->Float32Vectors(train.data())
                   ->Owner(false);
        queries = vsag::Dataset::Make()
                      ->NumElements(QUERY_COUNT)
                      ->Dim(DIM)
                      ->Float32Vectors(test.data())
                      ->Owner(false);
        ground_truth = vsag::Dataset::Make()
                           ->NumElements(QUERY_COUNT)
                           ->Dim(GROUND_TRUTH_K)
                           ->Ids(neighbors.data())
                           ->Distances(distances.data())
                           ->Owner(false);
    }

    vsag::autotune::IndexRequest
    Request(const std::string& workspace) const {
        vsag::autotune::IndexRequest result;
        result.base = base;
        result.metric_type = vsag::METRIC_L2;
        result.workload = {queries, ground_truth, 3, 2};
        result.index_spaces = {
            {"hgraph",
             R"({"index_param":{"base_quantization_type":"fp32",)"
             R"("max_degree":[8,12],"ef_construction":40,"build_thread_count":2}})",
             R"({"hgraph":{"ef_search":16}})"}};
        result.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.0},
                              {vsag::autotune::Metric::BUILD_SECONDS, 1000.0}};
        result.objective = vsag::autotune::Metric::LATENCY_AVG_MS;
        result.config.workspace_path = workspace;
        result.config.max_trials = 2;
        return result;
    }

    std::vector<int64_t> base_ids;
    std::vector<float> train;
    std::vector<float> test;
    std::vector<int64_t> neighbors;
    std::vector<float> distances;
    vsag::DatasetPtr base;
    vsag::DatasetPtr queries;
    vsag::DatasetPtr ground_truth;
};

}  // namespace

TEST_CASE("AutoTune candidate rules only fill missing fields") {
    vsag::autotune::internal::IndexTuningRequest request;
    request.context.base_count = 10000;
    request.context.top_k = 10;
    request.context.max_trials = 10;
    request.indexes = {
        {"hgraph",
         {{"dim", 8},
          {"dtype", "float32"},
          {"metric_type", "l2"},
          {"index_param",
           {{"base_quantization_type", "fp32"}, {"max_degree", 24}, {"ef_construction", 80}}}},
         JsonType::object()}};

    const auto candidates = vsag::autotune::internal::GenerateCandidates(request);
    REQUIRE(candidates.size() == 3);
    for (const auto& candidate : candidates) {
        REQUIRE(candidate.create_params["index_param"]["max_degree"] == 24);
        REQUIRE(candidate.create_params["index_param"]["ef_construction"] == 80);
    }

    request.indexes[0].create_params = JsonType::object();
    request.context.max_trials = 24;
    REQUIRE(vsag::autotune::internal::GenerateCandidates(request).size() == 24);

    request.indexes[0].create_params = candidates[0].create_params;
    request.indexes[0].search_params = {
        {"hgraph", {{"ef_search", {{"$range", {{"start", 0.1}, {"stop", 0.3}, {"step", 0.1}}}}}}}};
    const auto float_range = vsag::autotune::internal::GenerateCandidates(request);
    REQUIRE(float_range.size() == 3);
    REQUIRE(float_range.back().search_params["hgraph"]["ef_search"] == 0.3);
    REQUIRE_FALSE(float_range.back().ef_search_range.has_value());

    const auto minimum = std::numeric_limits<int64_t>::min();
    request.indexes[0].search_params = {
        {"hgraph",
         {{"ef_search", {{"$range", {{"start", minimum}, {"stop", minimum + 5}, {"step", 10}}}}}}}};
    const auto integer_range = vsag::autotune::internal::GenerateCandidates(request);
    REQUIRE(integer_range.size() == 1);
    REQUIRE(integer_range[0].search_params["hgraph"]["ef_search"] == minimum);

    constexpr double large_start = 1e15;
    request.indexes[0].search_params = {
        {"hgraph",
         {{"ef_search",
           {{"$range", {{"start", large_start}, {"stop", large_start + 1.0}, {"step", 0.25}}}}}}}};
    const auto large_float_range = vsag::autotune::internal::GenerateCandidates(request);
    REQUIRE(large_float_range.size() == 5);
    REQUIRE(large_float_range.front().search_params["hgraph"]["ef_search"] == large_start);
    REQUIRE(large_float_range.back().search_params["hgraph"]["ef_search"] == large_start + 1.0);

    request.indexes[0].search_params = {
        {"hgraph",
         {{"ef_search",
           {{"$range",
             {{"start", large_start},
              {"stop", std::nextafter(large_start, std::numeric_limits<double>::infinity())},
              {"step", 0.01}}}}}}}};
    REQUIRE_THROWS_WITH(vsag::autotune::internal::GenerateCandidates(request),
                        Catch::Matchers::ContainsSubstring("$range step is too small to advance"));

    const auto oversized = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + uint64_t{1};
    request.indexes[0].search_params = {
        {"hgraph",
         {{"ef_search", {{"$range", {{"start", oversized}, {"stop", oversized}, {"step", 1}}}}}}}};
    REQUIRE_THROWS_WITH(vsag::autotune::internal::GenerateCandidates(request),
                        Catch::Matchers::ContainsSubstring("$range integer values must fit int64"));
}

TEST_CASE("AutoTune proposes conservative defaults for existing IVF indexes") {
    vsag::autotune::internal::SearchTuningRequest request;
    request.context.top_k = 10;
    request.context.max_trials = 4;
    request.index_input.name = "ivf";

    const auto candidates = vsag::autotune::internal::GenerateCandidates(request);
    REQUIRE(candidates.size() == 4);
    const std::vector<uint64_t> expected{1, 4, 16, 64};
    for (uint64_t i = 0; i < candidates.size(); ++i) {
        REQUIRE(candidates[i].search_params["ivf"]["scan_buckets_count"] == expected[i]);
    }
}

TEST_CASE("AutoTune validates IVF buckets_count before proposing search candidates") {
    vsag::autotune::internal::IndexTuningRequest request;
    request.context.base_count = 10000;
    request.context.top_k = 10;
    request.context.max_trials = 3;
    request.indexes = {{"ivf", JsonType::object(), JsonType::object()}};

    for (const auto& buckets : {JsonType(-1), JsonType(1024.0)}) {
        request.indexes[0].create_params = {{"index_param", {{"buckets_count", buckets}}}};
        REQUIRE_THROWS_WITH(
            vsag::autotune::internal::GenerateCandidates(request),
            Catch::Matchers::ContainsSubstring(
                "ivf create_params.index_param.buckets_count must be a positive integer"));
    }
}

TEST_CASE("AutoTune plans an adaptive HGraph ef_search range") {
    vsag::autotune::internal::IndexTuningRequest request;
    request.context.base_count = 10000;
    request.context.top_k = 10;
    request.context.objective = "latency_avg_ms";
    request.context.constraints = {{"recall_at_k", 0.95}};
    request.context.max_trials = 30;
    request.indexes = {
        {"hgraph",
         {{"dim", 8},
          {"dtype", "float32"},
          {"metric_type", "l2"},
          {"index_param",
           {{"base_quantization_type", "fp32"}, {"max_degree", 24}, {"ef_construction", 80}}}},
         {{"hgraph",
           {{"ef_search", {{"$range", {{"start", 40}, {"stop", 1000}}}}},
            {"use_reorder", {true, false}}}}}}};

    const auto candidates = vsag::autotune::internal::GenerateCandidates(request);
    REQUIRE(candidates.size() == 2);
    for (const auto& candidate : candidates) {
        REQUIRE(candidate.ef_search_range.has_value());
        const auto range = candidate.ef_search_range.value_or(
            vsag::autotune::internal::HGraphEfSearchRange{-1, -1});
        REQUIRE(range.start == 40);
        REQUIRE(range.stop == 1000);
        REQUIRE_FALSE(candidate.search_params["hgraph"].contains("ef_search"));
        REQUIRE(candidate.search_params["hgraph"]["use_reorder"].is_boolean());
    }

    request.context.max_trials = 29;
    REQUIRE_THROWS_WITH(
        vsag::autotune::internal::GenerateCandidates(request),
        Catch::Matchers::ContainsSubstring("planned trial count exceeds tuning_config.max_trials"));

    request.context.max_trials = 1;
    request.indexes[0].search_params = {
        {"hgraph", {{"ef_search", {{"$range", {{"start", 8}, {"stop", 8}}}}}}}};
    REQUIRE(vsag::autotune::internal::GenerateCandidates(request).size() == 1);

    request.context.constraints = {{"latency_avg_ms", 1.0}};
    request.indexes[0].search_params = {
        {"hgraph", {{"ef_search", {{"$range", {{"start", 8}, {"stop", 16}}}}}}}};
    request.context.max_trials = 5;
    REQUIRE_THROWS_WITH(vsag::autotune::internal::GenerateCandidates(request),
                        Catch::Matchers::ContainsSubstring("requires a recall_at_k constraint"));

    request.context.constraints = {{"recall_at_k", 0.95}};
    request.context.objective = "recall_at_k";
    REQUIRE_THROWS_WITH(vsag::autotune::internal::GenerateCandidates(request),
                        Catch::Matchers::ContainsSubstring("requires a query-cost objective"));

    request.context.objective = "latency_avg_ms";
    request.indexes[0].search_params["hgraph"]["hops_limit"] = 100;
    REQUIRE_THROWS_WITH(
        vsag::autotune::internal::GenerateCandidates(request),
        Catch::Matchers::ContainsSubstring("does not support timeout_ms or hops_limit"));

    request.indexes[0].search_params["hgraph"].erase("hops_limit");
    request.indexes[0].search_params["hgraph"]["ef_search"]["$range"]["stop"] = 1001;
    request.context.max_trials = 30;
    const auto unbounded = vsag::autotune::internal::GenerateCandidates(request);
    REQUIRE(unbounded.size() == 1);
    REQUIRE(unbounded[0].ef_search_range.has_value());
    REQUIRE(unbounded[0].ef_search_range->stop == 1001);
}

TEST_CASE("AutoTune brackets and binary-searches the smallest passing ef_search") {
    const vsag::autotune::internal::HGraphEfSearchRange range{8, 64};
    std::vector<int64_t> evaluated;
    const auto evaluate = [&evaluated](int64_t ef_search) -> std::optional<double> {
        evaluated.emplace_back(ef_search);
        return ef_search >= 25 ? 0.95 : 0.90;
    };

    vsag::autotune::internal::EvaluateEfSearchRange(range, 0.95, evaluate);
    REQUIRE(evaluated == std::vector<int64_t>{8, 16, 32, 24, 28, 26, 25});

    evaluated.clear();
    vsag::autotune::internal::EvaluateEfSearchRange(range, 0.90, evaluate);
    REQUIRE(evaluated == std::vector<int64_t>{8});

    evaluated.clear();
    vsag::autotune::internal::EvaluateEfSearchRange({8, 50}, 1.0, evaluate);
    REQUIRE(evaluated == std::vector<int64_t>{8, 16, 32, 50});

    evaluated.clear();
    const auto fail_at_middle = [&evaluated](int64_t ef_search) -> std::optional<double> {
        evaluated.emplace_back(ef_search);
        if (ef_search == 24) {
            return std::nullopt;
        }
        return ef_search >= 32 ? 0.95 : 0.90;
    };
    vsag::autotune::internal::EvaluateEfSearchRange(range, 0.95, fail_at_middle);
    REQUIRE(evaluated == std::vector<int64_t>{8, 16, 32, 24});
}

TEST_CASE("AutoTune validates typed requests and optional ground truth") {
    MemoryFixture fixture;
    auto input = fixture.Request(temp_path("autotune-typed-validation"));

    const auto parsed = vsag::autotune::internal::ParseRequest(input);
    REQUIRE(parsed.context.dataset->GetTrainIds()[0] == fixture.base_ids[0]);
    REQUIRE(parsed.context.dataset->GetTrainIds()[1] == fixture.base_ids[1]);
    REQUIRE_FALSE(parsed.context.allow_quantization_tune);
    REQUIRE(parsed.context.effective_request["config"]["allow_quantization_tune"] == false);

    input.config.allow_quantization_tune = true;
    const auto quantization_tune = vsag::autotune::internal::ParseRequest(input);
    REQUIRE(quantization_tune.context.allow_quantization_tune);
    REQUIRE(quantization_tune.context.effective_request["config"]["allow_quantization_tune"] ==
            true);

    input.constraints = {{vsag::autotune::Metric::LATENCY_AVG_MS, 1000.0}};
    input.workload.ground_truth = nullptr;
    REQUIRE_NOTHROW(vsag::autotune::internal::ParseRequest(input));

    input.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.9}};
    REQUIRE_THROWS_WITH(vsag::autotune::internal::ParseRequest(input),
                        Catch::Matchers::ContainsSubstring("ground_truth is required"));

    input.workload.ground_truth = fixture.ground_truth;
    input.constraints.push_back({vsag::autotune::Metric::RECALL_AT_K, 0.8});
    REQUIRE_THROWS_WITH(vsag::autotune::internal::ParseRequest(input),
                        Catch::Matchers::ContainsSubstring("duplicate metric"));

    input = fixture.Request(temp_path("autotune-typed-duplicate-ids"));
    auto duplicate_ids = fixture.base_ids;
    duplicate_ids[1] = duplicate_ids[0];
    input.base = vsag::Dataset::Make()
                     ->NumElements(MemoryFixture::BASE_COUNT)
                     ->Dim(MemoryFixture::DIM)
                     ->Ids(duplicate_ids.data())
                     ->Float32Vectors(fixture.train.data())
                     ->Owner(false);
    REQUIRE_THROWS_WITH(vsag::autotune::internal::ParseRequest(input),
                        Catch::Matchers::ContainsSubstring("duplicate ids"));

    input = fixture.Request(temp_path("autotune-typed-unknown-ground-truth"));
    auto unknown_ids = fixture.neighbors;
    unknown_ids[0] = -1;
    input.workload.ground_truth = vsag::Dataset::Make()
                                      ->NumElements(MemoryFixture::QUERY_COUNT)
                                      ->Dim(MemoryFixture::GROUND_TRUTH_K)
                                      ->Ids(unknown_ids.data())
                                      ->Owner(false);
    REQUIRE_THROWS_WITH(vsag::autotune::internal::ParseRequest(input),
                        Catch::Matchers::ContainsSubstring("not present in base"));

    input = fixture.Request(temp_path("autotune-typed-duplicate-search-namespace"));
    input.index_spaces[0].search_parameter_space =
        R"({"HGRAPH":{"ef_search":8},"hgraph":{"ef_search":16}})";
    REQUIRE_THROWS_WITH(vsag::autotune::internal::ParseRequest(input),
                        Catch::Matchers::ContainsSubstring("duplicate index namespace"));
}

TEST_CASE("AutoTune returns structured offline validation failures") {
    ScopedPath invalid_dataset(temp_path("autotune-invalid-dataset", ".hdf5"));
    {
        std::ofstream output(invalid_dataset.Get(), std::ios::binary);
        output << "not an HDF5 file";
    }
    H5::Exception::dontPrint();

    const auto invalid_file =
        vsag::autotune::RunAutoTune({{"version", 1}, {"data_path", invalid_dataset.Get()}});
    REQUIRE(invalid_file["status"] == "failed");
    REQUIRE(invalid_file["failure"]["stage"] == "validation");
    REQUIRE(invalid_file["failure"]["code"] == "invalid_request");
    REQUIRE_THAT(invalid_file["failure"]["message"].get<std::string>(),
                 Catch::Matchers::ContainsSubstring("failed to load evaluation dataset"));

    const auto invalid_version = vsag::autotune::RunAutoTune({{"version", 2}});
    REQUIRE(invalid_version["status"] == "failed");
    REQUIRE(invalid_version["failure"]["stage"] == "validation");
    REQUIRE(invalid_version["failure"]["code"] == "invalid_request");
}

TEST_CASE("AutoTune serializes concurrent offline dataset loading") {
    ScopedPath dataset(temp_path("autotune-concurrent-dataset", ".hdf5"));
    write_dataset(dataset.Get());
    const JsonType input{{"version", 1}, {"data_path", dataset.Get()}};
    std::atomic<uint64_t> ready{0};
    std::atomic<bool> start{false};
    JsonType results[2];

    const auto run = [&](uint64_t position) {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        results[position] = vsag::autotune::RunAutoTune(input);
    };
    std::thread first(run, 0);
    std::thread second(run, 1);
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    first.join();
    second.join();

    for (const auto& result : results) {
        REQUIRE(result["status"] == "failed");
        REQUIRE(result["failure"]["stage"] == "validation");
        REQUIRE(result["failure"]["code"] == "invalid_request");
        REQUIRE_THAT(result["failure"]["message"].get<std::string>(),
                     Catch::Matchers::ContainsSubstring("request.workload is required"));
    }
}

TEST_CASE("AutoTune keeps normalized offline request metadata") {
    ScopedPath dataset(temp_path("autotune-normalized-dataset", ".hdf5"));
    ScopedPath workspace(temp_path("autotune-normalized-workspace"));
    write_dataset(dataset.Get());
    auto input = request(dataset.Get(), workspace.Get());
    input.erase("indexes");

    const auto default_parsed = vsag::autotune::internal::ParseRequest(input);
    const auto& default_request =
        std::get<vsag::autotune::internal::IndexTuningRequest>(default_parsed);
    REQUIRE_FALSE(default_request.context.allow_quantization_tune);
    REQUIRE(default_request.context.effective_request["config"]["allow_quantization_tune"] ==
            false);

    input["tuning_config"]["allow_quantization_tune"] = true;

    const auto parsed = vsag::autotune::internal::ParseRequest(input);
    const auto& index_request = std::get<vsag::autotune::internal::IndexTuningRequest>(parsed);
    const auto& effective = index_request.context.effective_request;
    const auto* train_ids = index_request.context.dataset->GetTrainIds();
    REQUIRE(train_ids != nullptr);
    REQUIRE(train_ids[0] == 0);
    REQUIRE(train_ids[63] == 63);
    REQUIRE(effective["data_path"] == dataset.Get());
    REQUIRE(effective["dataset"]["dim"] == 8);
    REQUIRE(effective["dataset"]["dtype"] == "float32");
    REQUIRE(effective["dataset"]["metric_type"] == "l2");
    REQUIRE(effective["index_spaces"].size() == 2);
    REQUIRE(effective["index_spaces"][0]["name"] == "hgraph");
    REQUIRE(effective["index_spaces"][1]["name"] == "ivf");
    for (const auto& space : effective["index_spaces"]) {
        REQUIRE(space["create_parameter_space"]["dim"] == 8);
        REQUIRE(space["create_parameter_space"]["dtype"] == "float32");
        REQUIRE(space["create_parameter_space"]["metric_type"] == "l2");
    }
    REQUIRE(effective["workload"]["concurrency"] == 2);
    REQUIRE(index_request.context.allow_quantization_tune);
    REQUIRE(effective["config"]["allow_quantization_tune"] == true);
    REQUIRE(effective["config"]["keep_intermediate"] == true);
    REQUIRE_FALSE(effective.contains("tuning_config"));

    input["tuning_config"]["allow_quantization_tune"] = 1;
    REQUIRE_THROWS_WITH(vsag::autotune::internal::ParseRequest(input),
                        Catch::Matchers::ContainsSubstring(
                            "request.tuning_config.allow_quantization_tune must be a boolean"));
}

TEST_CASE("AutoTune rejects index artifacts that fail to flush") {
    if (!std::filesystem::exists("/dev/full")) {
        return;
    }

    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath run_path(temp_path("autotune-full-device"));
    std::filesystem::create_directories(run_path.Get() + "/artifacts");
    std::error_code link_error;
    std::filesystem::create_symlink(
        "/dev/full", run_path.Get() + "/artifacts/build-0.index", link_error);
    REQUIRE_FALSE(link_error);

    MemoryFixture fixture;
    auto input = fixture.Request(run_path.Get());
    input.index_spaces[0].create_parameter_space =
        R"({"index_param":{"base_quantization_type":"fp32","max_degree":8,)"
        R"("ef_construction":40,"build_thread_count":2}})";
    input.config.max_trials = 1;
    const auto request = vsag::autotune::internal::ParseRequest(input);
    const auto candidates = vsag::autotune::internal::GenerateCandidates(request);
    REQUIRE(candidates.size() == 1);

    const auto evaluation =
        vsag::autotune::internal::EvaluateCandidates(request, candidates, run_path.Get());
    REQUIRE(evaluation.builds.size() == 1);
    REQUIRE(evaluation.builds[0]["status"] == "failed");
    REQUIRE_THAT(evaluation.builds[0]["failure"]["message"].get<std::string>(),
                 Catch::Matchers::ContainsSubstring("failed to write index artifact"));
    REQUIRE(evaluation.trials.size() == 1);
    REQUIRE(evaluation.trials[0]["failure"]["code"] == "build_failed");
}

TEST_CASE("AutoTune rejects report paths that alias an existing index") {
    ScopedPath dataset(temp_path("autotune-alias-dataset", ".hdf5"));
    ScopedPath workspace(temp_path("autotune-alias-workspace"));
    ScopedPath index_path(temp_path("autotune-alias-index", ".index"));
    ScopedPath report_path(temp_path("autotune-alias-report", ".json"));
    write_dataset(dataset.Get());
    {
        std::ofstream output(index_path.Get(), std::ios::binary);
        output << "serialized-index-placeholder";
    }
    std::error_code link_error;
    std::filesystem::create_hard_link(index_path.Get(), report_path.Get(), link_error);
    REQUIRE_FALSE(link_error);
    const auto original_size = std::filesystem::file_size(index_path.Get());

    auto input = request(dataset.Get(), workspace.Get());
    input["indexes"] = JsonType::array({input["indexes"][0]});
    input["index_path"] = index_path.Get();
    input["output"] = {{"result_path", report_path.Get()}};
    const auto result = vsag::autotune::RunAutoTune(input);

    REQUIRE(result["status"] == "failed");
    REQUIRE(result["failure"]["stage"] == "validation");
    REQUIRE(result["failure"]["code"] == "invalid_request");
    REQUIRE_THAT(result["failure"]["message"].get<std::string>(),
                 Catch::Matchers::ContainsSubstring("result_path must not alias index_path"));
    REQUIRE(std::filesystem::file_size(index_path.Get()) == original_size);
}

TEST_CASE("AutoTune reports the closest successful trial when constraints are infeasible") {
    vsag::autotune::internal::RequestContext request;
    request.top_k = 10;
    request.objective = "latency_avg_ms";
    request.constraints = {{"recall_at_k", 0.95}};
    vsag::autotune::internal::Evaluation evaluation;
    const auto trial = [](const std::string& id, double recall, double latency) {
        return JsonType{{"trial_id", id},
                        {"build_id", "build-0"},
                        {"index_name", "hgraph"},
                        {"create_params", JsonType::object()},
                        {"search_params", JsonType::object()},
                        {"status", "success"},
                        {"metrics", {{"recall_at_k", recall}, {"latency_avg_ms", latency}}},
                        {"artifacts", JsonType::object()}};
    };
    evaluation.trials = {
        trial("trial-0", 0.90, 1.0), trial("trial-1", 0.80, 2.0), trial("trial-2", 0.85, 3.0)};

    const auto result = vsag::autotune::internal::SelectResult(request, evaluation);
    REQUIRE(result["status"] == "no_feasible_candidate");
    REQUIRE(result["recommendation"].is_null());
    REQUIRE(result["best_effort"]["evidence"]["trial_id"] == "trial-0");
}

TEST_CASE("AutoTune distinguishes a missing objective metric from failed trials") {
    vsag::autotune::internal::RequestContext request;
    request.top_k = 10;
    request.objective = "index_memory_mb";
    request.constraints = {{"recall_at_k", 0.0}};
    vsag::autotune::internal::Evaluation evaluation;
    evaluation.trials = {{{"trial_id", "trial-0"},
                          {"index_name", "pyramid"},
                          {"search_params", JsonType::object()},
                          {"status", "success"},
                          {"metrics", {{"recall_at_k", 1.0}}}}};

    const auto result = vsag::autotune::internal::SelectResult(request, evaluation);
    REQUIRE(result["status"] == "failed");
    REQUIRE(result["failure"]["stage"] == "selection");
    REQUIRE(result["failure"]["code"] == "objective_metric_unavailable");
    REQUIRE_THAT(result["failure"]["message"].get<std::string>(),
                 Catch::Matchers::ContainsSubstring("index_memory_mb"));
}

TEST_CASE("AutoTune builds once per create candidate and supports an existing index") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedOpenMpThreads openmp_threads(3);
    ScopedPath dataset(temp_path("autotune-dataset", ".hdf5"));
    ScopedPath workspace(temp_path("autotune-workspace"));
    write_dataset(dataset.Get());

    auto input = request(dataset.Get(), workspace.Get());
    input["tuning_config"]["allow_quantization_tune"] = true;
    const auto result = vsag::autotune::RunAutoTune(input);
    INFO(result.dump(2));
    REQUIRE(result["status"] == "success");
    REQUIRE(omp_get_max_threads() == 3);
    REQUIRE(result["builds"].size() == 2);
    REQUIRE(result["trials"].size() == 4);
    REQUIRE(result["recommendation"]["create_params"]["dim"] == 8);
    REQUIRE(result["request"]["config"]["allow_quantization_tune"] == true);
    for (const auto& build : result["builds"]) {
        REQUIRE(build["status"] == "success");
        REQUIRE(build["metrics"].contains("build_seconds"));
        REQUIRE(build["metrics"].contains("index_size_mb"));
        REQUIRE(build["metrics"].contains("index_memory_mb"));
    }
    for (const auto& trial : result["trials"]) {
        REQUIRE(trial["status"] == "success");
        REQUIRE(trial["metrics"].contains("recall_at_k"));
        REQUIRE(trial["metrics"].contains("latency_avg_ms"));
        REQUIRE(trial["metrics"].contains("qps"));
    }

    const auto hgraph =
        std::find_if(result["builds"].begin(), result["builds"].end(), [](const auto& build) {
            return build["index_name"] == "hgraph";
        });
    REQUIRE(hgraph != result["builds"].end());
    auto existing = request(dataset.Get(), workspace.Get());
    existing["index_path"] = (*hgraph)["artifacts"]["index_path"];
    existing["indexes"] =
        JsonType::array({{{"name", "hgraph"},
                          {"create_params", (*hgraph)["create_params"]},
                          {"search_params", {{"hgraph", {{"ef_search", {8, 16}}}}}}}});
    existing["constraints"] = {{"recall_at_k", 0.0}};
    existing["tuning_config"]["max_trials"] = 2;

    const auto existing_result = vsag::autotune::RunAutoTune(existing);
    INFO(existing_result.dump(2));
    REQUIRE(existing_result["status"] == "success");
    REQUIRE(existing_result["builds"].empty());
    REQUIRE(existing_result["trials"].size() == 2);
    REQUIRE(existing_result["request"]["data_path"] == dataset.Get());
    REQUIRE(existing_result["request"]["index_path"] == existing["index_path"]);
    REQUIRE(existing_result["request"]["index_name"] == "hgraph");
    REQUIRE(existing_result["request"]["create_params"]["dim"] == 8);
    REQUIRE(existing_result["request"]["create_params"]["dtype"] == "float32");
    REQUIRE(existing_result["request"]["create_params"]["metric_type"] == "l2");
}

TEST_CASE("AutoTune derives HGraph degree and quantization candidates from canonical builds") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-hgraph-reuse-workspace"));
    MemoryFixture fixture;
    auto input = fixture.Request(workspace.Get());
    input.index_spaces[0].create_parameter_space =
        R"({"index_param":{"base_quantization_type":["fp32","sq8_uniform"],)"
        R"("max_degree":[8,12],"ef_construction":40,"build_thread_count":2,)"
        R"("store_raw_vector":[false,true]}})";
    input.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.0}};
    input.objective = vsag::autotune::Metric::INDEX_SIZE_MB;
    input.config.allow_quantization_tune = true;
    input.config.keep_intermediate = true;
    input.config.max_trials = 8;

    const auto tuned = vsag::autotune::TuneIndex(input);
    REQUIRE(tuned.has_value());
    const auto& report = tuned->report;
    INFO(report.dump(2));
    REQUIRE(report["builds"].size() == 8);
    REQUIRE(report["trials"].size() == 8);
    REQUIRE(std::all_of(report["builds"].begin(), report["builds"].end(), [](const auto& build) {
        return build["status"] == "success" && build["strategy"] == "hgraph_tune";
    }));
    REQUIRE(std::all_of(report["builds"].begin(), report["builds"].end(), [](const auto& build) {
        return build.contains("source_build_id") && build.contains("transform_seconds");
    }));
    REQUIRE(count_index_artifacts(workspace.Get()) == 8);
    std::vector<std::string> source_ids;
    std::string keep_raw_source;
    std::string discard_raw_source;
    for (const auto& build : report["builds"]) {
        const auto source_id = build["source_build_id"].get<std::string>();
        source_ids.emplace_back(source_id);
        const auto keep_raw = build["create_params"]["index_param"]["store_raw_vector"].get<bool>();
        auto& expected_source = keep_raw ? keep_raw_source : discard_raw_source;
        if (expected_source.empty()) {
            expected_source = source_id;
        } else {
            REQUIRE(expected_source == source_id);
        }
    }
    std::sort(source_ids.begin(), source_ids.end());
    source_ids.erase(std::unique(source_ids.begin(), source_ids.end()), source_ids.end());
    REQUIRE(source_ids.size() == 2);
    REQUIRE(keep_raw_source != discard_raw_source);

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(MemoryFixture::DIM)
                     ->Float32Vectors(fixture.test.data())
                     ->Owner(false);
    for (const auto& build : report["builds"]) {
        const auto create_params = build["create_params"];
        REQUIRE(create_params["index_param"]["store_raw_vector"].is_boolean());
        auto restored = vsag::Factory::CreateIndex("hgraph", create_params.dump());
        REQUIRE(restored.has_value());
        std::ifstream artifact(build["artifacts"]["index_path"].get<std::string>(),
                               std::ios::binary);
        REQUIRE(artifact.good());
        REQUIRE(restored.value()->Deserialize(artifact).has_value());
        REQUIRE(
            restored.value()->KnnSearch(query, 3, R"({"hgraph":{"ef_search":16}})").has_value());
    }
}

TEST_CASE("AutoTune keeps HGraph quantization build families separate by default") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-hgraph-degree-reuse-workspace"));
    MemoryFixture fixture;
    auto input = fixture.Request(workspace.Get());
    input.index_spaces[0].create_parameter_space =
        R"({"index_param":{"base_quantization_type":["fp32","sq8_uniform"],)"
        R"("max_degree":[8,12],"ef_construction":40,"build_thread_count":2}})";
    input.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.0}};
    input.objective = vsag::autotune::Metric::INDEX_SIZE_MB;
    input.config.keep_intermediate = true;
    input.config.max_trials = 4;

    const auto tuned = vsag::autotune::TuneIndex(input);
    REQUIRE(tuned.has_value());
    const auto& report = tuned->report;
    INFO(report.dump(2));
    REQUIRE(report["request"]["config"]["allow_quantization_tune"] == false);
    REQUIRE(report["builds"].size() == 4);
    REQUIRE(report["trials"].size() == 4);

    std::string fp32_source;
    std::string sq8_source;
    for (const auto& build : report["builds"]) {
        REQUIRE(build["status"] == "success");
        REQUIRE(build["strategy"] == "hgraph_tune");
        const auto quantization =
            build["create_params"]["index_param"]["base_quantization_type"].get<std::string>();
        REQUIRE((quantization == "fp32" || quantization == "sq8_uniform"));
        auto& source = quantization == "fp32" ? fp32_source : sq8_source;
        const auto source_id = build["source_build_id"].get<std::string>();
        if (source.empty()) {
            source = source_id;
        } else {
            REQUIRE(source == source_id);
        }
    }
    REQUIRE_FALSE(fp32_source.empty());
    REQUIRE_FALSE(sq8_source.empty());
    REQUIRE(fp32_source != sq8_source);
}

TEST_CASE("AutoTune disables HGraph build reuse for build-cost metrics") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-hgraph-native-build-workspace"));
    MemoryFixture fixture;
    auto input = fixture.Request(workspace.Get());
    input.index_spaces[0].create_parameter_space =
        R"({"index_param":{"base_quantization_type":["fp32","sq8_uniform"],)"
        R"("max_degree":[8,12],"ef_construction":40,"build_thread_count":2}})";
    input.config.keep_intermediate = true;
    input.config.max_trials = 4;

    const auto require_native_builds = [](const auto& tuned) {
        REQUIRE(tuned.has_value());
        const auto& report = tuned->report;
        INFO(report.dump(2));
        REQUIRE(report["builds"].size() == 4);
        REQUIRE(std::all_of(report["builds"].begin(),
                            report["builds"].end(),
                            [](const auto& build) { return build["strategy"] == "full_build"; }));
    };

    SECTION("constraint") {
        require_native_builds(vsag::autotune::TuneIndex(input));
    }

    SECTION("objective") {
        input.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.0}};
        input.objective = vsag::autotune::Metric::BUILD_SECONDS;
        require_native_builds(vsag::autotune::TuneIndex(input));
    }
}

TEST_CASE("AutoTune CLI keeps only the recommended artifact by default") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath dataset(temp_path("autotune-retained-dataset", ".hdf5"));
    ScopedPath workspace(temp_path("autotune-retained-workspace"));
    write_dataset(dataset.Get());
    auto input = request(dataset.Get(), workspace.Get());
    input["indexes"] = JsonType::array({input["indexes"][0]});
    input["indexes"][0]["search_params"]["hgraph"]["ef_search"] = 8;
    input["tuning_config"].erase("keep_intermediate");
    input["tuning_config"]["max_trials"] = 1;

    const auto result = vsag::autotune::RunAutoTune(input);
    INFO(result.dump(2));
    REQUIRE(result["status"] == "success");
    REQUIRE(result["recommendation"]["artifacts"]["retained"] == true);
    REQUIRE(std::filesystem::is_regular_file(
        result["recommendation"]["artifacts"]["index_path"].get<std::string>()));
    REQUIRE(count_index_artifacts(workspace.Get()) == 1);
    REQUIRE(std::filesystem::is_regular_file(result["report_path"].get<std::string>()));
}

TEST_CASE("AutoTune searches an in-memory existing index") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-memory-index-workspace"));
    MemoryFixture fixture;
    const std::string create_params =
        R"({"dim":8,"dtype":"float32","metric_type":"l2","index_param":{)"
        R"("base_quantization_type":"fp32","max_degree":8,"ef_construction":40,)"
        R"("build_thread_count":2}})";
    auto created = vsag::Factory::CreateIndex("hgraph", create_params);
    REQUIRE(created.has_value());
    REQUIRE(created.value()->Build(fixture.base).has_value());

    vsag::autotune::SearchRequest input;
    input.index = created.value();
    input.workload = {fixture.queries, fixture.ground_truth, 3, 2};
    input.parameter_space = R"({"hgraph":{"ef_search":[8,16]}})";
    input.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.0}};
    input.objective = vsag::autotune::Metric::LATENCY_AVG_MS;
    input.config.workspace_path = workspace.Get();
    input.config.max_trials = 2;

    const auto tuned = vsag::autotune::TuneSearch(input);
    REQUIRE(tuned.has_value());
    REQUIRE(tuned->status == vsag::autotune::TuneStatus::SUCCESS);
    const auto& result = tuned->report;
    REQUIRE(result["status"] == "success");
    REQUIRE(result["builds"].empty());
    REQUIRE(result["trials"].size() == 2);
    for (const auto& trial : result["trials"]) {
        REQUIRE(trial["status"] == "success");
        REQUIRE(trial["metrics"].contains("recall_at_k"));
        REQUIRE(trial["metrics"].contains("latency_avg_ms"));
    }

    input.workload.ground_truth = nullptr;
    input.constraints = {{vsag::autotune::Metric::QPS, 0.0}};
    const auto latency_only = vsag::autotune::TuneSearch(input);
    REQUIRE(latency_only.has_value());
    REQUIRE(latency_only->status == vsag::autotune::TuneStatus::SUCCESS);
    REQUIRE_FALSE(latency_only->metrics.contains("recall_at_k"));
    REQUIRE_FALSE(latency_only->report.contains("report_path"));
    REQUIRE(load_json_reports(workspace.Get()).empty());

    input.objective = vsag::autotune::Metric::INDEX_MEMORY_MB;
    REQUIRE_THROWS_WITH(
        vsag::autotune::internal::ParseRequest(input),
        Catch::Matchers::ContainsSubstring("objective cannot rank search candidates"));
    input.objective = vsag::autotune::Metric::LATENCY_AVG_MS;

    input.parameter_space = R"({"ivf":{"scan_buckets_count":1}})";
    REQUIRE_THROWS_WITH(
        vsag::autotune::internal::ParseRequest(input),
        Catch::Matchers::ContainsSubstring("search_parameter_space.ivf is unsupported"));
}

TEST_CASE("AutoTune uses default candidates for an in-memory IVF index") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-memory-ivf-workspace"));
    MemoryFixture fixture;
    const std::string create_params =
        R"({"dim":8,"dtype":"float32","metric_type":"l2","index_param":{)"
        R"("base_quantization_type":"fp32","buckets_count":4,"thread_count":2}})";
    auto created = vsag::Factory::CreateIndex("ivf", create_params);
    REQUIRE(created.has_value());
    REQUIRE(created.value()->Build(fixture.base).has_value());

    vsag::autotune::SearchRequest input;
    input.index = created.value();
    input.workload = {fixture.queries, fixture.ground_truth, 3, 2};
    input.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.0}};
    input.objective = vsag::autotune::Metric::LATENCY_AVG_MS;
    input.config.workspace_path = workspace.Get();
    input.config.max_trials = 4;

    const auto tuned = vsag::autotune::TuneSearch(input);
    REQUIRE(tuned.has_value());
    REQUIRE(tuned->status == vsag::autotune::TuneStatus::SUCCESS);
    const auto& result = tuned->report;
    REQUIRE(result["status"] == "success");
    REQUIRE(result["trials"].size() == 4);
    REQUIRE(result["recommendation"]["search_params"]["ivf"].contains("scan_buckets_count"));
}

TEST_CASE("AutoTune searches an existing Pyramid index for one path workload") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-pyramid-workspace"));
    MemoryFixture fixture;
    std::vector<std::string> base_paths(MemoryFixture::BASE_COUNT, "a/d/f");
    std::vector<std::string> query_paths(MemoryFixture::QUERY_COUNT, "a/d/f");
    auto base = vsag::Dataset::Make()
                    ->NumElements(MemoryFixture::BASE_COUNT)
                    ->Dim(MemoryFixture::DIM)
                    ->Ids(fixture.base_ids.data())
                    ->Float32Vectors(fixture.train.data())
                    ->Paths(base_paths.data())
                    ->Owner(false);
    auto queries = vsag::Dataset::Make()
                       ->NumElements(MemoryFixture::QUERY_COUNT)
                       ->Dim(MemoryFixture::DIM)
                       ->Float32Vectors(fixture.test.data())
                       ->Paths(query_paths.data())
                       ->Owner(false);
    const std::string create_params =
        R"({"dim":8,"dtype":"float32","metric_type":"l2","index_param":{)"
        R"("base_quantization_type":"fp32","max_degree":8,"alpha":1.2,)"
        R"("graph_iter_turn":5,"neighbor_sample_rate":0.5,"no_build_levels":[0,1],)"
        R"("use_reorder":true,"graph_type":"odescent","build_thread_count":2}})";
    auto created = vsag::Factory::CreateIndex("pyramid", create_params);
    REQUIRE(created.has_value());
    REQUIRE(created.value()->Build(base).has_value());
    REQUIRE(created.value()->GetMemoryUsage() > 0);

    vsag::autotune::SearchRequest input;
    input.index = created.value();
    input.workload.queries = queries;
    input.workload.ground_truth = fixture.ground_truth;
    input.workload.top_k = 3;
    input.workload.concurrency = 2;
    input.parameter_space = R"({"pyramid":{"ef_search":[4,8]}})";
    input.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.0}};
    input.objective = vsag::autotune::Metric::LATENCY_AVG_MS;
    input.config.workspace_path = workspace.Get();
    input.config.max_trials = 2;

    const auto tuned = vsag::autotune::TuneSearch(input);
    REQUIRE(tuned.has_value());
    REQUIRE(tuned->status == vsag::autotune::TuneStatus::SUCCESS);
    const auto& result = tuned->report;
    REQUIRE(result["status"] == "success");
    REQUIRE(result["trials"].size() == 2);
    REQUIRE(result["recommendation"]["index_name"] == "pyramid");
    REQUIRE(result["recommendation"]["search_params"].contains("pyramid"));
    REQUIRE(result["builds"].empty());

    input.constraints.push_back({vsag::autotune::Metric::INDEX_MEMORY_MB, 0.0});
    const auto memory_constrained = vsag::autotune::TuneSearch(input);
    REQUIRE(memory_constrained.has_value());
    REQUIRE(memory_constrained->status == vsag::autotune::TuneStatus::NO_FEASIBLE_CANDIDATE);
    REQUIRE(memory_constrained->best_effort["constraint_evaluation"]["satisfied"] == false);

    input.constraints.pop_back();
    input.workload.queries = fixture.queries;
    REQUIRE_NOTHROW(vsag::autotune::internal::ParseRequest(input));
}

TEST_CASE("AutoTune writes concrete trials for an adaptive ef_search range") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath dataset(temp_path("autotune-binary-dataset", ".hdf5"));
    ScopedPath workspace(temp_path("autotune-binary-workspace"));
    write_dataset(dataset.Get());

    auto input = request(dataset.Get(), workspace.Get());
    auto hgraph = input["indexes"][0];
    hgraph["search_params"] = {
        {"hgraph", {{"ef_search", {{"$range", {{"start", 3}, {"stop", 35}}}}}}}};
    input["indexes"] = JsonType::array({std::move(hgraph)});
    input["constraints"] = {{"recall_at_k", 0.0}};
    input["tuning_config"]["max_trials"] = 9;

    const auto result = vsag::autotune::RunAutoTune(input);
    INFO(result.dump(2));
    REQUIRE(result["status"] == "success");
    REQUIRE(result["builds"].size() == 1);
    REQUIRE(result["trials"].size() == 1);
    REQUIRE(result["trials"][0]["search_params"]["hgraph"]["ef_search"] == 3);
    REQUIRE(result["trials"][0]["search_params"]["hgraph"]["ef_search"].is_number_integer());
    REQUIRE(result["recommendation"]["search_params"]["hgraph"]["ef_search"] == 3);
}

TEST_CASE("TuneIndex returns a queryable selected index") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-factory-workspace"));
    MemoryFixture fixture;
    auto input = fixture.Request(workspace.Get());

    auto tuned = vsag::autotune::TuneIndex(input);
    REQUIRE(tuned.has_value());
    const auto& result = tuned.value();
    REQUIRE(result.status == vsag::autotune::TuneStatus::SUCCESS);
    REQUIRE(result.best_effort.is_null());
    REQUIRE(result.index != nullptr);
    REQUIRE(result.index->GetNumElements() == 64);
    REQUIRE(result.index_name == "hgraph");
    REQUIRE(JsonType::parse(result.create_parameters)["dim"] == 8);
    REQUIRE(JsonType::parse(result.search_parameters)["hgraph"]["ef_search"] == 16);
    REQUIRE(result.metrics.contains("recall_at_k"));
    REQUIRE(std::filesystem::is_regular_file(result.artifact_path));
    REQUIRE_FALSE(result.report.contains("report_path"));

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(MemoryFixture::DIM)
        ->Float32Vectors(fixture.test.data())
        ->Owner(false);
    auto neighbors = result.index->KnnSearch(query, 3, result.search_parameters);
    REQUIRE(neighbors.has_value());
    REQUIRE(neighbors.value()->GetDim() == 3);
    REQUIRE(std::find(fixture.base_ids.begin(),
                      fixture.base_ids.end(),
                      neighbors.value()->GetIds()[0]) != fixture.base_ids.end());

    auto restored = vsag::Factory::CreateIndex(result.index_name, result.create_parameters);
    REQUIRE(restored.has_value());
    std::ifstream artifact(result.artifact_path, std::ios::binary);
    REQUIRE(artifact.good());
    REQUIRE(restored.value()->Deserialize(artifact).has_value());
    REQUIRE(restored.value()->KnnSearch(query, 3, result.search_parameters).has_value());

    const auto& report = result.report;
    REQUIRE(report["status"] == "success");
    REQUIRE(report["builds"].size() == 2);
    REQUIRE(std::count_if(report["builds"].begin(), report["builds"].end(), [](const auto& build) {
                return build["artifacts"]["retained"].template get<bool>();
            }) == 1);
    REQUIRE(count_index_artifacts(workspace.Get()) == 1);
    REQUIRE(report["recommendation"]["create_params"]["dim"] == 8);
    REQUIRE(report["recommendation"]["artifacts"]["retained"] == true);
    REQUIRE(report["request"]["config"]["workspace_path"] == workspace.Get());
}

TEST_CASE("TuneIndex reports an infeasible recommendation") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-infeasible-workspace"));
    MemoryFixture fixture;
    auto input = fixture.Request(workspace.Get());
    input.index_spaces[0].create_parameter_space =
        R"({"index_param":{"base_quantization_type":"fp32","max_degree":8,)"
        R"("ef_construction":40,"build_thread_count":2}})";
    input.workload.ground_truth = nullptr;
    input.constraints = {{vsag::autotune::Metric::QPS, 1e100}};
    input.config.max_trials = 1;

    const auto tuned = vsag::autotune::TuneIndex(input);
    REQUIRE(tuned.has_value());
    REQUIRE(tuned->status == vsag::autotune::TuneStatus::NO_FEASIBLE_CANDIDATE);
    REQUIRE(tuned->index == nullptr);
    REQUIRE(tuned->best_effort["constraint_evaluation"]["satisfied"] == false);
    REQUIRE_FALSE(tuned->report.contains("report_path"));
    REQUIRE(count_index_artifacts(workspace.Get()) == 0);
    REQUIRE(load_json_reports(workspace.Get()).empty());
    REQUIRE(tuned->report["status"] == "no_feasible_candidate");
    REQUIRE(tuned->report["request"]["config"]["keep_intermediate"] == false);
    REQUIRE(tuned->report["builds"][0]["artifacts"]["retained"] == false);
    REQUIRE(tuned->best_effort["artifacts"]["retained"] == false);
}

TEST_CASE("TuneIndex cleans artifacts when all search trials fail") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-all-failed-workspace"));
    MemoryFixture fixture;
    auto input = fixture.Request(workspace.Get());
    input.index_spaces[0].create_parameter_space =
        R"({"index_param":{"base_quantization_type":"fp32","max_degree":8,)"
        R"("ef_construction":40,"build_thread_count":2}})";
    input.index_spaces[0].search_parameter_space = R"({"hgraph":{"ef_search":0}})";
    input.config.max_trials = 1;

    const auto tuned = vsag::autotune::TuneIndex(input);
    REQUIRE_FALSE(tuned.has_value());
    REQUIRE(tuned.error().type == vsag::ErrorType::INTERNAL_ERROR);
    REQUIRE(count_index_artifacts(workspace.Get()) == 0);
    REQUIRE(load_json_reports(workspace.Get()).empty());
}

TEST_CASE("TuneIndex returns its report without writing a report file") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath workspace(temp_path("autotune-in-memory-report-workspace"));
    MemoryFixture fixture;
    auto input = fixture.Request(workspace.Get());
    input.index_spaces[0].create_parameter_space =
        R"({"index_param":{"base_quantization_type":"fp32","max_degree":8,)"
        R"("ef_construction":40,"build_thread_count":2}})";
    input.config.max_trials = 1;

    const auto tuned = vsag::autotune::TuneIndex(input);
    REQUIRE(tuned.has_value());
    REQUIRE(tuned->status == vsag::autotune::TuneStatus::SUCCESS);
    REQUIRE(tuned->report["status"] == "success");
    REQUIRE_FALSE(tuned->report.contains("report_path"));
    REQUIRE(load_json_reports(workspace.Get()).empty());
}

TEST_CASE("AutoTune validates the CLI report path before evaluation") {
    ScopedPath dataset(temp_path("autotune-report-path-dataset", ".hdf5"));
    ScopedPath workspace(temp_path("autotune-report-path-workspace"));
    write_dataset(dataset.Get());
    const auto report_directory = workspace.Get() + "/report-is-a-directory";
    std::filesystem::create_directories(report_directory);

    auto input = request(dataset.Get(), workspace.Get());
    input["output"] = {{"result_path", report_directory}};
    const auto result = vsag::autotune::RunAutoTune(input);

    REQUIRE(result["status"] == "failed");
    REQUIRE(result["failure"]["stage"] == "report");
    REQUIRE_THAT(result["failure"]["message"].get<std::string>(),
                 Catch::Matchers::ContainsSubstring("failed to open report path"));
    REQUIRE(count_index_artifacts(workspace.Get()) == 0);
}
