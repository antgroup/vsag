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

#include <vsag/vsag.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "autotune.h"

namespace {

constexpr int64_t DIM = 32;
constexpr int64_t EASY_COUNT = 512;
constexpr int64_t HARD_COUNT = 4096;
constexpr int64_t QUERY_COUNT = 512;
constexpr int64_t TOP_K = 10;

struct PathWorkload {
    std::vector<float> query_vectors;
    std::vector<int64_t> ground_truth_ids;
    std::vector<std::string> query_paths;
    vsag::DatasetPtr queries;
    vsag::DatasetPtr ground_truth;
};

float
L2(const float* left, const float* right) {
    float result = 0.0F;
    for (int64_t i = 0; i < DIM; ++i) {
        const auto difference = left[i] - right[i];
        result += difference * difference;
    }
    return result;
}

void
PrepareWorkload(PathWorkload& workload,
                const std::vector<int64_t>& base_ids,
                const std::vector<float>& base_vectors,
                int64_t begin,
                int64_t count,
                const std::string& path,
                std::mt19937& random) {
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    workload.query_vectors.resize(QUERY_COUNT * DIM);
    workload.ground_truth_ids.resize(QUERY_COUNT * TOP_K);
    workload.query_paths.assign(QUERY_COUNT, path);

    for (int64_t query = 0; query < QUERY_COUNT; ++query) {
        auto* query_vector = workload.query_vectors.data() + query * DIM;
        for (int64_t column = 0; column < DIM; ++column) {
            query_vector[column] = unit(random);
        }

        std::vector<std::pair<float, int64_t>> ranked;
        ranked.reserve(count);
        for (int64_t row = begin; row < begin + count; ++row) {
            ranked.emplace_back(L2(query_vector, base_vectors.data() + row * DIM), base_ids[row]);
        }
        std::partial_sort(ranked.begin(), ranked.begin() + TOP_K, ranked.end());
        for (int64_t k = 0; k < TOP_K; ++k) {
            workload.ground_truth_ids[query * TOP_K + k] = ranked[k].second;
        }
    }

    workload.queries = vsag::Dataset::Make()
                           ->NumElements(QUERY_COUNT)
                           ->Dim(DIM)
                           ->Float32Vectors(workload.query_vectors.data())
                           ->Paths(workload.query_paths.data())
                           ->Owner(false);
    workload.ground_truth = vsag::Dataset::Make()
                                ->NumElements(QUERY_COUNT)
                                ->Dim(TOP_K)
                                ->Ids(workload.ground_truth_ids.data())
                                ->Owner(false);
}

tl::expected<vsag::autotune::SearchResult, vsag::Error>
TunePath(const vsag::IndexPtr& index, const PathWorkload& workload) {
    vsag::autotune::SearchRequest request;
    request.index = index;
    request.workload = {workload.queries, workload.ground_truth, TOP_K, 1};
    request.parameter_space = R"({"pyramid":{"ef_search":[10,20,40,80,160]}})";
    request.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.80}};
    request.objective = vsag::autotune::Metric::LATENCY_AVG_MS;
    request.config.max_trials = 5;
    return vsag::autotune::TuneSearch(request);
}

void
PrintResult(const std::string& path, const vsag::autotune::SearchResult& result) {
    std::cout << "\npath: " << path << '\n'
              << "recommended search_params: " << result.parameters << '\n'
              << "validated metrics: " << result.metrics.dump() << '\n'
              << "trials evaluated: " << result.report["trials"].size() << std::endl;
}

}  // namespace

int
main() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    vsag::Options::Instance().set_block_size_limit(2UL * 1024 * 1024);
    std::mt19937 random(47);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);

    constexpr int64_t BASE_COUNT = EASY_COUNT + HARD_COUNT;
    std::vector<int64_t> base_ids(BASE_COUNT);
    std::vector<float> base_vectors(BASE_COUNT * DIM);
    std::vector<std::string> base_paths(BASE_COUNT);
    for (int64_t row = 0; row < BASE_COUNT; ++row) {
        base_ids[row] = 1000 + row;
        auto* vector = base_vectors.data() + row * DIM;
        base_paths[row] = row < EASY_COUNT ? "catalog/easy" : "catalog/hard";
        for (int64_t column = 0; column < DIM; ++column) {
            vector[column] = unit(random);
        }
    }

    auto base = vsag::Dataset::Make()
                    ->NumElements(BASE_COUNT)
                    ->Dim(DIM)
                    ->Ids(base_ids.data())
                    ->Float32Vectors(base_vectors.data())
                    ->Paths(base_paths.data())
                    ->Owner(false);

    const std::string create_params =
        R"({
            "dim": 32,
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {
                "base_quantization_type": "fp32",
                "max_degree": 16,
                "alpha": 1.2,
                "graph_iter_turn": 5,
                "neighbor_sample_rate": 0.2,
                "no_build_levels": [0, 1],
                "use_reorder": true,
                "graph_type": "odescent",
                "build_thread_count": 8
            }
        })";
    auto created = vsag::Factory::CreateIndex("pyramid", create_params);
    if (!created.has_value()) {
        std::cerr << "Failed to create Pyramid: " << created.error().message << std::endl;
        return 1;
    }
    auto index = created.value();
    auto built = index->Build(base);
    if (!built.has_value()) {
        std::cerr << "Failed to build Pyramid: " << built.error().message << std::endl;
        return 1;
    }

    PathWorkload easy_workload;
    PathWorkload hard_workload;
    PrepareWorkload(easy_workload, base_ids, base_vectors, 0, EASY_COUNT, "catalog/easy", random);
    PrepareWorkload(
        hard_workload, base_ids, base_vectors, EASY_COUNT, HARD_COUNT, "catalog/hard", random);

    auto easy_result = TunePath(index, easy_workload);
    auto hard_result = TunePath(index, hard_workload);
    if (!easy_result.has_value() || !hard_result.has_value()) {
        const auto message =
            !easy_result.has_value() ? easy_result.error().message : hard_result.error().message;
        std::cerr << "AutoTune failed: " << message << std::endl;
        return 1;
    }
    if (easy_result->status == vsag::autotune::TuneStatus::NO_FEASIBLE_CANDIDATE ||
        hard_result->status == vsag::autotune::TuneStatus::NO_FEASIBLE_CANDIDATE) {
        const auto& result =
            easy_result->status == vsag::autotune::TuneStatus::NO_FEASIBLE_CANDIDATE
                ? easy_result.value()
                : hard_result.value();
        std::cerr << "No candidate satisfied one path workload. Best effort:\n"
                  << result.best_effort.dump(2) << std::endl;
        return 2;
    }

    std::cout << "The same Pyramid index is tuned once per representative path workload."
              << std::endl;
    PrintResult("catalog/easy (512 random vectors)", easy_result.value());
    PrintResult("catalog/hard (4096 random vectors)", hard_result.value());
    return 0;
}
