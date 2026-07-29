// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vsag/vsag.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "autotune.h"

int
main() {
    constexpr int64_t DIM = 4;
    constexpr int64_t BASE_COUNT = 16;
    constexpr int64_t QUERY_COUNT = 4;
    vsag::Options::Instance().set_block_size_limit(2UL * 1024 * 1024);

    std::vector<int64_t> base_ids(BASE_COUNT);
    std::vector<float> base_vectors(BASE_COUNT * DIM);
    for (int64_t i = 0; i < BASE_COUNT; ++i) {
        base_ids[i] = 1000 + i;
        for (int64_t j = 0; j < DIM; ++j) {
            base_vectors[i * DIM + j] = static_cast<float>(i * DIM + j);
        }
    }

    std::vector<float> query_vectors(base_vectors.begin(),
                                     base_vectors.begin() + QUERY_COUNT * DIM);
    std::vector<int64_t> ground_truth_ids(base_ids.begin(), base_ids.begin() + QUERY_COUNT);

    auto base = vsag::Dataset::Make()
                    ->NumElements(BASE_COUNT)
                    ->Dim(DIM)
                    ->Ids(base_ids.data())
                    ->Float32Vectors(base_vectors.data())
                    ->Owner(false);
    auto queries = vsag::Dataset::Make()
                       ->NumElements(QUERY_COUNT)
                       ->Dim(DIM)
                       ->Float32Vectors(query_vectors.data())
                       ->Owner(false);
    auto ground_truth = vsag::Dataset::Make()
                            ->NumElements(QUERY_COUNT)
                            ->Dim(1)
                            ->Ids(ground_truth_ids.data())
                            ->Owner(false);

    const std::string create_params = R"(
        {
            "dim": 4,
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {
                "base_quantization_type": "fp32",
                "max_degree": 8,
                "ef_construction": 40
            }
        })";
    auto created = vsag::Factory::CreateIndex("hgraph", create_params);
    if (!created.has_value()) {
        std::cerr << "Failed to create index: " << created.error().message << std::endl;
        return 1;
    }
    auto index = created.value();
    auto built = index->Build(base);
    if (!built.has_value()) {
        std::cerr << "Failed to build index: " << built.error().message << std::endl;
        return 1;
    }

    vsag::autotune::SearchRequest request;
    request.index = index;
    request.base = base;
    request.metric_type = vsag::METRIC_L2;
    request.workload = {queries, ground_truth, 1, 1};
    request.index_name = "hgraph";
    request.parameter_space = R"({"hgraph":{"ef_search":[4,8,16]}})";
    request.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 1.0}};
    request.objective = vsag::autotune::Metric::LATENCY_AVG_MS;
    request.config.workspace_path = "/tmp/vsag_autotune_existing_index_example";
    request.config.max_trials = 3;

    const auto tuned = vsag::autotune::TuneSearch(request);
    if (!tuned.has_value()) {
        std::cerr << "AutoTune failed: " << tuned.error().message << std::endl;
        return 1;
    }

    const auto& result = tuned.value();
    std::cout << "recommended search_params: " << result.parameters << '\n'
              << "validated metrics: " << result.metrics << '\n'
              << "report: " << result.report_path << std::endl;

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(DIM)
                     ->Float32Vectors(query_vectors.data())
                     ->Owner(false);
    auto neighbors = index->KnnSearch(query, 1, result.parameters);
    if (!neighbors.has_value()) {
        std::cerr << "Search failed: " << neighbors.error().message << std::endl;
        return 1;
    }
    std::cout << "first neighbor id: " << neighbors.value()->GetIds()[0] << std::endl;
    return 0;
}
