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
#include <vsag/vsag.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int
main() {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    constexpr int64_t dim = 128;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> datas(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->Owner(false);

    /******************* Create HGraph Index *****************/
    auto hgraph_build_parameters = fmt::format(R"(
    {{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100,
            "alpha":1.2
        }}
    }}
    )",
                                               dim);
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    vsag::Options::Instance().set_block_size_limit(2 * 1024 * 1024);
    auto created = engine.CreateIndex("hgraph", hgraph_build_parameters);
    if (not created.has_value()) {
        std::cerr << "Failed to create index: " << created.error().message << std::endl;
        return EXIT_FAILURE;
    }
    auto index = created.value();

    /******************* Build HGraph Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HGraph contains: " << index->GetNumElements()
                  << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        return EXIT_FAILURE;
    }

    /******************* Prepare Query Dataset *****************/
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    /******************* Search With Timeout *****************/
    // Set timeout_ms to 0 to trigger immediate timeout
    auto timeout_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100,
            "timeout_ms": 0
        }
    }
    )";
    int64_t topk = 10;
    auto search = index->KnnSearch(query, topk, timeout_search_parameters);
    if (not search.has_value()) {
        std::cerr << "Timeout search failed: " << search.error().message << std::endl;
        return EXIT_FAILURE;
    }
    auto result = search.value();

    auto stats = result->GetStatistics({"is_timeout"});
    std::cout << "Search with timeout_ms=0: is_timeout = "
              << (stats.size() == 1 ? stats[0] : "missing is_timeout statistic") << std::endl;

    /******************* Search Without Timeout (Normal) *****************/
    auto normal_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    search = index->KnnSearch(query, topk, normal_search_parameters);
    if (not search.has_value()) {
        std::cerr << "Normal search failed: " << search.error().message << std::endl;
        return EXIT_FAILURE;
    }
    result = search.value();

    stats = result->GetStatistics({"is_timeout"});
    std::cout << "Search without timeout_ms: is_timeout = "
              << (stats.size() == 1 ? stats[0] : "missing is_timeout statistic") << std::endl;

    // Search results store the neighbor count in GetDim(), not GetNumElements().
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    engine.Shutdown();
    return 0;
}
