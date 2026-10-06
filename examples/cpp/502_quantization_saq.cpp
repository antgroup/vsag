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

#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

int
main() {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    constexpr int64_t num_vectors = 1000;
    constexpr int64_t dim = 128;
    std::vector<int64_t> ids(num_vectors);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<float> data(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distribution;
    for (auto& value : data) {
        value = distribution(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(data.data())
        ->Owner(false);

    /******************* Create HGraph Index *****************/
    const std::string build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "saq",
            "saq_avg_bits": 4,
            "max_degree": 32,
            "ef_construction": 100,
            "use_reorder": true,
            "precise_quantization_type": "fp32"
        }
    }
    )";
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto created = engine.CreateIndex("hgraph", build_parameters);
    if (not created.has_value()) {
        std::cerr << "Failed to create index: " << created.error().message << std::endl;
        return 1;
    }
    auto index = created.value();

    /******************* Train And Build HGraph Index *****************/
    // Build learns the PCA transform and segment plan before encoding the base vectors.
    auto built = index->Build(base);
    if (not built.has_value() or not built.value().empty()) {
        std::cerr << "Failed to build the complete index" << std::endl;
        return 1;
    }
    std::cout << "After Build(), Index HGraph contains: " << index->GetNumElements() << std::endl;

    /******************* Prepare Query Dataset *****************/
    std::vector<float> query_vector(dim);
    for (auto& value : query_vector) {
        value = distribution(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    /******************* KnnSearch For HGraph Index *****************/
    constexpr int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, R"({"hgraph": {"ef_search": 100}})");
    if (not result.has_value()) {
        std::cerr << "Failed to search index: " << result.error().message << std::endl;
        return 1;
    }
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        std::cout << result.value()->GetIds()[i] << ": " << result.value()->GetDistances()[i]
                  << std::endl;
    }

    engine.Shutdown();
    return 0;
}
