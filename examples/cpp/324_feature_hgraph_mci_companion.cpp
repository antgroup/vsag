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
#include <random>
#include <string>
#include <vector>

class ModFilter : public vsag::Filter {
public:
    bool
    CheckValid(int64_t id) const override {
        return id % 20 == 0;
    }

    float
    ValidRatio() const override {
        return 0.05F;
    }
};

int
main(int argc, char** argv) {
    vsag::init();

    int64_t num_vectors = 2000;
    int64_t dim = 32;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> vectors(num_vectors * dim);

    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = distrib_real(rng);
        }
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    std::string build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 100,
            "alpha": 1.2,
            "mci": {
                "use_mci": true,
                "mcs": 64,
                "clique_max": 32,
                "alpha": 1.2,
                "hgraph_valid_ratio_threshold": 0.2
            }
        }
    }
    )";

    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index = engine.CreateIndex("hgraph", build_parameters).value();

    auto build_result = index->Build(base);
    if (not build_result.has_value()) {
        std::cerr << "Failed to build HGraph with MCI companion: " << build_result.error().message
                  << std::endl;
        return -1;
    }

    auto stats = index->GetStats();
    std::cout << "stats: " << stats << std::endl;

    auto filter = std::make_shared<ModFilter>();
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

    std::string search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 80,
            "use_mci": true,
            "seed_ratio": 0.02,
            "hgraph_valid_ratio_threshold": 0.2
        }
    }
    )";
    auto result = index->KnnSearch(query, 10, search_parameters, filter).value();
    std::cout << "filtered results:" << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    engine.Shutdown();
    return 0;
}
