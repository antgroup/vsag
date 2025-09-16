
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
#include <iostream>

#include "fixtures/test_logger.h"
#include "vsag/vsag.h"

TEST_CASE("Test Engine", "[ft][engine]") {
    int dim = 16;
    int max_elements = 1000;

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

    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);

    std::shared_ptr<vsag::Index> hnsw;
    auto index = engine.CreateIndex("hnsw", index_parameters);
    REQUIRE(index.has_value());
    hnsw = index.value();
    // Generate random data
    std::mt19937 rng(97);
    std::uniform_real_distribution<float> distrib_real;
    auto* ids = new int64_t[max_elements];
    auto* data = new float[dim * max_elements];
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
            fixtures::logger::error << "failed to search on index: internalError" << std::endl;
        }
    }
    float recall = correct / static_cast<float>(max_elements);

    REQUIRE(recall == 1);

    engine.Shutdown();
}
