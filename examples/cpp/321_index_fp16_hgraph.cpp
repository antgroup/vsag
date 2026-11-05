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

// This example demonstrates how to build and search a HGraph index with FP16 input vectors.
// For BF16 input vectors, use BFloat16Vectors() instead of Float16Vectors(), FloatToBF16()
// instead of FloatToFP16(), and "bfloat16" instead of "float16" in the build parameters.

#include <vsag/vsag.h>

#include <iostream>
#include <random>

#include "simd/fp16_simd.h"

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset (FP16) *****************/
    int64_t num_vectors = 10000;
    int64_t dim = 128;
    std::vector<int64_t> ids(num_vectors);
    std::vector<uint16_t> datas(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_float(-1.0f, 1.0f);
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        float val = distrib_float(rng);
        datas[i] = vsag::generic::FloatToFP16(val);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float16Vectors(datas.data())
        ->Owner(false);

    /******************* Create HGraph Index *****************/
    std::string hgraph_build_parameters = R"(
    {
        "dtype": "float16",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "pq",
            "max_degree": 26,
            "ef_construction": 100,
            "alpha": 1.2
        }
    }
    )";
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();

    /******************* Build HGraph Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HGraph contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    /******************* Prepare Query Dataset (FP16) *****************/
    std::vector<uint16_t> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        float val = distrib_float(rng);
        query_vector[i] = vsag::generic::FloatToFP16(val);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float16Vectors(query_vector.data())->Owner(false);

    /******************* KnnSearch For HGraph Index *****************/
    auto hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, hgraph_search_parameters).value();

    /******************* Print Search Result *****************/
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    engine.Shutdown();
    return 0;
}
