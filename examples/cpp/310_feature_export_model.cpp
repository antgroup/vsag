
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

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    int64_t dim = 128;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    // Transfer the ownership of the data (ids, vectors) to the base.
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

    /******************* Create HGraph Index *****************/
    auto hgraph_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "max_degree": 32,
            "ef_construction": 100,
            "base_quantization_type": "sq8"
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hgraph", hgraph_build_paramesters).value();

    /******************* Build Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HGraph contains: " << index->GetNumElements()
                  << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    /******************* Prepare Query *****************/
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);

    auto hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";

    /******************* Get KnnSearch Result *****************/
    auto result = index->KnnSearch(query, topk, hgraph_search_parameters);
    if (not result.has_value()) {
        std::cerr << "Search Error: " << result.error().message << std::endl;
    }

    /******************* Export Model And Build New Index *****************/
    auto index_model_result = index->ExportModel();
    if (not index_model_result.has_value()) {
        std::cerr << "Index Export Model failed: " << index_model_result.error().message
                  << std::endl;
    }
    auto index_model = index_model_result.value();
    for (int64_t i = 0; i < num_vectors; ++i) {
        auto cur_element = vsag::Dataset::Make();
        cur_element->NumElements(1)
            ->Dim(dim)
            ->Ids(ids + i)
            ->Float32Vectors(vectors + i * dim)
            ->Owner(false);
        auto add_result = index_model->Add(cur_element);
        if (not add_result.has_value()) {
            std::cerr << "Failed to add index: " << add_result.error().message << std::endl;
            exit(-1);
        }
    }
    std::cout << "After Add(), Index HGraph contains: " << index_model->GetNumElements()
              << std::endl;

    /******************* Compare KnnSearch Result *****************/
    auto result_new = index_model->KnnSearch(query, topk, hgraph_search_parameters);
    if (not result_new.has_value()) {
        std::cerr << "Search Error: " << result_new.error().message << std::endl;
    }

    std::cout << "origin index results: " << std::endl;
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        std::cout << result.value()->GetIds()[i] << ": " << result.value()->GetDistances()[i]
                  << std::endl;
    }

    std::cout << "new index results: " << std::endl;
    for (int64_t i = 0; i < result_new.value()->GetDim(); ++i) {
        std::cout << result_new.value()->GetIds()[i] << ": "
                  << result_new.value()->GetDistances()[i] << std::endl;
    }

    return 0;
}
