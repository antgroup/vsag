
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

    /******************* Create Hnsw Index *****************/
    auto hnsw_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();

    /******************* Build Hnsw Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index Hnsw contains: " << index->GetNumElements() << std::endl;
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

    auto hnsw_search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";

    /******************* Get Threshold *****************/
    auto result = index->KnnSearch(query, topk, hnsw_search_parameters);
    if (not result.has_value()) {
        std::cerr << "Search Error: " << result.error().message << std::endl;
    }
    float threshold = result.value()->GetDistances()[5];

    /******************* RangeSearch *****************/
    auto range_result = index->RangeSearch(query, threshold, hnsw_search_parameters);
    if (not range_result.has_value()) {
        std::cerr << "Search Error: " << range_result.error().message << std::endl;
    }
    auto final_result = range_result.value();

    /******************* Print Search Result *****************/
    std::cout << "threshold:" << threshold << std::endl;
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < final_result->GetDim(); ++i) {
        std::cout << final_result->GetIds()[i] << ": " << final_result->GetDistances()[i]
                  << std::endl;
    }

    return 0;
}
