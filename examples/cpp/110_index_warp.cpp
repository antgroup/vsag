
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
#include <memory>
#include <vector>

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Multi-Vector Base Dataset *****************/
    // In ColBERT-style retrieval, each element (document) has multiple vectors (one per token)
    // We'll create 100 elements (documents), each with variable number of vectors (5-10)
    int64_t num_elements = 100;
    int64_t dim = 128;

    auto ids = std::make_unique<int64_t[]>(num_elements);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    std::uniform_int_distribution<int> vec_count_dist(5, 10);

    // Generate MultiVector array: each document owns its own vectors buffer
    vsag::MultiVector* multi_vectors = new vsag::MultiVector[num_elements];
    uint64_t total_vectors = 0;
    for (int64_t i = 0; i < num_elements; ++i) {
        ids[i] = i;
        uint32_t len = vec_count_dist(rng);
        multi_vectors[i].len_ = len;
        multi_vectors[i].vectors_ = new float[len * dim];
        for (uint32_t j = 0; j < len * dim; ++j) {
            multi_vectors[i].vectors_[j] = distrib_real(rng);
        }
        total_vectors += len;
    }

    // Create dataset with MultiVectors API (Owner(true) transfers memory ownership to Dataset)
    vsag::DatasetPtr base = vsag::Dataset::Make();
    base->NumElements(num_elements)
        ->Dim(dim)
        ->Ids(ids.release())
        ->MultiVectors(multi_vectors)
        ->MultiVectorDim(dim)
        ->Owner(true);

    std::cout << "Created multi-vector dataset with " << num_elements << " elements (documents)"
              << std::endl;
    std::cout << "Total vectors: " << total_vectors << " (avg "
              << (double)total_vectors / num_elements << " vectors per element)" << std::endl;

    /******************* Create WARP Index *****************/
    // WARP index for ColBERT-style maxsin similarity
    std::string warp_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "ip",
        "dim": 128
    }
    )";
    std::shared_ptr<vsag::Index> index =
        vsag::Factory::CreateIndex("warp", warp_build_parameters).value();

    /******************* Build WARP Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index WARP contains: " << index->GetNumElements()
                  << " elements (documents)" << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    /******************* Prepare Multi-Vector Query *****************/
    // Query also has multiple vectors (representing query tokens)
    uint32_t query_vec_count = 3;
    std::vector<float> query_vectors(query_vec_count * dim);
    for (uint32_t i = 0; i < query_vec_count * dim; ++i) {
        query_vectors[i] = distrib_real(rng);
    }

    vsag::MultiVector query_multi_vectors;
    query_multi_vectors.len_ = query_vec_count;
    query_multi_vectors.vectors_ = query_vectors.data();

    vsag::DatasetPtr query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dim)
        ->MultiVectors(&query_multi_vectors)
        ->MultiVectorDim(dim)
        ->Owner(false);

    std::cout << "Query has " << query_vec_count << " vectors" << std::endl;

    /******************* KnnSearch For WARP Index *****************/
    // WARP performs maxsin similarity: for each query vector, find max similarity
    // with any document vector, sum across query vectors
    std::string warp_search_parameters = R"({})";
    int64_t topk = 5;
    vsag::DatasetPtr result = index->KnnSearch(query, topk, warp_search_parameters).value();

    /******************* Print Search Result *****************/
    std::cout << "Top-" << topk << " results (maxsin similarity): " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << "  Document " << result->GetIds()[i]
                  << ": score = " << result->GetDistances()[i] << std::endl;
    }

    return 0;
}
