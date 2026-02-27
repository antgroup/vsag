
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
#include <vector>

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Multi-Vector Base Dataset *****************/
    // In ColBERT-style retrieval, each document has multiple vectors (one per token)
    // We'll create 100 documents, each with variable number of vectors (5-10)
    int64_t num_docs = 100;
    int64_t dim = 128;

    std::vector<int64_t> ids(num_docs);
    std::vector<uint32_t> vector_counts(num_docs);
    std::vector<float> datas;
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    std::uniform_int_distribution<int> vec_count_dist(5, 10);

    // Generate document IDs and variable vector counts
    uint64_t total_vectors = 0;
    for (int64_t i = 0; i < num_docs; ++i) {
        ids[i] = i;
        vector_counts[i] = vec_count_dist(rng);
        total_vectors += vector_counts[i];
    }

    // Generate all vectors
    datas.reserve(total_vectors * dim);
    for (uint64_t i = 0; i < total_vectors * dim; ++i) {
        datas.push_back(distrib_real(rng));
    }

    // Create dataset with VectorCounts for multi-vector support
    auto base = vsag::Dataset::Make();
    base->NumElements(num_docs)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->VectorCounts(vector_counts.data())  // Specify number of vectors per document
        ->Owner(false);

    std::cout << "Created multi-vector dataset with " << num_docs << " documents" << std::endl;
    std::cout << "Total vectors: " << total_vectors << " (avg " << (double)total_vectors / num_docs
              << " vectors per doc)" << std::endl;

    /******************* Create WARP Index *****************/
    // WARP index for ColBERT-style maxsin similarity
    std::string warp_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "ip",
        "dim": 128
    }
    )";
    auto index = vsag::Factory::CreateIndex("warp", warp_build_parameters).value();

    /******************* Build WARP Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index WARP contains: " << index->GetNumElements()
                  << " documents" << std::endl;
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

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dim)
        ->Float32Vectors(query_vectors.data())
        ->VectorCounts(&query_vec_count)  // Specify query has multiple vectors
        ->Owner(false);

    std::cout << "Query has " << query_vec_count << " vectors" << std::endl;

    /******************* KnnSearch For WARP Index *****************/
    // WARP performs maxsin similarity: for each query vector, find max similarity
    // with any document vector, sum across query vectors
    auto warp_search_parameters = R"({})";
    int64_t topk = 5;
    auto result = index->KnnSearch(query, topk, warp_search_parameters).value();

    /******************* Print Search Result *****************/
    std::cout << "Top-" << topk << " results (maxsin similarity): " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << "  Document " << result->GetIds()[i]
                  << ": score = " << result->GetDistances()[i] << std::endl;
    }

    return 0;
}
