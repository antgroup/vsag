
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

#include <sys/stat.h>
#include <sys/types.h>
#include <vsag/vsag.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include "vsag/binaryset.h"
#include "vsag/options.h"

int
main(int32_t argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    uint32_t num_vectors = 1000;
    uint32_t dim = 128;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (uint32_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (uint64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    /******************* Create an Index *****************/
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    vsag::IndexPtr index = nullptr;
    if (auto create_index = engine.CreateIndex("hgraph", index_paramesters);
        not create_index.has_value()) {
        std::cout << "create index failed: " << create_index.error().message << std::endl;
        abort();
    } else {
        index = *create_index;
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors)->Owner(false);
    if (auto build_index = index->Build(base); not build_index.has_value()) {
        std::cerr << "build index failed: " << build_index.error().message << std::endl;
        abort();
    }
    std::cout << "index contains vectors: " << index->GetNumElements() << std::endl;

    /******************* Save Index to OStream *****************/
    vsag::Options::Instance().set_new_version(true);
    std::ofstream out_stream("/tmp/vsag-persistent-streaming.index");
    auto serialize_result = index->Serialize(out_stream);
    out_stream.close();
    if (not serialize_result.has_value()) {
        std::cerr << serialize_result.error().message << std::endl;
        abort();
    }

    /******************* Load Index from IStream *****************/
    index = nullptr;
    if (auto create_index = engine.CreateIndex("hgraph", index_paramesters);
        not create_index.has_value()) {
        std::cout << "create index failed: " << create_index.error().message << std::endl;
        abort();
    } else {
        index = *create_index;
    }

    std::ifstream in_stream("/tmp/vsag-persistent-streaming.index");
    if (auto deserialize = index->Deserialize(in_stream); not deserialize.has_value()) {
        std::cerr << "load index failed: " << deserialize.error().message << std::endl;
        abort();
    }

    /******************* Search and Print Results *****************/
    auto topk = 10;
    auto query_vector = new float[dim];
    for (uint64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
    auto search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    if (auto knn_search = index->KnnSearch(query, topk, search_parameters);
        not knn_search.has_value()) {
        std::cerr << "search knn failed: " << knn_search.error().message << std::endl;
        abort();
    } else {
        auto result = *knn_search;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << " " << result->GetDistances()[i] << std::endl;
        }
    }

    return 0;
}
