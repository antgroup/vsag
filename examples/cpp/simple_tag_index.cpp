
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
    vsag::init();

    int64_t num_vectors = 1000;
    int64_t dim = 128;
    int64_t tag_num = 10;

    std::string tag_name = "tag";

    // prepare ids and vectors
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    auto tags = new int64_t[num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    std::uniform_int_distribution<> distrib_int(0, tag_num);
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    for (int64_t i = 0; i < num_vectors; ++i) {
        tags[i] = distrib_int(rng);
    }

    // create index
    auto tag_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
             "sub_index_type": "hnsw"
        },
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("tag_index", tag_build_paramesters).value();
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors)->Tags(tag_name, tags)->Owner(false);
    index->Build(base);

    // prepare a query vector
    // memory will be released by query the dataset since owner is set to true when creating the query.
    auto query_vector = new float[dim];
    auto query_tag = new int64_t[1];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    query_tag[0] = 1;

    // search on the index
    auto tag_search_parameters = R"(
    {
        "tag_index": {
            "sub_index_type": "hnsw",
            "index_param": {
                "ef_search": 100
            }
        }
    }
    )";
    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Tags(tag_name, query_tag)->Owner(true);
    auto result = index->KnnSearch(query, topk, tag_search_parameters).value();

    // print the results
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << " tag:" << tags[result->GetIds()[i]] << std::endl;
    }

    // free memory
    delete[] ids;
    delete[] vectors;
    delete[] tags;

    return 0;
}
