
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

bool
is_path_belong_to(const std::string& a, const std::string& b) {
    return b.compare(0, a.size(), a) == 0;
}

std::string
create_random_string(bool is_full) {
    const std::vector<std::string> level1 = {"a", "b", "c"};
    const std::vector<std::string> level2 = {"d", "e"};
    const std::vector<std::string> level3 = {"f", "g", "h"};

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> distr;

    std::vector<std::string> selected_levels;

    if (is_full) {
        selected_levels.emplace_back(level1[distr(mt) % level1.size()]);
        selected_levels.emplace_back(level2[distr(mt) % level2.size()]);
        selected_levels.emplace_back(level3[distr(mt) % level3.size()]);
    } else {
        std::uniform_int_distribution<> dist(1, 3);
        int num_levels = dist(mt);

        if (num_levels >= 1) {
            selected_levels.emplace_back(level1[distr(mt) % level1.size()]);
        }
        if (num_levels >= 2) {
            selected_levels.emplace_back(level2[distr(mt) % level2.size()]);
        }
        if (num_levels == 3) {
            selected_levels.emplace_back(level3[distr(mt) % level3.size()]);
        }
    }

    std::string random_string = selected_levels.empty() ? "" : selected_levels[0];
    for (size_t i = 1; i < selected_levels.size(); ++i) {
        random_string += "/" + selected_levels[i];
    }

    return random_string;
}

int
main(int argc, char** argv) {
    vsag::init();

    int64_t num_vectors = 10000;
    int64_t dim = 128;

    // prepare ids and vectors
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];
    auto paths = new std::string[num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    for (int64_t i = 0; i < num_vectors; ++i) {
        paths[i] = create_random_string(true);
    }

    // create index
    auto pyramid_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "sub_index_type": "hnsw",
            "index_param": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("pyramid", pyramid_build_paramesters).value();
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->Paths(paths)
        ->Owner(false);
    index->Build(base);

    // prepare a query vector
    // memory will be released by query the dataset since owner is set to true when creating the query.
    auto query_vector = new float[dim];
    auto query_path = new std::string[1];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    query_path[0] = create_random_string(false);
    std::cout << "query_path:" << query_path[0] << std::endl;
    // search on the index
    auto pyramid_search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Paths(query_path)->Owner(true);
    auto result = index->KnnSearch(query, topk, pyramid_search_parameters).value();

    // print the results
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i]
                  << "  path:" << paths[result->GetIds()[i]] << std::endl;
    }

    // free memory
    delete[] ids;
    delete[] vectors;
    delete[] paths;

    return 0;
}
