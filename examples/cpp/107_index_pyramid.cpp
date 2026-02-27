
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
        selected_levels.push_back(level1[distr(mt) % level1.size()]);
        selected_levels.push_back(level2[distr(mt) % level2.size()]);
        selected_levels.push_back(level3[distr(mt) % level3.size()]);
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
    for (uint64_t i = 1; i < selected_levels.size(); ++i) {
        random_string += "/" + selected_levels[i];
    }

    return random_string;
}

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
    // Generate random paths for the vectors
    std::string* paths = new std::string[num_vectors];
    for (int i = 0; i < num_vectors; ++i) {
        paths[i] = create_random_string(true);
    }

    auto base = vsag::Dataset::Make();
    // Transfer the ownership of the data (ids, vectors) to the base.
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Paths(paths)->Float32Vectors(vectors);

    /******************* Create Pyramid Index *****************/
    // pyramid_build_parameters is the configuration for building a Pyramid index.
    // The "dtype" specifies the data type, which supports float32 and int8.
    // The "metric_type" indicates the distance metric type (e.g., cosine, inner product, and L2).
    // The "dim" represents the dimensionality of the vectors, indicating the number of features for each data point.
    // The "pyramid" section contains parameters specific to Pyramid:
    // - "odescent": graph type
    // - "max_degree": The maximum number of connections for each node in the graph.
    // - "alpha": The parameter for the graph construction, which influences the pruning process.
    // - "no_build_levels": The levels that do not need to be built.
    // - "base_quantization_type": The base quantization codes
    // - "precise_quantization_type": The precise quantization codes
    // - "index_min_size": The minimum size required to build the index
    // - "support_duplicate": support for duplicate data in the index
    auto pyramid_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 32,
            "alpha": 1.2,
            "graph_iter_turn": 15,
            "neighbor_sample_rate": 0.2,
            "no_build_levels": [0, 1],
            "use_reorder": true,
            "graph_type": "odescent",
            "build_thread_count": 16
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("pyramid", pyramid_build_paramesters).value();

    /******************* Build Pyramid Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index Pyramid contains: " << index->GetNumElements()
                  << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    /******************* KnnSearch For Pyramid Index *****************/
    auto query_path = new std::string[1];
    query_path[0] = create_random_string(false);
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    // pyramid_search_parameters is the configuration for searching in an Pyramid index.
    // The "pyramid" section contains parameters specific to the search operation:
    // - "ef_search": The size of the dynamic list used for nearest neighbor search, which influences both recall and search speed.
    auto pyramid_search_parameters = R"(
    {
        "pyramid": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Paths(query_path)->Owner(true);
    auto knn_result = index->KnnSearch(query, topk, pyramid_search_parameters);

    /******************* Print Search Result *****************/
    std::cout << "Query path: " << query_path[0] << std::endl;
    if (knn_result.has_value()) {
        auto result = knn_result.value();
        std::cout << "results: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i]
                      << " paths:" << paths[result->GetIds()[i]] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result.error().message << std::endl;
    }

    return 0;
}
