
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

void
print_result(const std::string& title, const vsag::DatasetPtr& result, const std::string* paths) {
    std::cout << title << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        auto id = result->GetIds()[i];
        std::cout << "id=" << id << ", distance=" << result->GetDistances()[i]
                  << ", path=" << paths[id - 101] << std::endl;
    }
}

bool
has_id(const vsag::DatasetPtr& result, int64_t target_id) {
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (result->GetIds()[i] == target_id) {
            return true;
        }
    }
    return false;
}

int
main(int argc, char** argv) {
    constexpr int64_t num_vectors = 4;
    constexpr int64_t dim = 4;

    auto* ids = new int64_t[num_vectors]{101, 102, 103, 104};
    auto* vectors = new float[num_vectors * dim]{
        1.0F,
        2.0F,
        3.0F,
        4.0F,  // id 101
        1.0F,
        2.0F,
        3.0F,
        4.0F,  // id 102: same vector and same path as id 101
        1.0F,
        2.0F,
        3.0F,
        4.0F,  // id 103: same vector in another child path
        4.0F,
        3.0F,
        2.0F,
        1.0F,  // id 104: different vector in another tenant
    };
    auto* paths = new std::string[num_vectors]{"tenant_a/category_x/item_1",
                                               "tenant_a/category_x/item_1",
                                               "tenant_a/category_x/item_2",
                                               "tenant_b/category_y/item_1"};

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->Paths(paths)
        ->Owner(true);

    const auto* pyramid_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 4,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "alpha": 1.2,
            "graph_type": "nsw",
            "ef_construction": 64,
            "no_build_levels": [0, 1, 2],
            "index_min_size": 28,
            "use_reorder": false,
            "support_duplicate": true
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("pyramid", pyramid_build_parameters).value();

    if (auto build_result = index->Build(base); not build_result.has_value()) {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        return -1;
    }

    auto* query_vector = new float[dim]{1.0F, 2.0F, 3.0F, 4.0F};
    auto* query_path = new std::string[1]{"tenant_a/category_x/item_1"};
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Paths(query_path)->Owner(true);

    const auto* search_parameters = R"(
    {
        "pyramid": {
            "ef_search": 10
        }
    }
    )";
    auto knn_result = index->KnnSearch(query, 4, search_parameters);
    if (not knn_result.has_value()) {
        std::cerr << "Search Error: " << knn_result.error().message << std::endl;
        return -1;
    }

    print_result("Search exact duplicate path:", knn_result.value(), paths);
    std::cout << "Same-path duplicates are visible: id 101=" << has_id(knn_result.value(), 101)
              << ", id 102=" << has_id(knn_result.value(), 102)
              << ", other-path id 103=" << has_id(knn_result.value(), 103) << std::endl;

    query_path[0] = "tenant_a/category_x";
    knn_result = index->KnnSearch(query, 4, search_parameters);
    if (not knn_result.has_value()) {
        std::cerr << "Search Error: " << knn_result.error().message << std::endl;
        return -1;
    }

    print_result("\nSearch shared prefix path:", knn_result.value(), paths);
    std::cout << "The prefix path can see same-vector data from both child paths: id 103="
              << has_id(knn_result.value(), 103) << std::endl;

    return 0;
}
