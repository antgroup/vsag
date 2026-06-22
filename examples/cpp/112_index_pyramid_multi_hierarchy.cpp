
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
#include <random>
#include <string>
#include <vector>

// Generate a random "site" path, e.g. "site-a/lang-en" or "site-b/lang-zh".
std::string
generate_site_path(std::mt19937& rng) {
    const std::vector<std::string> sites = {"site-a", "site-b", "site-c"};
    const std::vector<std::string> langs = {"lang-en", "lang-zh", "lang-ja"};
    return sites[rng() % sites.size()] + "/" + langs[rng() % langs.size()];
}

// Generate a random "category" path, e.g. "news/politics" or "sports/football".
std::string
generate_category_path(std::mt19937& rng) {
    const std::vector<std::string> top = {"news", "sports", "tech"};
    const std::vector<std::string> sub_news = {"politics", "economy", "world"};
    const std::vector<std::string> sub_sports = {"football", "basketball", "tennis"};
    const std::vector<std::string> sub_tech = {"ai", "web", "cloud"};

    auto top_cat = top[rng() % top.size()];
    if (top_cat == "news") {
        return top_cat + "/" + sub_news[rng() % sub_news.size()];
    } else if (top_cat == "sports") {
        return top_cat + "/" + sub_sports[rng() % sub_sports.size()];
    } else {
        return top_cat + "/" + sub_tech[rng() % sub_tech.size()];
    }
}

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    int64_t dim = 128;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    // Generate two independent sets of paths — one per hierarchy.
    auto site_paths = new std::string[num_vectors];
    auto category_paths = new std::string[num_vectors];
    for (int64_t i = 0; i < num_vectors; ++i) {
        site_paths[i] = generate_site_path(rng);
        category_paths[i] = generate_category_path(rng);
    }

    auto base = vsag::Dataset::Make();
    // Vectors and IDs are shared; each hierarchy gets its own path array.
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->Paths("site", site_paths)
        ->Paths("category", category_paths)
        ->Owner(false);

    /******************* Create Pyramid Index *****************/
    // Build parameters with two named hierarchies:
    //   - "site": inherits all top-level graph params (string shorthand).
    //   - "category": overrides max_degree and no_build_levels (object form).
    auto build_params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 32,
            "alpha": 1.2,
            "graph_type": "odescent",
            "graph_iter_turn": 15,
            "neighbor_sample_rate": 0.2,
            "no_build_levels": [0, 1],
            "use_reorder": true,
            "build_thread_count": 16,
            "hierarchies": [
                "site",
                {
                    "name": "category",
                    "max_degree": 64,
                    "no_build_levels": [0]
                }
            ]
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("pyramid", build_params).value();

    /******************* Build Index *****************/
    if (auto result = index->Build(base); result.has_value()) {
        std::cout << "After Build(), Pyramid contains: " << index->GetNumElements()
                  << " vectors with 2 hierarchies (site, category)" << std::endl;
    } else {
        std::cerr << "Build failed: " << result.error().message << std::endl;
        return 1;
    }

    /******************* Search on "site" Hierarchy *****************/
    {
        std::cout << "\n=== Search on 'site' hierarchy ===" << std::endl;
        auto query_path = new std::string[1];
        query_path[0] = "site-a";
        auto query_vector = new float[dim];
        for (int64_t i = 0; i < dim; ++i) {
            query_vector[i] = distrib_real(rng);
        }

        // Search parameters select the "site" hierarchy.
        auto search_params = R"(
        {
            "pyramid": {
                "ef_search": 100,
                "hierarchies": ["site"]
            }
        }
        )";
        int64_t topk = 10;
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(query_vector)
            ->Paths("site", query_path)
            ->Owner(true);
        auto knn_result = index->KnnSearch(query, topk, search_params);

        std::cout << "Query site path: " << query_path[0] << std::endl;
        if (knn_result.has_value()) {
            auto result = knn_result.value();
            std::cout << "Results:" << std::endl;
            for (int64_t i = 0; i < result->GetDim(); ++i) {
                auto id = result->GetIds()[i];
                std::cout << "  id=" << id << "  dist=" << result->GetDistances()[i]
                          << "  site=" << site_paths[id] << "  category=" << category_paths[id]
                          << std::endl;
            }
        } else {
            std::cerr << "Search error: " << knn_result.error().message << std::endl;
        }
    }

    /******************* Search on "category" Hierarchy *****************/
    {
        std::cout << "\n=== Search on 'category' hierarchy ===" << std::endl;
        auto query_path = new std::string[1];
        query_path[0] = "news";
        auto query_vector = new float[dim];
        for (int64_t i = 0; i < dim; ++i) {
            query_vector[i] = distrib_real(rng);
        }

        auto search_params = R"(
        {
            "pyramid": {
                "ef_search": 100,
                "hierarchies": ["category"]
            }
        }
        )";
        int64_t topk = 10;
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(query_vector)
            ->Paths("category", query_path)
            ->Owner(true);
        auto knn_result = index->KnnSearch(query, topk, search_params);

        std::cout << "Query category path: " << query_path[0] << std::endl;
        if (knn_result.has_value()) {
            auto result = knn_result.value();
            std::cout << "Results:" << std::endl;
            for (int64_t i = 0; i < result->GetDim(); ++i) {
                auto id = result->GetIds()[i];
                std::cout << "  id=" << id << "  dist=" << result->GetDistances()[i]
                          << "  site=" << site_paths[id] << "  category=" << category_paths[id]
                          << std::endl;
            }
        } else {
            std::cerr << "Search error: " << knn_result.error().message << std::endl;
        }
    }

    /******************* Add (Incremental Insert) *****************/
    {
        std::cout << "\n=== Add 100 new vectors ===" << std::endl;
        int64_t add_count = 100;
        auto add_ids = new int64_t[add_count];
        auto add_vectors = new float[dim * add_count];
        auto add_site_paths = new std::string[add_count];
        auto add_cat_paths = new std::string[add_count];
        for (int64_t i = 0; i < add_count; ++i) {
            add_ids[i] = num_vectors + i;
            add_site_paths[i] = generate_site_path(rng);
            add_cat_paths[i] = generate_category_path(rng);
        }
        for (int64_t i = 0; i < dim * add_count; ++i) {
            add_vectors[i] = distrib_real(rng);
        }
        auto add_data = vsag::Dataset::Make();
        add_data->NumElements(add_count)
            ->Dim(dim)
            ->Ids(add_ids)
            ->Float32Vectors(add_vectors)
            ->Paths("site", add_site_paths)
            ->Paths("category", add_cat_paths)
            ->Owner(false);
        if (auto result = index->Add(add_data); result.has_value()) {
            std::cout << "After Add(), Pyramid contains: " << index->GetNumElements() << std::endl;
        } else {
            std::cerr << "Add failed: " << result.error().message << std::endl;
        }
        delete[] add_ids;
        delete[] add_vectors;
        delete[] add_site_paths;
        delete[] add_cat_paths;
    }

    /******************* RangeSearch on "category" Hierarchy *****************/
    {
        std::cout << "\n=== RangeSearch on 'category' hierarchy ===" << std::endl;
        auto query_path = new std::string[1];
        query_path[0] = "sports";
        auto query_vector = new float[dim];
        for (int64_t i = 0; i < dim; ++i) {
            query_vector[i] = distrib_real(rng);
        }
        auto search_params = R"(
        {
            "pyramid": {
                "ef_search": 100,
                "hierarchies": ["category"]
            }
        }
        )";
        float radius = 20.0f;
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(query_vector)
            ->Paths("category", query_path)
            ->Owner(true);
        auto range_result = index->RangeSearch(query, radius, search_params);
        std::cout << "Query category path: " << query_path[0] << "  radius: " << radius
                  << std::endl;
        if (range_result.has_value()) {
            auto result = range_result.value();
            std::cout << "Found " << result->GetDim() << " results within radius" << std::endl;
            for (int64_t i = 0; i < std::min<int64_t>(result->GetDim(), 5); ++i) {
                std::cout << "  id=" << result->GetIds()[i]
                          << "  dist=" << result->GetDistances()[i] << std::endl;
            }
        } else {
            std::cerr << "RangeSearch error: " << range_result.error().message << std::endl;
        }
    }

    /******************* Serialize & Deserialize *****************/
    {
        std::cout << "\n=== Serialize & Deserialize ===" << std::endl;
        // Serialize
        auto bs = index->Serialize();
        if (not bs.has_value()) {
            std::cerr << "Serialize failed: " << bs.error().message << std::endl;
            return 1;
        }
        auto keys = bs->GetKeys();
        vsag::BinarySet binary_set;
        for (const auto& key : keys) {
            binary_set.Set(key, bs->Get(key));
        }
        std::cout << "Serialized " << keys.size() << " blobs" << std::endl;

        // Deserialize into a new index
        auto new_index = vsag::Factory::CreateIndex("pyramid", build_params).value();
        new_index->Deserialize(binary_set);
        std::cout << "Deserialized index contains: " << new_index->GetNumElements() << " vectors"
                  << std::endl;

        // Verify search on deserialized index
        auto query_path = new std::string[1];
        query_path[0] = "site-b";
        auto query_vector = new float[dim];
        for (int64_t i = 0; i < dim; ++i) {
            query_vector[i] = distrib_real(rng);
        }
        auto search_params = R"(
        {
            "pyramid": {
                "ef_search": 100,
                "hierarchies": ["site"]
            }
        }
        )";
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(query_vector)
            ->Paths("site", query_path)
            ->Owner(true);
        auto knn_result = new_index->KnnSearch(query, 5, search_params);
        if (knn_result.has_value()) {
            auto result = knn_result.value();
            std::cout << "Search on deserialized index (site-b), top-5:" << std::endl;
            for (int64_t i = 0; i < result->GetDim(); ++i) {
                std::cout << "  id=" << result->GetIds()[i]
                          << "  dist=" << result->GetDistances()[i] << std::endl;
            }
        } else {
            std::cerr << "Search error: " << knn_result.error().message << std::endl;
        }
    }

    delete[] ids;
    delete[] vectors;
    delete[] site_paths;
    delete[] category_paths;
    return 0;
}
