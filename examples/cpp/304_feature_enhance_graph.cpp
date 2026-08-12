
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
    int dim = 128;
    int base_elements = 2000;
    int query_elements = 1000;
    int ef_search = 10;
    int64_t k = 10;

    auto base = vsag::Dataset::Make();
    std::shared_ptr<int64_t[]> base_ids(new int64_t[base_elements]);
    std::shared_ptr<float[]> base_data(new float[dim * base_elements]);
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distribution_real(-1, 1);
    for (int i = 0; i < base_elements; i++) {
        base_ids[i] = i;

        for (int d = 0; d < dim; d++) {
            base_data[d + i * dim] = distribution_real(rng);
        }
    }
    base->Dim(dim)
        ->NumElements(base_elements)
        ->Ids(base_ids.get())
        ->Float32Vectors(base_data.get())
        ->Owner(false);

    /******************* Build HGraph Index *****************/
    // When you want to use EnhanceGraph, the use_conjugate_graph must be set to true
    auto hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 100,
            "use_conjugate_graph": true
        }
    }
    )";
    std::shared_ptr<vsag::Index> hgraph;
    if (auto index = vsag::Factory::CreateIndex("hgraph", hgraph_build_parameters);
        index.has_value()) {
        hgraph = index.value();
    } else {
        std::cerr << "Failed to create HGraph: " << index.error().message << std::endl;
        return -1;
    }

    if (const auto build_result = hgraph->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index contains: " << hgraph->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    /******************* Search HGraph Index without Conjugate Graph *****************/
    // record the failed ids
    std::set<std::pair<int, int64_t>> failed_queries;
    // use_conjugate_graph_search indicates whether to use information from the conjugate_graph to enhance the search results.
    auto before_enhance_parameters = R"(
    {
        "hgraph": {
            "ef_search": 10,
            "use_conjugate_graph_search": false
        }
    }
    )";
    {
        int correct = 0;
        std::cout << "====Search Stage====" << std::endl;

        for (int i = 0; i < query_elements; i++) {
            auto query = vsag::Dataset::Make();
            query->Dim(dim)
                ->Float32Vectors(base_data.get() + i * dim)
                ->NumElements(1)
                ->Owner(false);

            auto result = hgraph->KnnSearch(query, k, before_enhance_parameters);
            int64_t global_optimum = i;  // global optimum is itself
            if (result.has_value()) {
                int64_t local_optimum = result.value()->GetIds()[0];
                if (local_optimum == global_optimum) {
                    correct++;
                } else {
                    failed_queries.emplace(i, global_optimum);
                }
            } else {
                std::cerr << "Search Error: " << result.error().message << std::endl;
            }
        }
        std::cout << "Recall: " << correct / (1.0 * query_elements) << std::endl;
    }

    /******************* Enhance Phase *****************/
    //
    {
        uint32_t error_fixed = 0;
        std::cout << "====Feedback Stage====" << std::endl;
        for (auto item : failed_queries) {
            auto query = vsag::Dataset::Make();
            query->Dim(dim)
                ->Float32Vectors(base_data.get() + item.first * dim)
                ->NumElements(1)
                ->Owner(false);
            auto feedback = hgraph->Feedback(query, 1, before_enhance_parameters, item.second);
            if (not feedback.has_value()) {
                std::cerr << "Feedback Error: " << feedback.error().message << std::endl;
                return -1;
            }
            error_fixed += feedback.value();
        }
        std::cout << "Fixed queries num: " << error_fixed << std::endl;
    }

    /******************* Search HGraph Index with Conjugate Graph *****************/
    auto after_enhance_parameters = R"(
    {
        "hgraph": {
            "ef_search": 10,
            "use_conjugate_graph_search": true
        }
    }
    )";
    {
        int correct = 0;
        std::cout << "====Enhanced Search Stage====" << std::endl;

        for (int i = 0; i < query_elements; i++) {
            auto query = vsag::Dataset::Make();
            query->Dim(dim)
                ->Float32Vectors(base_data.get() + i * dim)
                ->NumElements(1)
                ->Owner(false);

            auto result = hgraph->KnnSearch(query, k, after_enhance_parameters);
            int64_t global_optimum = i;  // global optimum is itself
            if (result.has_value()) {
                int64_t local_optimum = result.value()->GetIds()[0];
                if (local_optimum == global_optimum) {
                    correct++;
                }
            } else {
                std::cerr << "Search Error: " << result.error().message << std::endl;
            }
        }
        std::cout << "Enhanced Recall: " << correct / (1.0 * query_elements) << std::endl;
    }

    return 0;
}
