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

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    int64_t dim = 128;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> datas(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->Owner(false);

    /******************* Create HGraph Index *****************/
    std::string hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100,
            "alpha":1.2
        }
    }
    )";
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    vsag::Options::Instance().set_block_size_limit(2 * 1024 * 1024);
    auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();

    /******************* Build HGraph Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HGraph contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    /******************* Prepare Query Dataset *****************/
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    /******************* Basic KnnSearch For Reference *****************/
    auto hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto knn_result = index->KnnSearch(query, topk, hgraph_search_parameters).value();

    std::cout << "=== KnnSearch (reference) ===" << std::endl;
    for (int64_t i = 0; i < knn_result->GetDim(); ++i) {
        std::cout << knn_result->GetIds()[i] << ": " << knn_result->GetDistances()[i] << std::endl;
    }

    /******************* OpenSearchSession *****************/
    // Open a session with the same query and search parameters.
    // The session retains traversal state across Next calls.
    int64_t batch_size = 3;
    auto session = index->OpenSearchSession(query, batch_size, hgraph_search_parameters).value();

    std::cout << std::endl << "=== SearchSession (paginated) ===" << std::endl;
    int call = 0;
    while (session->HasMore()) {
        auto result = session->Next(batch_size);
        if (result.has_value()) {
            std::cout << "--- Call " << ++call << " (got " << result.value()->GetDim()
                      << " results) ---" << std::endl;
            for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
                std::cout << result.value()->GetIds()[i] << ": "
                          << result.value()->GetDistances()[i] << std::endl;
            }
            // Statistics are cumulative across calls
            std::cout << "  stats: " << result.value()->Statistics("") << std::endl;
        }
    }

    /******************* SearchSession With Dynamic Parameters *****************/
    std::cout << std::endl << "=== SearchSession (dynamic ef_search) ===" << std::endl;

    // Start with a small ef_search
    std::string initial_params = R"({"hgraph":{"ef_search":32}})";
    auto session2 = index->OpenSearchSession(query, 10, initial_params).value();

    // First batch with default ef_search=32
    auto batch1 = session2->Next(5);
    if (batch1.has_value()) {
        std::cout << "Batch 1 (ef=32): " << batch1.value()->GetDim() << " results" << std::endl;
        std::cout << "  stats: " << batch1.value()->Statistics("") << std::endl;
    }

    // Second batch: boost ef_search to 128 for higher recall
    vsag::SearchSessionNextOptions opts;
    opts.max_candidates = 5;
    opts.search_parameters = R"({"hgraph":{"ef_search":128}})";
    auto batch2 = session2->Next(opts);
    if (batch2.has_value()) {
        std::cout << "Batch 2 (ef=128): " << batch2.value()->GetDim() << " results" << std::endl;
        std::cout << "  stats: " << batch2.value()->Statistics("") << std::endl;
    }

    /******************* Close Sessions *****************/
    session->Close();
    session2->Close();

    engine.Shutdown();
    return 0;
}
