
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
hnsw_iter_filter() {
    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
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
    auto base = vsag::Dataset::Make();
    // Transfer the ownership of the data (ids, vectors) to the base.
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

    /******************* Create HNSW Index *****************/
    // hnsw_build_parameters is the configuration for building an HNSW index.
    // The "dtype" specifies the data type, which supports float32 and int8.
    // The "metric_type" indicates the distance metric type (e.g., cosine, inner product, and L2).
    // The "dim" represents the dimensionality of the vectors, indicating the number of features for each data point.
    // The "hnsw" section contains parameters specific to HNSW:
    // - "max_degree": The maximum number of connections for each node in the graph.
    // - "ef_construction": The size used for nearest neighbor search during graph construction, which affects both speed and the quality of the graph.
    auto hnsw_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();

    /******************* Build HNSW Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HNSW contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    /******************* KnnSearch For HNSW Index *****************/
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    // hnsw_search_parameters is the configuration for searching in an HNSW index.
    // The "hnsw" section contains parameters specific to the search operation:
    // - "ef_search": The size of the dynamic list used for nearest neighbor search, which influences both recall and search speed.
    auto hnsw_search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);

    /******************* Prepare Filter Object *****************/
    class MyFilter : public vsag::Filter {
    public:
        bool
        CheckValid(int64_t id) const override {
            return id % 2;
        }

        float
        ValidRatio() const override {
            return 0.618f;
        }
    };
    auto filter_object = std::make_shared<MyFilter>();
    std::unordered_map<int64_t, bool> myMap;

    /******************* Print Search Result All topK * 2 *****************/
    auto time7 = std::chrono::steady_clock::now();
    auto knn_result0 = index->KnnSearch(query, topk * 2, hnsw_search_parameters, filter_object);
    auto time8 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration0 = time8 - time7;
    std::cout << "knn_result0: " << duration0.count() << std::endl;
    if (knn_result0.has_value()) {
        auto result = knn_result0.value();
        std::cout << "results0: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result0.error().message << std::endl;
    }

    vsag::IteratorContextPtr filter_ctx = nullptr;
    /******************* Search And Print Result1 *****************/
    auto time0 = std::chrono::steady_clock::now();
    auto knn_result =
        index->KnnSearch(query, topk, hnsw_search_parameters, filter_object, &filter_ctx, false);
    auto time1 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = time1 - time0;
    std::cout << "knn_result1: " << duration.count() << std::endl;

    if (knn_result.has_value()) {
        auto result = knn_result.value();
        std::cout << "results1: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            myMap[result->GetIds()[i]] = true;
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result.error().message << std::endl;
    }

    /******************* Search And Print Result2 *****************/
    auto time3 = std::chrono::steady_clock::now();
    auto knn_result2 =
        index->KnnSearch(query, topk, hnsw_search_parameters, filter_object, &filter_ctx, false);

    auto time4 = std::chrono::steady_clock::now();
    duration = time4 - time3;
    std::cout << "knn_result2: " << duration.count() << std::endl;

    if (knn_result2.has_value()) {
        auto result = knn_result2.value();
        std::cout << "results2: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            auto it = myMap.find(result->GetIds()[i]);
            if (it == myMap.end()) {
                myMap[result->GetIds()[i]] = true;
                std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
            } else {
                std::cerr << "Search Duplicate: " << result->GetIds()[i] << std::endl;
            }
        }
    } else {
        std::cerr << "Search Error: " << knn_result2.error().message << std::endl;
    }

    /******************* Search And Print Result3 *****************/
    auto time5 = std::chrono::steady_clock::now();
    auto knn_result3 =
        index->KnnSearch(query, topk, hnsw_search_parameters, filter_object, &filter_ctx, true);

    auto time6 = std::chrono::steady_clock::now();
    duration = time6 - time5;
    std::cout << "knn_result3: " << duration.count() << std::endl;

    if (knn_result3.has_value()) {
        auto result = knn_result3.value();
        std::cout << "results3: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            auto it = myMap.find(result->GetIds()[i]);
            if (it == myMap.end()) {
                myMap[result->GetIds()[i]] = true;
                std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
            } else {
                std::cerr << "Search Duplicate: " << result->GetIds()[i] << std::endl;
            }
        }
    } else {
        std::cerr << "Search Error: " << knn_result3.error().message << std::endl;
    }
}

void
hgraph_iter_filter() {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
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
            "ef_construction": 100
        }
    }
    )";
    vsag::Engine engine;
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

    vsag::IteratorContextPtr iter_ctx = nullptr;

    /******************* Prepare Filter Object *****************/
    class MyFilter : public vsag::Filter {
    public:
        bool
        CheckValid(int64_t id) const override {
            return id % 2;
        }

        float
        ValidRatio() const override {
            return 0.618f;
        }
    };
    auto filter_object = std::make_shared<MyFilter>();
    std::unordered_map<int64_t, bool> myMap;

    /******************* KnnSearch For HGraph Index *****************/
    auto hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;

    /******************* Search And Print Result1 *****************/
    auto result1 =
        index->KnnSearch(query, topk, hgraph_search_parameters, filter_object, &iter_ctx, false);

    if (result1.has_value()) {
        auto result = result1.value();
        std::cout << "results1: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            myMap[result->GetIds()[i]] = true;
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << result1.error().message << std::endl;
    }

    /******************* Search And Print Result2 *****************/
    auto result2 =
        index->KnnSearch(query, topk, hgraph_search_parameters, filter_object, &iter_ctx, false);

    if (result2.has_value()) {
        auto result = result2.value();
        std::cout << "results2: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            auto it = myMap.find(result->GetIds()[i]);
            if (it == myMap.end()) {
                myMap[result->GetIds()[i]] = true;
                std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
            } else {
                std::cerr << "Search Duplicate: " << result->GetIds()[i] << std::endl;
            }
        }
    } else {
        std::cerr << "Search Error: " << result2.error().message << std::endl;
    }

    /******************* Search And Print Result3 *****************/
    auto result3 =
        index->KnnSearch(query, topk, hgraph_search_parameters, filter_object, &iter_ctx, true);

    if (result3.has_value()) {
        auto result = result3.value();
        std::cout << "results3: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            auto it = myMap.find(result->GetIds()[i]);
            if (it == myMap.end()) {
                myMap[result->GetIds()[i]] = true;
                std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
            } else {
                std::cerr << "Search Duplicate: " << result->GetIds()[i] << std::endl;
            }
        }
    } else {
        std::cerr << "Search Error: " << result3.error().message << std::endl;
    }

    /******************* Print Search Result All *****************/
    auto new_filter = [filter_object](int64_t id) -> bool { return filter_object->CheckValid(id); };
    auto result0 = index->KnnSearch(query, topk * 3, hgraph_search_parameters, new_filter);

    if (result0.has_value()) {
        auto result = result0.value();
        std::cout << "results: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << result0.error().message << std::endl;
    }

    engine.Shutdown();
}

int
main(int argc, char** argv) {
    hnsw_iter_filter();
    hgraph_iter_filter();
    return 0;
}