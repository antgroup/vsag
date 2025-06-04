
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

#include <iostream>

#include "nlohmann/json.hpp"
#include "vsag/logger.h"
#include "vsag/search_param.h"
#include "vsag/vsag.h"

class ExampleAllocator : public vsag::Allocator {
public:
    std::string
    Name() override {
        return "example-allocator";
    }

    void*
    Allocate(size_t size) override {
        vsag::Options::Instance().logger()->Debug("allocate " + std::to_string(size) + " bytes.");
        auto addr = (void*)malloc(size);
        sizes_[addr] = size;
        return addr;
    }

    void
    Deallocate(void* p) override {
        if (sizes_.find(p) == sizes_.end())
            return;
        vsag::Options::Instance().logger()->Debug("deallocate " + std::to_string(sizes_[p]) +
                                                  " bytes.");
        sizes_.erase(p);
        return free(p);
    }

    void*
    Reallocate(void* p, size_t size) override {
        vsag::Options::Instance().logger()->Debug("reallocate " + std::to_string(size) + " bytes.");
        auto addr = (void*)realloc(p, size);
        sizes_.erase(p);
        sizes_[addr] = size;
        return addr;
    }

private:
    std::unordered_map<void*, size_t> sizes_;
};

int
main() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kINFO);

    ExampleAllocator allocator;
    vsag::Resource resource(&allocator, nullptr);
    vsag::Engine engine(&resource);

    auto paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 4,
        "hnsw": {
            "max_degree": 5,
            "ef_construction": 20
        }
    }
    )";
    std::cout << "create index" << std::endl;
    auto index = engine.CreateIndex("hnsw", paramesters).value();

    std::cout << "prepare data" << std::endl;
    int64_t num_vectors = 100;
    int64_t dim = 4;

    // prepare ids and vectors
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> vectors(num_vectors * dim);

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
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    index->Build(base);

    // search on the index
    auto query_vector = new float[dim];  // memory will be released by query the dataset
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);

    /******************* HNSW Search *****************/
    {
        nlohmann::json search_parameters = {
            {"hnsw", {{"ef_search", 100}, {"skip_ratio", 0.7f}}},
        };
        int64_t topk = 10;
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);

        std::string param_str = search_parameters.dump();
        vsag::SearchParam search_param(false, param_str, nullptr, &allocator);
        auto result = index->KnnSearch(query, topk, search_param).value();

        // print the results
        std::cout << "results: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }

        allocator.Deallocate((void*)result->GetIds());
        allocator.Deallocate((void*)result->GetDistances());
    }

    /******************* HNSW Iterator Filter *****************/
    {
        vsag::IteratorContext* iter_ctx = nullptr;
        nlohmann::json search_parameters = {
            {"hnsw", {{"ef_search", 100}, {"skip_ratio", 0.7f}}},
        };
        std::string param_str = search_parameters.dump();
        vsag::SearchParam search_param(true, param_str, nullptr, &allocator, iter_ctx, false);

        /* first search */
        {
            auto result = index->KnnSearch(query, topk, search_param).value();

            // print the results
            std::cout << "results: " << std::endl;
            for (int64_t i = 0; i < result->GetDim(); ++i) {
                std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
            }

            allocator.Deallocate((void*)result->GetIds());
            allocator.Deallocate((void*)result->GetDistances());
        }

        /* last search */
        {
            search_param.is_last_search = true;
            auto result = index->KnnSearch(query, topk, search_param).value();

            // print the results
            std::cout << "results: " << std::endl;
            for (int64_t i = 0; i < result->GetDim(); ++i) {
                std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
            }

            allocator.Deallocate((void*)result->GetIds());
            allocator.Deallocate((void*)result->GetDistances());
        }
    }

    std::cout << "delete index" << std::endl;
    index = nullptr;
    engine.Shutdown();

    return 0;
}
