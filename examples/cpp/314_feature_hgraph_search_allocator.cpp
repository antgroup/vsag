
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
    vsag::init();
    std::cout << "hgraph index example" << std::endl;

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
            "ef_construction": 100
        }
    }
    )";
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    std::cout << "create index" << std::endl;
    auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();

    ExampleAllocator allocator;

    /******************* Build HGraph Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HGraph contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    /******************* Prepare Query Dataset *****************/
    std::cout << "prepare index" << std::endl;
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    /******************* KnnSearch For HGraph Index *****************/
    auto hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;

    /******************* Hgraph sq8 Search *****************/
    {
        nlohmann::json search_parameters = {
            {"hgraph", {{"ef_search", 100}, {"skip_ratio", 0.7f}}},
        };
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

    /******************* Hgraph sq8 Iterator Filter *****************/
    {
        vsag::IteratorContext* iter_ctx = nullptr;
        nlohmann::json search_parameters = {
            {"hgraph", {{"ef_search", 100}, {"skip_ratio", 0.7f}}},
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

    engine.Shutdown();
    return 0;
}
