
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

#include "vsag/logger.h"
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

    // uncomment this line to show the memory allocation
    // vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kDEBUG);

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
    auto ids = (int64_t*)allocator.Allocate(sizeof(int64_t) * num_vectors);
    auto vectors = (float*)allocator.Allocate(sizeof(float) * dim * num_vectors);

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
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->Owner(true, &allocator);
    index->Build(base);

    // search on the index
    auto query_vector = new float[dim];  // memory will be released by query the dataset
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
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
    auto result = index->KnnSearch(query, topk, hnsw_search_parameters).value();

    // print the results
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    std::cout << "delete index" << std::endl;
    index = nullptr;

    return 0;
}
