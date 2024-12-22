
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

class SimpleAllocator : public vsag::Allocator {
public:
    SimpleAllocator() = default;
    ~SimpleAllocator() override = default;

public:
    std::string
    Name() override {
        return "DefaultAllocator";
    }

    void*
    Allocate(size_t size) override {
        auto ptr = malloc(size);
        allocate_count_++;
        return ptr;
    }

    void
    Deallocate(void* p) override {
        free(p);
    }

    void*
    Reallocate(void* p, size_t size) override {
        auto ptr = realloc(p, size);
        return ptr;
    }

    void
    Print() {
        std::cout << "allocate times : " << allocate_count_ << std::endl;
    }
    uint64_t allocate_count_{0};
};

int
main(int argc, char** argv) {
    vsag::init();

    int64_t num_vectors = 1000;
    int64_t dim = 128;

    // prepare ids and vectors
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

    // create index
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

    auto* allocator = new SimpleAllocator();

    {
        vsag::Resource resource(allocator, nullptr);
        vsag::Engine engine(&resource);
        auto index = engine.CreateIndex("hnsw", hnsw_build_paramesters).value();
        auto base = vsag::Dataset::Make();
        base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors)->Owner(false);
        index->Build(base);

        // prepare a query vector
        // memory will be released by query the dataset since owner is set to true when creating the query.
        auto query_vector = new float[dim];
        for (int64_t i = 0; i < dim; ++i) {
            query_vector[i] = distrib_real(rng);
        }

        // search on the index
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
        engine.Shutdown();
        allocator->Print();
    }

    // free memory
    delete[] ids;
    delete[] vectors;
    delete allocator;

    return 0;
}
