
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

// TODO(LHT): don't use inner struct
#include "impl/thread_pool/default_thread_pool.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "vsag/logger.h"
#include "vsag/vsag.h"

int
main() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kINFO);

    /******************* Customize Thread Pool *****************/
    auto pool = vsag::Engine::CreateThreadPool(16).value();
    auto allocator = vsag::Engine::CreateDefaultAllocator();
    vsag::Resource resource(allocator.get(), pool.get());
    vsag::Engine engine(&resource);

    vsag::init();

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
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

    /******************* Create DiskANN Index *****************/
    auto diskann_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "diskann": {
            "max_degree": 16,
            "ef_construction": 200,
            "pq_sample_rate": 0.5,
            "pq_dims": 9,
            "use_pq_search": true,
            "use_async_io": false,
            "use_bsa": true
        }
    }
    )";
    auto index = engine.CreateIndex("diskann", diskann_build_paramesters).value();

    /******************* Build DiskANN Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index DiskANN contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    engine.Shutdown();

    return 0;
}
