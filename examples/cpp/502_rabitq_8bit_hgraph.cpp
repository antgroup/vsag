
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

// Example: 8-bit RaBitQ (1+7bit split) on top of an HGraph index.
//
// This uses the v4 RabitQ "split_1bit_7bit" layout:
//   - rabitq_bits_per_dim_base  = 8   (1 sign bit + 7 supplement bits per dim)
//   - rabitq_bits_per_dim_query = 32  (float query, required by v4 split)
//   - base_codes_type           = "rabitq_split"
//   - rabitq_version            = "split_1bit_7bit"
//
// See docs/rabitq_split_1bit_7bit.md for the full parameter reference.

#include <vsag/vsag.h>

#include <iostream>
#include <random>
#include <vector>

int
main(int /*argc*/, char** /*argv*/) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    const int64_t num_vectors = 10000;
    const int64_t dim = 128;

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

    /******************* Create HGraph Index w/ 8-bit RaBitQ ***************/
    // Notes:
    //   - rabitq_bits_per_dim_base = 8 selects the 1+7bit configuration.
    //   - rabitq_version must be "split_1bit_7bit" to enable the v4 path.
    //   - rabitq_bits_per_dim_query must be 32 in v4 split.
    //   - use_reorder + precise_quantization_type = "fp32" gives full-precision
    //     reorder on top of the compressed RaBitQ base codes.
    std::string hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "rabitq",
            "base_codes_type": "rabitq_split",
            "rabitq_version": "split_1bit_7bit",
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_query": 32,
            "rabitq_error_rate": 1.9,
            "max_degree": 32,
            "ef_construction": 300,
            "alpha": 1.2,
            "use_reorder": true,
            "precise_quantization_type": "fp32"
        }
    }
    )";

    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();

    /******************* Build HGraph Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), HGraph(RaBitQ 1+7bit) contains: " << index->GetNumElements()
                  << " vectors" << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        return -1;
    }

    /******************* Prepare Query Dataset *****************/
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    /******************* KnnSearch *****************/
    // rabitq_one_bit_search:
    //   true  -> graph traversal uses only the 1-bit sign code (fastest).
    //   false -> graph traversal uses the full 1+7bit code (more accurate).
    // Reorder with the fp32 precise codes is applied on top either way.
    auto hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100,
            "rabitq_one_bit_search": true
        }
    }
    )";

    const int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, hgraph_search_parameters).value();

    /******************* Print Search Result *****************/
    std::cout << "top-" << topk << " results:" << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << "  " << result->GetIds()[i] << " : " << result->GetDistances()[i] << std::endl;
    }

    engine.Shutdown();
    return 0;
}
