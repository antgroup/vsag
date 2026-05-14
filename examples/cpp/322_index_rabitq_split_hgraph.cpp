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

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    constexpr int64_t num_vectors = 1000;
    constexpr int64_t dim = 128;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> vectors(num_vectors * dim);

    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < num_vectors * dim; ++i) {
        vectors[i] = distrib_real(rng);
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    /******************* Create Split RabitQ HGraph Index *****************/
    // Parameter suggestions for the current split RabitQ path:
    // 1. Keep base_quantization_type="rabitq" with base_codes_type="rabitq_split".
    // 2. Use rabitq_bits_per_dim_query=32; split RabitQ requires 32-bit query codes.
    // 3. Use rabitq_bits_per_dim_base=8 for the common 1+xbit layout (1+7bit today).
    // 4. Omit build_by_base to keep the default SQ8 fallback build path. Set
    //    build_by_base=true only if you explicitly want base-code graph construction.
    // 5. Use rabitq_one_bit_search=true at search time to enable the one-bit scan path.
    const std::string hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "rabitq",
            "base_codes_type": "rabitq_split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_error_rate": 1.9,
            "max_degree": 32,
            "ef_construction": 100,
            "graph_storage_type": "compressed"
        }
    }
    )";

    auto index = vsag::Factory::CreateIndex("hgraph", hgraph_build_parameters).value();

    /******************* Build Index *****************/
    if (auto build_result = index->Build(base); not build_result.has_value()) {
        std::cerr << "build index failed: " << build_result.error().message << std::endl;
        return -1;
    }
    std::cout << "index contains vectors: " << index->GetNumElements() << std::endl;

    /******************* Serialize Index *****************/
    const std::string index_path = "/tmp/vsag-rabitq-split-hgraph.index";
    std::ofstream out_stream(index_path, std::ios::binary);
    if (auto serialize_result = index->Serialize(out_stream); not serialize_result.has_value()) {
        std::cerr << "serialize index failed: " << serialize_result.error().message << std::endl;
        return -1;
    }
    out_stream.close();

    /******************* Deserialize Index *****************/
    auto loaded_index = vsag::Factory::CreateIndex("hgraph", hgraph_build_parameters).value();
    std::ifstream in_stream(index_path, std::ios::binary);
    if (auto deserialize_result = loaded_index->Deserialize(in_stream);
        not deserialize_result.has_value()) {
        std::cerr << "deserialize index failed: " << deserialize_result.error().message
                  << std::endl;
        return -1;
    }
    in_stream.close();

    /******************* Prepare Query Dataset *****************/
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    /******************* Search With One-Bit Split RabitQ *****************/
    const std::string hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100,
            "rabitq_one_bit_search": true
        }
    }
    )";
    constexpr int64_t topk = 10;
    if (auto search_result = loaded_index->KnnSearch(query, topk, hgraph_search_parameters);
        not search_result.has_value()) {
        std::cerr << "search index failed: " << search_result.error().message << std::endl;
        return -1;
    } else {
        auto result = *search_result;
        std::cout << "search results:" << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    }

    return 0;
}