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
#include <random>
#include <string>
#include <vector>

namespace {

vsag::DatasetPtr
MakeDataset(int64_t first_id,
            int64_t count,
            int64_t dim,
            std::vector<int64_t>& ids,
            std::vector<float>& vectors) {
    ids.resize(count);
    vectors.resize(count * dim);
    std::mt19937 rng(static_cast<uint32_t>(first_id));
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = first_id + i;
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = distrib_real(rng);
        }
    }
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

}  // namespace

int
main() {
    vsag::init();

    int64_t dim = 128;
    uint64_t transition_threshold = 1000;
    std::string lazy_hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "lazy_hgraph": {
            "transition_threshold": 1000,
            "hgraph": {
                "base_quantization_type": "sq8",
                "max_degree": 26,
                "ef_construction": 100,
                "build_thread_count": 4
            }
        }
    }
    )";

    auto index = vsag::Factory::CreateIndex("lazy_hgraph", lazy_hgraph_build_parameters).value();

    std::vector<int64_t> small_ids;
    std::vector<float> small_vectors;
    auto small_batch = MakeDataset(0, 100, dim, small_ids, small_vectors);
    index->Add(small_batch);
    std::cout << "After small Add(), LazyHGraph contains: " << index->GetNumElements()
              << " vectors; it remains in its low-overhead FLAT phase." << std::endl;

    std::vector<int64_t> grow_ids;
    std::vector<float> grow_vectors;
    auto grow_batch = MakeDataset(
        100, transition_threshold - small_batch->GetNumElements(), dim, grow_ids, grow_vectors);
    index->Add(grow_batch);
    std::cout << "After reaching transition_threshold, LazyHGraph contains: "
              << index->GetNumElements() << " vectors and has converted to HGraph internally."
              << std::endl;

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(small_vectors.data())
                     ->Owner(false);
    auto search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    auto result = index->KnnSearch(query, 5, search_parameters).value();
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    return 0;
}
