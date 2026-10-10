
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
#include <stdexcept>

int
main() {
    auto created = vsag::Factory::CreateIndex("uhg", R"({
        "dtype":"float32", "metric_type":"ip", "dim":2,
        "index_param":{"neighbors_per_alpha":3,"construction_candidates":3}
    })");
    if (!created.has_value()) {
        throw std::runtime_error(created.error().message);
    }
    auto index = created.value();
    int64_t ids[]{10, 20, 30};
    float dense[]{1, 0, 0.1F, 0, 0.7F, 0};
    uint32_t term = 3;
    float values[]{0.1F, 1, 0.7F};
    vsag::SparseVector sparse[]{{1, &term, values}, {1, &term, values + 1}, {1, &term, values + 2}};
    auto base = vsag::Dataset::Make()
                    ->Owner(false)
                    ->NumElements(3)
                    ->Dim(2)
                    ->Ids(ids)
                    ->Float32Vectors(dense)
                    ->SparseVectors(sparse);
    auto built = index->Build(base);
    if (!built.has_value()) {
        throw std::runtime_error(built.error().message);
    }
    float query_dense[]{1, 0};
    float query_value = 1;
    vsag::SparseVector query_sparse{1, &term, &query_value};
    auto query = vsag::Dataset::Make()
                     ->Owner(false)
                     ->NumElements(1)
                     ->Dim(2)
                     ->Float32Vectors(query_dense)
                     ->SparseVectors(&query_sparse);
    // One topology: sparse favors ID 20, hybrid favors ID 30, dense favors ID 10.
    for (const auto* parameters : {R"({"uhg":{"alpha":0,"ef_search":10,"entry_points":1}})",
                                   R"({"uhg":{"alpha":0.5,"ef_search":10,"entry_points":1}})",
                                   R"({"uhg":{"alpha":1,"ef_search":10,"entry_points":1}})"}) {
        auto result = index->KnnSearch(query, 1, parameters);
        if (!result.has_value()) {
            throw std::runtime_error(result.error().message);
        }
        std::cout << parameters << " -> id=" << result.value()->GetIds()[0]
                  << ", distance=" << result.value()->GetDistances()[0] << '\n';
    }
    return 0;
}
