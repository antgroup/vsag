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

#include <vector>

#include "unittest.h"
#include "vsag/dataset.h"
#include "vsag/factory.h"
#include "vsag/filter.h"

namespace {

vsag::DatasetPtr
MakeBuildDataset(std::vector<int64_t>& ids,
                 std::vector<float>& vectors,
                 int64_t count,
                 int64_t dim) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    return dataset;
}

}  // namespace

TEST_CASE("HGraph batch build stays searchable without an upper level",
          "[ut][hgraph][build][odescent]") {
    // Upper levels drive the entry point of a batch-built HGraph. The level sampler draws from a
    // fixed-seed engine, but mapping those draws to uniforms is implementation defined: on libc++
    // six points can all fall on level zero. The entry point then stayed INVALID_ENTRY_POINT, so
    // Build succeeded while every search returned an empty result set.
    constexpr int64_t dim = 2;
    constexpr int64_t total = 6;
    constexpr const char* params = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 2,
            "index_param": {
                "base_quantization_type": "fp32",
                "graph_type": "odescent",
                "max_degree": 6,
                "alpha": 1.2,
                "graph_iter_turn": 6,
                "neighbor_sample_rate": 0.3
            }
        })";
    std::vector<int64_t> ids{0, 1, 2, 3, 4, 5};
    std::vector<float> vectors{0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6};

    auto index = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(index.has_value());
    auto built = index.value()->Build(MakeBuildDataset(ids, vectors, total, dim));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    REQUIRE(index.value()->GetNumElements() == total);

    auto result = index.value()->KnnSearch(MakeBuildDataset(ids, vectors, 1, dim),
                                           total,
                                           R"({"hgraph":{"ef_search":100,"use_mci":false}})",
                                           vsag::FilterPtr{});
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == total);
}
