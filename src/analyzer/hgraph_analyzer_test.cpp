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

#include "hgraph_analyzer.h"

#include <fmt/format.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "algorithm/hgraph.h"
#include "fixtures.h"
#include "index/index_impl.h"
#include "vsag/vsag.h"

static vsag::IndexPtr
create_test_hgraph_index(int64_t dim, int64_t num_vectors, int64_t max_degree = 16) {
    auto params = vsag::JsonType::Parse(fmt::format(R"(
    {{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "hgraph": {{
            "max_degree": {},
            "ef_construction": 100,
            "base_codes": {{
                "codes_type": "flatten_codes",
                "io_params": {{ "type": "block_memory_io" }},
                "quantization_params": {{ "type": "fp32" }}
            }},
            "graph": {{
                "io_params": {{ "type": "block_memory_io" }}
            }}
        }}
    }}
    )",
                                                    dim,
                                                    max_degree));

    auto index_result = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(index_result.has_value());
    auto index = index_result.value();

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim, true, 47);
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto add_result = index->Build(dataset);
    REQUIRE(add_result.has_value());

    return index;
}

TEST_CASE("HGraphAnalyzer functionality", "[ut][hgraph_analyzer]") {
    int64_t dim = 32;
    int64_t num_vectors = 50;

    auto index = create_test_hgraph_index(dim, num_vectors);
    auto inner_index = std::dynamic_pointer_cast<vsag::IndexImpl<vsag::HGraph>>(index);
    REQUIRE(inner_index != nullptr);

    auto allocator = vsag::Engine::CreateDefaultAllocator();
    vsag::AnalyzerParam param(allocator.get());
    param.topk = 10;
    param.base_sample_size = 5;

    auto hgraph_analyzer = std::dynamic_pointer_cast<vsag::HGraphAnalyzer>(
        vsag::CreateAnalyzer(inner_index->GetInnerIndex().get(), param));
    REQUIRE(hgraph_analyzer != nullptr);

    SECTION("GetComponentCount") {
        auto components = hgraph_analyzer->GetComponentCount();

        REQUIRE_FALSE(components.empty());
        REQUIRE(*std::max_element(components.begin(), components.end()) > 0);
    }

    SECTION("GetDuplicateRatio") {
        auto ratio = hgraph_analyzer->GetDuplicateRatio();
        REQUIRE(ratio >= 0.0F);
        REQUIRE(ratio <= 1.0F);
    }

    SECTION("GetBaseAvgDistance") {
        auto avg_distance = hgraph_analyzer->GetBaseAvgDistance();
        REQUIRE(avg_distance >= 0.0F);
    }

    SECTION("GetNeighborRecall") {
        auto recall = hgraph_analyzer->GetNeighborRecall();
        REQUIRE(recall >= 0.0F);
        REQUIRE(recall <= 1.0F);
    }
}