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

#include "analyzer.h"

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "algorithm/hgraph.h"
#include "fixtures.h"
#include "index/index_impl.h"
#include "vsag/vsag.h"

TEST_CASE("CreateAnalyzer with null index", "[ut][analyzer]") {
    auto allocator = vsag::Engine::CreateDefaultAllocator();
    vsag::AnalyzerParam param(allocator.get());

    REQUIRE_THROWS_AS(vsag::CreateAnalyzer(nullptr, param), vsag::VsagException);
}

TEST_CASE("CreateAnalyzer with HGraph index", "[ut][analyzer]") {
    int64_t dim = 64;
    int64_t num_vectors = 100;

    auto params = vsag::JsonType::Parse(fmt::format(R"(
    {{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "hgraph": {{
            "max_degree": 16,
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
                                                    dim));

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

    auto inner_index = std::dynamic_pointer_cast<vsag::IndexImpl<vsag::HGraph>>(index);
    REQUIRE(inner_index != nullptr);

    auto allocator = vsag::Engine::CreateDefaultAllocator();
    vsag::AnalyzerParam param(allocator.get());
    param.topk = 10;
    param.base_sample_size = 5;

    {
        auto analyzer = vsag::CreateAnalyzer(inner_index->GetInnerIndex().get(), param);
        REQUIRE(analyzer != nullptr);
    }
}