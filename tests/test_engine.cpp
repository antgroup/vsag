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

#include <catch2/catch_test_macros.hpp>

#include "vsag/vsag.h"

TEST_CASE("Engine uses the supported index registry", "[ft][engine]") {
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);

    auto supported = engine.CreateIndex("brute_force", R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 4,
        "index_param": {
            "base_quantization_type": "fp32"
        }
    }
    )");
    REQUIRE(supported.has_value());

    std::vector<int64_t> ids = {0, 1, 2, 3};
    std::vector<float> vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
        2.0F,
        2.0F,
        2.0F,
        2.0F,
        3.0F,
        3.0F,
        3.0F,
        3.0F,
    };
    auto base = vsag::Dataset::Make()
                    ->NumElements(ids.size())
                    ->Dim(4)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);
    auto build_result = supported.value()->Build(base);
    REQUIRE(build_result.has_value());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(4)
                     ->Float32Vectors(vectors.data() + 8)
                     ->Owner(false);
    auto search_result = supported.value()->KnnSearch(query, 1, "{}");
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() == 1);
    REQUIRE(search_result.value()->GetIds()[0] == 2);

    for (const auto* removed_name : {"hnsw", "fresh_hnsw", "diskann"}) {
        auto removed =
            engine.CreateIndex(removed_name, R"({"dtype":"float32","metric_type":"l2","dim":4})");
        INFO(removed_name);
        REQUIRE_FALSE(removed.has_value());
        REQUIRE(removed.error().type == vsag::ErrorType::UNSUPPORTED_INDEX);
    }

    engine.Shutdown();
}
