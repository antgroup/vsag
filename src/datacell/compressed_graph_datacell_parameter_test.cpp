
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

#include "compressed_graph_datacell_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

namespace vsag {

TEST_CASE("CompressedGraphDatacellParameter ToJson Test",
          "[ut][CompressedGraphDatacellParameter]") {
    std::string param_str = R"(
        {
            "max_degree": 100,
            "graph_storage_type": "compressed"
        }
        )";
    auto param = std::make_shared<CompressedGraphDatacellParameter>();
    auto json = JsonType::Parse(param_str);
    param->FromJson(json);
    ParameterTest::TestToJson(param);
}

}  // namespace vsag
