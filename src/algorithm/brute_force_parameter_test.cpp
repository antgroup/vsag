
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

#include "brute_force_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

TEST_CASE("BruteForce Parameters CheckCompatibility",
          "[ut][BruteForceParameter][CheckCompatibility]") {
    auto param_str = R"({
        "io_params": {
            "type": "block_memory_io"
        },
        "quantization_params": {
            "type": "sq8"
        },
        "type": "brute_force"
    })";

    SECTION("wrong parameter type") {
        auto param = std::make_shared<vsag::BruteForceParameter>();
        param->FromString(param_str);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }
}
