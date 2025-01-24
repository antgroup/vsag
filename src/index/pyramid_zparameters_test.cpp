
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

#include "pyramid_zparameters.h"

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "parameter_test.h"

TEST_CASE("Pyramid Parameyers Test", "[ut][pyramid_param]") {
    auto param_str = R"(
        {
            "odescent": {
                "io_params": {
                    "type": "memory_io"
                },
                "max_degree": 16,
                "alpha": 1.5,
                "graph_iter_turn": 10,
                "neighbor_sample_rate": 0.5
            },
            "base_codes": {
                "io_params": {
                    "type": "memory_io"
                },
                "quantization_params": {
                    "type": "fp32"
                }
            }
        }
    )";
    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto param = std::make_shared<vsag::PyramidParameters>();
    param->FromJson(param_json);
    fixtures::dist_t alpha = param->alpha;
    fixtures::dist_t sample_rate = param->sample_rate;
    REQUIRE(param->max_degree == 16);
    REQUIRE(alpha == 1.5);
    REQUIRE(param->turn == 10);
    REQUIRE(sample_rate == 0.5);
    vsag::ParameterTest::TestToJson(param);
}
