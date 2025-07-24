
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

#include "sindi_parameter.h"

#include <catch2/catch_test_macros.hpp>

#include "parameter_test.h"

TEST_CASE("SINDI Index Parameters Test", "[ut][SINDIParameters]") {
    auto param_str = R"({
        "use_reorder": true,
        "query_prune_ratio": 0.9,
        "doc_prune_ratio": 0.8,
        "term_prune_ratio": 0.7,
        "window_size": 66666
    })";
    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto param = std::make_shared<vsag::SINDIParameters>();
    param->FromJson(param_json);
    REQUIRE(param->use_reorder == true);
    REQUIRE(std::abs(param->query_prune_ratio - 0.9) < 1e-3);
    REQUIRE(std::abs(param->doc_prune_ratio - 0.8) < 1e-3);
    REQUIRE(std::abs(param->term_prune_ratio - 0.7) < 1e-3);
    REQUIRE(param->window_size == 66666);

    vsag::ParameterTest::TestToJson(param);
}
