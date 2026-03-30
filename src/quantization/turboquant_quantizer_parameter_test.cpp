
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

#include "turboquant_quantizer_parameter.h"

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "parameter_test.h"

using namespace vsag;

struct TurboQuantDefaultParam {
    int bits_per_dim = 4;
    bool use_fht = true;
    bool enable_qjl = true;
    int qjl_projection_dim = 64;
};

std::string
generate_turboquant_param(const TurboQuantDefaultParam& param) {
    static constexpr auto param_str = R"(
        {{
            "type": "turboquant",
            "turboquant_bits_per_dim": {},
            "use_fht": {},
            "turboquant_enable_qjl": {},
            "turboquant_qjl_projection_dim": {}
        }}
    )";
    return fmt::format(
        param_str, param.bits_per_dim, param.use_fht, param.enable_qjl, param.qjl_projection_dim);
}

TEST_CASE("TurboQuant Quantizer Parameter ToJson Test", "[ut][TurboQuantQuantizerParameter]") {
    TurboQuantDefaultParam default_param;
    auto param_str = generate_turboquant_param(default_param);
    auto param = std::make_shared<TurboQuantizerParameter>();
    param->FromString(param_str);
    ParameterTest::TestToJson(param);

    REQUIRE(param->bits_per_dim_ == 4);
    REQUIRE(param->use_fht_);
    REQUIRE(param->enable_qjl_);
    REQUIRE(param->qjl_projection_dim_ == 64);

    TestParamCheckCompatibility<TurboQuantizerParameter>(param_str);
}

TEST_CASE("TurboQuant Quantizer Parameter CheckCompatibility",
          "[ut][TurboQuantQuantizerParameter]") {
    TurboQuantDefaultParam param_1;
    TurboQuantDefaultParam param_2;
    param_2.bits_per_dim = 6;

    auto turboquant_param_1 = std::make_shared<TurboQuantizerParameter>();
    auto turboquant_param_2 = std::make_shared<TurboQuantizerParameter>();
    turboquant_param_1->FromString(generate_turboquant_param(param_1));
    turboquant_param_2->FromString(generate_turboquant_param(param_2));

    REQUIRE_FALSE(turboquant_param_1->CheckCompatibility(turboquant_param_2));
}

TEST_CASE("Wrong turboquant_bits_per_dim parameter", "[ut][TurboQuantQuantizerParameter]") {
    auto wrong_bits_per_dim = GENERATE(0, 1, 9, 16);
    TurboQuantDefaultParam default_param;
    default_param.bits_per_dim = wrong_bits_per_dim;
    auto param = std::make_shared<TurboQuantizerParameter>();
    REQUIRE_THROWS(param->FromString(generate_turboquant_param(default_param)));
}
