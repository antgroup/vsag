
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

#include "sq4_uniform_quantizer_parameter.h"

#include <limits>

#include "parameter_test.h"
#include "unittest.h"
using namespace vsag;

TEST_CASE("SQ4 Uniform Quantizer Parameter ToJson Test", "[ut][SQ4UniformQuantizerParameter]") {
    std::string param_str = R"(
        {
            "sq4_uniform_trunc_rate": 0.06
        }
    )";
    auto param = std::make_shared<SQ4UniformQuantizerParameter>();
    param->FromJson(JsonType::Parse(param_str));
    REQUIRE(std::abs(param->trunc_rate_ - 0.06) < 1e-5F);
    ParameterTest::TestToJson(param);

    TestParamCheckCompatibility<SQ4UniformQuantizerParameter>(param_str);
}

TEST_CASE("SQ4 Uniform Quantizer Parameter validates truncation rate",
          "[ut][SQ4UniformQuantizerParameter]") {
    SECTION("keeps the default when the key is absent") {
        auto param = std::make_shared<SQ4UniformQuantizerParameter>();
        REQUIRE_NOTHROW(param->FromJson(JsonType{}));
        REQUIRE(param->trunc_rate_ == 0.05F);
    }

    SECTION("accepts finite boundary values") {
        for (const float rate : {0.0F, 0.05F, 0.5F}) {
            CAPTURE(rate);
            JsonType json;
            json[SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY].SetFloat(rate);

            auto param = std::make_shared<SQ4UniformQuantizerParameter>();
            REQUIRE_NOTHROW(param->FromJson(json));
            REQUIRE(param->trunc_rate_ == rate);
        }
    }

    SECTION("rejects non-finite and out-of-range values") {
        const float invalid_rates[] = {-0.01F,
                                       0.5001F,
                                       std::numeric_limits<float>::quiet_NaN(),
                                       std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity()};
        for (const float rate : invalid_rates) {
            CAPTURE(rate);
            JsonType json;
            json[SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY].SetFloat(rate);
            auto param = std::make_shared<SQ4UniformQuantizerParameter>();

            try {
                param->FromJson(json);
                FAIL("invalid SQ4 uniform truncation rate was accepted");
            } catch (const VsagException& error) {
                REQUIRE(error.error_.type == ErrorType::INVALID_ARGUMENT);
            }
        }
    }
}
