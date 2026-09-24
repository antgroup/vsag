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

#include "saq_quantizer_parameter.h"

#include "parameter_test.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("SAQ Quantizer Parameter", "[ut][SAQQuantizerParameter]") {
    constexpr auto parameters = R"({
        "type": "saq",
        "saq_avg_bits": 3.5,
        "saq_segment_count": 2,
        "saq_adjustment_rounds": 8,
        "saq_use_pca": false,
        "saq_random_rotation": false
    })";
    auto param = std::make_shared<SAQQuantizerParameter>();
    param->FromJson(JsonType::Parse(parameters));

    REQUIRE(param->avg_bits_ == 3.5F);
    REQUIRE(param->segment_count_ == 2);
    REQUIRE(param->adjustment_rounds_ == 8);
    REQUIRE_FALSE(param->use_pca_);
    REQUIRE_FALSE(param->random_rotation_);
    ParameterTest::TestToJson(param);
    TestParamCheckCompatibility<SAQQuantizerParameter>(parameters);
}

TEST_CASE("SAQ Quantizer Parameter validation", "[ut][SAQQuantizerParameter]") {
    auto parse = [](const std::string& value) {
        auto param = std::make_shared<SAQQuantizerParameter>();
        param->FromJson(JsonType::Parse(value));
    };

    REQUIRE_THROWS(parse(R"({"saq_avg_bits": 0.5})"));
    REQUIRE_THROWS(parse(R"({"saq_avg_bits": 9})"));
    REQUIRE_THROWS(parse(R"({"saq_avg_bits": "four"})"));
    REQUIRE_THROWS(parse(R"({"saq_segment_count": 1.5})"));
    REQUIRE_THROWS(parse(R"({"saq_segment_count": -1})"));
    REQUIRE_THROWS(parse(R"({"saq_adjustment_rounds": -1})"));
    REQUIRE_THROWS(parse(R"({"saq_adjustment_rounds": 33})"));
    REQUIRE_THROWS(parse(R"({"saq_use_pca": 1})"));
    REQUIRE_THROWS(parse(R"({"saq_random_rotation": 1})"));
}
