
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

#include "fp32_quantizer_parameter.h"

#include <map>

#include "parameter_test.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("FP32 Quantizer Parameter ToJson Test", "[ut][FP32QuantizerParameter]") {
    std::string param_str = "{}";
    auto param = std::make_shared<FP32QuantizerParameter>();
    JsonType param_json = JsonType::Parse(param_str);
    param->FromJson(param_json);
    ParameterTest::TestToJson(param);
}

namespace {
class CompatibilityTestParameter : public Parameter {
public:
    explicit CompatibilityTestParameter(JsonType json) : json_(std::move(json)) {
    }

    void
    FromJson(const JsonType& json) override {
        json_ = json;
    }

    JsonType
    ToJson() const override {
        return json_;
    }

private:
    JsonType json_;
};
}  // namespace

TEST_CASE("CompatibilityReport collects all JSON differences", "[ut][CompatibilityReport]") {
    auto left = std::make_shared<CompatibilityTestParameter>(
        JsonType::Parse(R"({"same":1,"first":2,"nested":{"second":3},"a/b":7})"));
    auto right = std::make_shared<CompatibilityTestParameter>(
        JsonType::Parse(R"({"same":1,"first":4,"nested":{"second":5},"a/b":8,"extra":6})"));

    auto report = left->CollectCompatibilityIssues(right);
    REQUIRE_FALSE(report.IsCompatible());
    REQUIRE(report.issues.size() == 4);
    std::map<std::string, std::string> issues;
    for (const auto& issue : report.issues) {
        issues.emplace(issue.path, issue.message);
    }
    REQUIRE(issues.at("/first") == "values differ");
    REQUIRE(issues.at("/nested/second") == "values differ");
    REQUIRE(issues.at("/a~1b") == "values differ");
    REQUIRE(issues.at("/extra") == "unexpected field");
    REQUIRE_FALSE(left->CheckCompatibility(right));
}
