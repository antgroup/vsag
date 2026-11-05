
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

#pragma once

#include <catch2/catch_test_macros.hpp>

#include "parameter.h"

namespace vsag {
class ParameterTest {
public:
    static void
    TestToJson(const ParamPtr& param) {
        auto json1 = param->ToJson();
        auto str1 = param->ToString();
        param->FromJson(json1);
        REQUIRE(param->ToString() == str1);
    }
};

class EmptyParameter : public Parameter {
    void
    FromJson(const JsonType& json) override {
    }

    JsonType
    ToJson() const override {
        return JsonType();
    }
};

template <typename T>
void
TestParamCheckCompatibility(const std::string& param_str) {
    auto param = std::make_shared<T>();
    param->FromString(param_str);
    REQUIRE(param->CheckCompatibility(param));
    REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
}
}  // namespace vsag
