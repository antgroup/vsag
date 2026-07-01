
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

#include "param/schema.h"

#include "parameter.h"
#include "unittest.h"

namespace vsag::param {

namespace {

// Tiny sub-parameter used to exercise SubParamField. Stores a single int64
// value and implements the minimal Parameter contract.
class TestSubParam : public vsag::Parameter {
public:
    int64_t value{0};

    void
    FromJson(const vsag::JsonType& json) override {
        if (json.Contains("value")) {
            this->value = json["value"].GetInt();
        }
    }

    vsag::JsonType
    ToJson() const override {
        vsag::JsonType j;
        j["value"].SetInt(this->value);
        return j;
    }

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override {
        auto p = std::dynamic_pointer_cast<TestSubParam>(other);
        return p != nullptr && p->value == this->value;
    }
};

std::shared_ptr<TestSubParam>
CreateTestSub(const vsag::JsonType& json) {
    auto sub = std::make_shared<TestSubParam>();
    sub->FromJson(json);
    return sub;
}

// Synthetic schema-driven parameter exercising every kind of field.
class TestParam : public vsag::Parameter {
public:
    uint64_t dim{0};
    std::string name;
    std::shared_ptr<TestSubParam> sub{nullptr};

    VSAG_PARAM_SCHEMA_DECL();

    void
    FromJson(const vsag::JsonType& json) override {
        schema().Parse(this, json);
        schema().CheckUnknownKeys(json);
    }

    vsag::JsonType
    ToJson() const override {
        vsag::JsonType j;
        schema().Serialize(this, j);
        return j;
    }

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override {
        auto p = std::dynamic_pointer_cast<TestParam>(other);
        if (p == nullptr) {
            return false;
        }
        return schema().Equal(this, p.get());
    }
};

VSAG_PARAM_SCHEMA(TestParam)
VSAG_PARAM(uint64_t, dim, "dim", 128, ::vsag::param::Range<uint64_t>(1, 1000000))
VSAG_PARAM(std::string, name, "name", "default")
VSAG_PARAM_SUBPARAM(TestSubParam, sub, "sub", CreateTestSub)
VSAG_PARAM_TYPE_TAG("type", "test_param")
VSAG_PARAM_SCHEMA_END()

}  // namespace

TEST_CASE("Schema parses scalar and sub-parameter fields", "[ut][param][schema]") {
    auto param_str = R"({
        "dim": 256,
        "name": "hello",
        "sub": {"value": 42},
        "type": "test_param"
    })";

    TestParam param;
    param.FromString(param_str);

    REQUIRE(param.dim == 256);
    REQUIRE(param.name == "hello");
    REQUIRE(param.sub != nullptr);
    REQUIRE(param.sub->value == 42);
}

TEST_CASE("Schema produces stable JSON round-trip", "[ut][param][schema]") {
    auto param_str = R"({
        "dim": 256,
        "name": "hello",
        "sub": {"value": 42},
        "type": "test_param"
    })";

    auto param = std::make_shared<TestParam>();
    param->FromString(param_str);
    auto str1 = param->ToString();

    auto param2 = std::make_shared<TestParam>();
    param2->FromString(str1);
    auto str2 = param2->ToString();

    REQUIRE(str1 == str2);
    REQUIRE(param->CheckCompatibility(param2));
}

TEST_CASE("Schema rejects out-of-range scalar via Range validator", "[ut][param][schema]") {
    auto param_str = R"({
        "dim": 0,
        "name": "x",
        "sub": {"value": 1},
        "type": "test_param"
    })";

    TestParam param;
    REQUIRE_THROWS_AS(param.FromString(param_str), vsag::VsagException);
}

TEST_CASE("Schema rejects unknown JSON keys", "[ut][param][schema]") {
    auto param_str = R"({
        "dim": 256,
        "name": "x",
        "sub": {"value": 1},
        "type": "test_param",
        "unknown_field": true
    })";

    TestParam param;
    REQUIRE_THROWS_AS(param.FromString(param_str), vsag::VsagException);
}

}  // namespace vsag::param
