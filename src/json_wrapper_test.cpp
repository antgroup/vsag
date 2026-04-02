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

#include "json_wrapper.h"

#include <catch2/catch_test_macros.hpp>
#include <string>

TEST_CASE("JsonWrapper Copy Empty Wrapper", "[ut][json_wrapper]") {
    vsag::JsonWrapper w1;
    vsag::JsonWrapper w2 = w1;
    CHECK_FALSE(w2.Contains("any_key"));
}

TEST_CASE("JsonWrapper Copy With String Data", "[ut][json_wrapper]") {
    auto w1 = vsag::JsonWrapper::Parse(R"({"key": "value"})");
    vsag::JsonWrapper w2 = w1;
    CHECK(w2.Contains("key"));
    CHECK(w2["key"].GetString() == "value");
}

TEST_CASE("JsonWrapper Copy With Integer Data", "[ut][json_wrapper]") {
    auto w1 = vsag::JsonWrapper::Parse(R"({"num": 42})");
    vsag::JsonWrapper w2 = w1;
    CHECK(w2.Contains("num"));
    CHECK(w2["num"].GetInt() == 42);
}

TEST_CASE("JsonWrapper Copy Independence", "[ut][json_wrapper]") {
    auto w1 = vsag::JsonWrapper::Parse(R"({"key": "value1"})");
    vsag::JsonWrapper w2 = w1;
    w2["key"].SetString("value2");
    CHECK(w1["key"].GetString() == "value1");
    CHECK(w2["key"].GetString() == "value2");
}

TEST_CASE("JsonWrapper Copy Nested Json", "[ut][json_wrapper]") {
    auto w1 = vsag::JsonWrapper::Parse(R"({"outer": {"inner": "data"}})");
    vsag::JsonWrapper w2 = w1;
    CHECK(w2.Contains("outer"));
    CHECK(w2["outer"].Contains("inner"));
    CHECK(w2["outer"]["inner"].GetString() == "data");
}

TEST_CASE("JsonWrapper Copy Non-owning Wrapper", "[ut][json_wrapper]") {
    vsag::JsonWrapper sub;
    {
        auto w1 = vsag::JsonWrapper::Parse(R"({"key": "value"})");
        sub = w1["key"];
    }
    vsag::JsonWrapper w2(sub);
    CHECK(w2.GetString() == "value");
}
