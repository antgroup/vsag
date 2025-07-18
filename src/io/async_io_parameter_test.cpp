
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

#include "async_io_parameter.h"

#include <fmt/format.h>

#include "fixtures.h"
#include "parameter_test.h"

using namespace vsag;

TEST_CASE("AsyncIO Parameters Test", "[ut][AsyncIOParameters]") {
    fixtures::TempDir dir("async_io");
    auto path = dir.GenerateRandomFile();
    constexpr const char* param_str = R"(
        {{
            "type": "async_io",
            "file_path": "{}"
        }}
    )";
    auto param_json = JsonType::parse(fmt::format(param_str, path));
    auto param = std::make_shared<AsyncIOParameter>();
    param->FromJson(param_json);
    ParameterTest::TestToJson(param);
}
