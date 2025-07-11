
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

#include "async_io.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "basic_io_test.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("AsyncIO Read And Write", "[ut][AsyncIO]") {
    fixtures::TempDir dir("async_io");
    auto path = dir.GenerateRandomFile();
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto io = std::make_unique<AsyncIO>(path, allocator.get());
    TestBasicReadWrite(*io);
}

TEST_CASE("AsyncIO Parameter", "[ut][AsyncIO]") {
    fixtures::TempDir dir("async_io");
    auto path = dir.GenerateRandomFile();
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr auto param_str = R"(
    {{
        "type": "async_io",
        "file_path" : "{}"
    }}
    )";
    auto json = JsonType::parse(fmt::format(param_str, path));
    auto io_param = IOParameter::GetIOParameterByJson(json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    auto io = std::make_unique<AsyncIO>(io_param, common_param);
    TestBasicReadWrite(*io);
}

TEST_CASE("AsyncIO Serialize & Deserialize", "[ut][AsyncIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("async_io");
    auto path1 = dir.GenerateRandomFile();
    auto path2 = dir.GenerateRandomFile();
    auto wio = std::make_unique<AsyncIO>(path1, allocator.get());
    auto rio = std::make_unique<AsyncIO>(path2, allocator.get());
    TestSerializeAndDeserialize(*wio, *rio);
}
