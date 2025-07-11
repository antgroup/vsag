
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

#include "buffer_io.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "basic_io_test.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("BufferIO Read & Write", "[ut][BufferIO]") {
    fixtures::TempDir dir("buffer_io");
    auto path = dir.GenerateRandomFile();
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto io = std::make_unique<BufferIO>(path, allocator.get());
    TestBasicReadWrite(*io);
}

TEST_CASE("BufferIO Parameter", "[ut][BufferIO]") {
    fixtures::TempDir dir("buffer_io");
    auto path = dir.GenerateRandomFile();
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr static const char* param_str = R"(
    {{
        "type": "buffer_io",
        "file_path" : "{}"
    }}
    )";
    auto json = JsonType::parse(fmt::format(param_str, path));
    auto io_param = IOParameter::GetIOParameterByJson(json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    auto io = std::make_unique<BufferIO>(io_param, common_param);
    TestBasicReadWrite(*io);
}

TEST_CASE("BufferIO Serialize & Deserialize", "[ut][BufferIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("buffer_io");
    auto path1 = dir.GenerateRandomFile();
    auto path2 = dir.GenerateRandomFile();
    auto wio = std::make_unique<BufferIO>(path1, allocator.get());
    auto rio = std::make_unique<BufferIO>(path2, allocator.get());
    TestSerializeAndDeserialize(*wio, *rio);
}
