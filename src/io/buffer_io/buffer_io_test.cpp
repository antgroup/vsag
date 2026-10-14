
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

#include "io/buffer_io/buffer_io.h"

#include <memory>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "io/common/io_contract_test.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("BufferIO Read & Write", "[ut][BufferIO]") {
    fixtures::TempDir dir("buffer_io");
    auto path = dir.GenerateRandomFile(false);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TestDistIOWrongInit<BufferIO>(allocator.get());
    auto io = std::make_unique<BufferIO>(path, allocator.get());
    TestBasicReadWrite(*io);
}

TEST_CASE("BufferIO direct read empty range", "[ut][BufferIO]") {
    fixtures::TempDir dir("buffer_io");
    auto path = dir.GenerateRandomFile(false);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto io = std::make_unique<BufferIO>(path, allocator.get());

    bool need_release = true;
    auto result = io->Read(0, 0, need_release);
    REQUIRE(result == nullptr);
    REQUIRE_FALSE(need_release);

    const uint8_t value = 1;
    io->Write(&value, 1, 0);

    need_release = true;
    result = io->Read(1, 1, need_release);
    REQUIRE(result == nullptr);
    REQUIRE_FALSE(need_release);
}

TEST_CASE("BufferIO Parameter", "[ut][BufferIO]") {
    fixtures::TempDir dir("buffer_io");
    auto path = dir.GenerateRandomFile();
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    static constexpr const char* param_str = R"(
    {{
        "type": "buffer_io",
        "file_path" : "{}"
    }}
    )";
    auto json = JsonType::Parse(fmt::format(param_str, path));
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

TEST_CASE("BufferIO enabled prefetch hint preserves reads", "[ut][BufferIO]") {
    fixtures::TempDir dir("buffer_io_prefetch");
    auto path = dir.GenerateRandomFile(false);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto parameter = std::make_shared<BufferIOParameter>();
    parameter->path_ = path;
    parameter->enable_prefetch_hint_ = true;
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    IOParamPtr io_parameter = parameter;
    BufferIO io(io_parameter, common_param);

    std::vector<uint8_t> data(8192, 0xCD);
    io.Write(data.data(), data.size(), 0);
    io.Prefetch(123, 256);
    io.Prefetch(data.size(), 64);

    std::vector<uint8_t> result(256);
    REQUIRE(io.Read(result.size(), 123, result.data()));
    REQUIRE(std::all_of(result.begin(), result.end(), [](uint8_t value) { return value == 0xCD; }));
}
