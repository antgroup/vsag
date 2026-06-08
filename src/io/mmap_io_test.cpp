
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

#include "mmap_io.h"

#include <memory>

#include "basic_io_test.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("MMapIO Read & Write", "[ut][MMapIO]") {
    fixtures::TempDir dir("mmap_io");
    auto path = dir.GenerateRandomFile(false);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TestDistIOWrongInit<MMapIO>(allocator.get());
    auto io = std::make_unique<MMapIO>(path, allocator.get());
    TestBasicReadWrite(*io);
}

TEST_CASE("MMapIO Parameter", "[ut][MMapIO]") {
    fixtures::TempDir dir("mmap_io");
    auto path = dir.GenerateRandomFile();
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    static constexpr const char* param_str = R"(
    {{
        "type": "mmap_io",
        "file_path" : "{}"
    }}
    )";
    auto json = JsonType::Parse(fmt::format(param_str, path));
    auto io_param = IOParameter::GetIOParameterByJson(json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    auto io = std::make_unique<MMapIO>(io_param, common_param);
    TestBasicReadWrite(*io);
}

TEST_CASE("MMapIO Serialize & Deserialize", "[ut][MMapIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("mmap_io");
    auto path1 = dir.GenerateRandomFile();
    auto path2 = dir.GenerateRandomFile();
    auto wio = std::make_unique<MMapIO>(path1, allocator.get());
    auto rio = std::make_unique<MMapIO>(path2, allocator.get());
    TestSerializeAndDeserialize(*wio, *rio);
}

TEST_CASE("MMapIO directory path error", "[ut][MMapIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("mmap_io_dir_test");
    auto dir_path = dir.path;
    REQUIRE_THROWS(std::make_unique<MMapIO>(dir_path, allocator.get()));
}

TEST_CASE("MMapIO resize shrink", "[ut][MMapIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("mmap_io_resize");
    auto path = dir.GenerateRandomFile(false);
    auto io = std::make_unique<MMapIO>(path, allocator.get());

    std::vector<uint8_t> data(4096, 0xAB);
    io->Write(data.data(), data.size(), 0);

    io->Resize(8192);
    REQUIRE(io->size_ >= 8192);

    io->Resize(2048);
    REQUIRE(io->size_ == 2048);

    std::vector<uint8_t> read_buf(2048);
    REQUIRE(io->Read(2048, 0, read_buf.data()) == true);
    for (uint64_t i = 0; i < 2048; ++i) {
        REQUIRE(read_buf[i] == 0xAB);
    }
}

TEST_CASE("MMapIO MultiRead", "[ut][MMapIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("mmap_io_multi");
    auto path = dir.GenerateRandomFile(false);
    auto io = std::make_unique<MMapIO>(path, allocator.get());

    std::vector<uint8_t> data(256);
    for (uint64_t i = 0; i < 256; ++i) {
        data[i] = static_cast<uint8_t>(i);
    }
    io->Write(data.data(), data.size(), 0);

    std::vector<uint64_t> sizes = {64, 64, 64};
    std::vector<uint64_t> offsets = {0, 64, 128};
    std::vector<uint8_t> result(192);
    io->MultiRead(result.data(), sizes.data(), offsets.data(), 3);

    for (uint64_t i = 0; i < 192; ++i) {
        REQUIRE(result[i] == static_cast<uint8_t>(i));
    }
}

TEST_CASE("MMapIO existing file", "[ut][MMapIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("mmap_io_exist");
    auto path = dir.GenerateRandomFile(true);

    {
        auto io = std::make_unique<MMapIO>(path, allocator.get());
        std::vector<uint8_t> data(128, 0xCD);
        io->Write(data.data(), data.size(), 0);
    }

    auto io2 = std::make_unique<MMapIO>(path, allocator.get());
    std::vector<uint8_t> data2(64, 0xEF);
    io2->Write(data2.data(), data2.size(), 0);
    std::vector<uint8_t> read_buf(64);
    io2->Read(64, 0, read_buf.data());
    for (uint64_t i = 0; i < 64; ++i) {
        REQUIRE(read_buf[i] == 0xEF);
    }
}
