// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/mmap_io/mmap_io.h"
#include "storage/serialization.h"
#include "unittest.h"

namespace vsag {
namespace {

void
RequireMMapContent(const MMapIO& io, const std::vector<uint8_t>& expected) {
    REQUIRE(io.Size() == expected.size());
    std::vector<uint8_t> actual(expected.size());
    REQUIRE(io.ReadAt(0, actual.size(), actual.data()));
    REQUIRE(actual == expected);
}

}  // namespace

TEST_CASE("MMapIO contract and existing file size", "[ut][MMapIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("mmap_io_canonical_contract");
    auto path = dir.GenerateRandomFile(false);
    std::vector<uint8_t> expected(5000);
    for (uint64_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<uint8_t>(i * 13);
    }
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(expected.data()), expected.size());
    }

    MMapIO io(path, allocator.get());
    REQUIRE_FALSE(MMapIO::InMemory);
    RequireMMapContent(io, expected);

    auto lease = io.Acquire(4090, 32);
    REQUIRE(lease);
    REQUIRE(std::memcmp(lease.Data(), expected.data() + 4090, 32) == 0);

    io.Resize(8192);
    REQUIRE(io.Size() == 8192);
    REQUIRE(std::filesystem::file_size(path) == 8192);
    io.Resize(2048);
    REQUIRE(io.Size() == 2048);
    REQUIRE(std::filesystem::file_size(path) == 4096);
}

TEST_CASE("MMapIO scatter and serialization compatibility", "[ut][MMapIO][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("mmap_io_canonical_compatibility");
    auto compatibility_path = dir.GenerateRandomFile(false);
    auto canonical_path = dir.GenerateRandomFile(false);
    auto second_compatibility_path = dir.GenerateRandomFile(false);
    auto second_canonical_path = dir.GenerateRandomFile(false);

    std::vector<uint8_t> data(8193);
    for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i * 37);
    }

    MMapIO compatibility_source(compatibility_path, allocator.get());
    MMapIO canonical_destination(canonical_path, allocator.get());
    compatibility_source.Write(data.data(), data.size(), 0);
    std::stringstream old_stream;
    IOStreamWriter old_writer(old_stream);
    compatibility_source.Serialize(old_writer);
    IOStreamReader old_reader(old_stream);
    canonical_destination.Deserialize(old_reader);
    RequireMMapContent(canonical_destination, data);

    std::array<uint8_t, 31> first{};
    std::array<uint8_t, 47> second{};
    std::array<ReadRequest, 2> requests{{
        {first.data(), 4090, first.size()},
        {second.data(), 8000, second.size()},
    }};
    REQUIRE(canonical_destination.ReadMany(requests.data(), requests.size()));
    REQUIRE(std::memcmp(first.data(), data.data() + 4090, first.size()) == 0);
    REQUIRE(std::memcmp(second.data(), data.data() + 8000, second.size()) == 0);

    MMapIO canonical_source(second_canonical_path, allocator.get());
    MMapIO compatibility_destination(second_compatibility_path, allocator.get());
    canonical_source.WriteAt(0, data.data(), data.size());
    std::stringstream new_stream;
    IOStreamWriter new_writer(new_stream);
    canonical_source.Serialize(new_writer);
    IOStreamReader new_reader(new_stream);
    compatibility_destination.Deserialize(new_reader);
    std::vector<uint8_t> actual(data.size());
    REQUIRE(compatibility_destination.Read(actual.size(), 0, actual.data()));
    REQUIRE(actual == data);
}

TEST_CASE("MMapIO removes newly created backing file", "[ut][MMapIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("mmap_io_canonical_ownership");
    auto path = dir.GenerateRandomFile(false);
    REQUIRE_FALSE(std::filesystem::exists(path));
    {
        MMapIO io(path, allocator.get());
        REQUIRE(std::filesystem::exists(path));
    }
    REQUIRE_FALSE(std::filesystem::exists(path));
}

TEST_CASE("MMapIO normalizes filesystem inspection errors", "[ut][MMapIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("mmap_io_filesystem_error");
    const auto regular_file = dir.GenerateRandomFile();
    const auto invalid_path = regular_file + "/child";

    REQUIRE_THROWS_AS(MMapIO(invalid_path, allocator.get()), VsagException);
    REQUIRE_FALSE(std::filesystem::exists(invalid_path));
}

}  // namespace vsag
