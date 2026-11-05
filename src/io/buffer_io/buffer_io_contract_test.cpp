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
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/buffer_io/buffer_io.h"
#include "io/buffer_io/buffer_io_parameter.h"
#include "io/read_cache/lru_page_cache.h"
#include "storage/serialization.h"
#include "unittest.h"

namespace vsag {
namespace {

class BufferCountingAllocator : public Allocator {
public:
    std::string
    Name() override {
        return "BufferCountingAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        allocations_.fetch_add(1, std::memory_order_relaxed);
        return std::malloc(size);
    }

    void
    Deallocate(void* data) override {
        deallocations_.fetch_add(1, std::memory_order_relaxed);
        std::free(data);
    }

    void*
    Reallocate(void* data, uint64_t size) override {
        return std::realloc(data, size);
    }

    void
    Reset() {
        allocations_.store(0, std::memory_order_relaxed);
        deallocations_.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t
    Allocations() const {
        return allocations_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t
    Deallocations() const {
        return deallocations_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> allocations_{0};
    std::atomic<uint64_t> deallocations_{0};
};

void
RequireBufferContent(const BufferIO& io, const std::vector<uint8_t>& expected) {
    REQUIRE(io.Size() == expected.size());
    std::vector<uint8_t> actual(expected.size());
    REQUIRE(io.ReadAt(0, actual.size(), actual.data()));
    REQUIRE(actual == expected);
}

void
RequireSameBufferContent(const BufferIO& compatibility, const BufferIO& canonical) {
    REQUIRE(compatibility.Size() == canonical.Size());
    std::vector<uint8_t> compatibility_data(compatibility.Size());
    std::vector<uint8_t> canonical_data(canonical.Size());
    REQUIRE(compatibility.Read(compatibility.Size(), 0, compatibility_data.data()));
    REQUIRE(canonical.ReadAt(0, canonical.Size(), canonical_data.data()));
    REQUIRE(compatibility_data == canonical_data);
}

}  // namespace

TEST_CASE("BufferIO contract, ownership, and existing size", "[ut][BufferIO]") {
    BufferCountingAllocator allocator;
    fixtures::TempDir dir("buffer_io_canonical_contract");
    auto existing_path = dir.GenerateRandomFile(false);
    std::vector<uint8_t> expected(4097);
    for (uint64_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<uint8_t>(i * 19);
    }
    {
        std::ofstream stream(existing_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(expected.data()), expected.size());
    }
    {
        BufferIO io(existing_path, &allocator);
        RequireBufferContent(io, expected);

        allocator.Reset();
        {
            auto lease = io.Acquire(13, 127);
            REQUIRE(lease);
            REQUIRE(lease.Size() == 127);
            REQUIRE(std::memcmp(lease.Data(), expected.data() + 13, 127) == 0);
            REQUIRE(allocator.Allocations() == 1);
        }
        REQUIRE(allocator.Deallocations() == 1);

        bool need_release = false;
        const uint8_t* compatibility_data = io.Read(64, 7, need_release);
        REQUIRE(compatibility_data != nullptr);
        REQUIRE(need_release);
        REQUIRE(std::memcmp(compatibility_data, expected.data() + 7, 64) == 0);
        io.Release(compatibility_data);
    }
    REQUIRE(std::filesystem::exists(existing_path));

    auto temporary_path = dir.GenerateRandomFile(false);
    {
        BufferIO io(temporary_path, &allocator);
        REQUIRE(std::filesystem::exists(temporary_path));
    }
    REQUIRE_FALSE(std::filesystem::exists(temporary_path));
}

TEST_CASE("BufferIO normalizes filesystem inspection errors", "[ut][BufferIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("buffer_io_filesystem_error");
    const auto regular_file = dir.GenerateRandomFile();
    const auto invalid_path = regular_file + "/child";

    REQUIRE_THROWS_AS(BufferIO(invalid_path, allocator.get()), VsagException);
    REQUIRE_FALSE(std::filesystem::exists(invalid_path));
}

TEST_CASE("BufferIO shrink truncates its backing file", "[ut][BufferIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("buffer_io_canonical_shrink");
    auto path = dir.GenerateRandomFile();
    std::vector<uint8_t> data(1024, 0x5A);

    {
        BufferIO io(path, allocator.get());
        io.WriteAt(0, data.data(), data.size());
        io.Shrink(127);
        REQUIRE(io.Size() == 127);
        REQUIRE(std::filesystem::file_size(path) == 127);
    }

    BufferIO reopened(path, allocator.get());
    REQUIRE(reopened.Size() == 127);
    std::vector<uint8_t> actual(127);
    REQUIRE(reopened.ReadAt(0, actual.size(), actual.data()));
    REQUIRE(actual == std::vector<uint8_t>(127, 0x5A));
}

TEST_CASE("BufferIO scatter and randomized differential", "[ut][BufferIO][differential]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("buffer_io_canonical_differential");
    BufferIO compatibility(dir.GenerateRandomFile(false), allocator.get());
    BufferIO canonical(dir.GenerateRandomFile(false), allocator.get());
    std::mt19937_64 random(20260823);

    for (uint64_t step = 0; step < 800; ++step) {
        uint64_t operation = random() % 4;
        if (operation <= 1) {
            uint64_t offset =
                compatibility.Size() == 0 ? 0 : random() % (compatibility.Size() + 65);
            uint64_t size = random() % 129;
            std::vector<uint8_t> data(size);
            for (auto& value : data) {
                value = static_cast<uint8_t>(random());
            }
            compatibility.Write(data.data(), size, offset);
            canonical.WriteAt(offset, data.data(), size);
        } else if (operation == 2) {
            uint64_t size = random() % 4096;
            compatibility.Resize(size);
            canonical.Resize(size);
        } else {
            uint64_t size = compatibility.Size() == 0 ? 0 : random() % (compatibility.Size() + 1);
            compatibility.Shrink(size);
            canonical.Shrink(size);
        }
        RequireSameBufferContent(compatibility, canonical);
    }

    if (canonical.Size() < 256) {
        std::vector<uint8_t> padding(256 - canonical.Size(), 0xA5);
        compatibility.Write(padding.data(), padding.size(), compatibility.Size());
        canonical.WriteAt(canonical.Size(), padding.data(), padding.size());
    }
    std::array<uint8_t, 17> first{};
    std::array<uint8_t, 31> second{};
    std::array<ReadRequest, 2> requests{{
        {first.data(), 3, first.size()},
        {second.data(), 129, second.size()},
    }};
    REQUIRE(canonical.ReadMany(requests.data(), requests.size()));
    std::array<uint8_t, 17> expected_first{};
    std::array<uint8_t, 31> expected_second{};
    REQUIRE(compatibility.Read(expected_first.size(), 3, expected_first.data()));
    REQUIRE(compatibility.Read(expected_second.size(), 129, expected_second.data()));
    REQUIRE(first == expected_first);
    REQUIRE(second == expected_second);
}

TEST_CASE("BufferIO serialization is cross compatible", "[ut][BufferIO][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("buffer_io_canonical_compatibility");
    std::vector<uint8_t> data(8193);
    for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i * 43);
    }

    BufferIO compatibility_source(dir.GenerateRandomFile(false), allocator.get());
    BufferIO canonical_destination(dir.GenerateRandomFile(false), allocator.get());
    compatibility_source.Write(data.data(), data.size(), 0);
    std::stringstream old_stream;
    IOStreamWriter old_writer(old_stream);
    compatibility_source.Serialize(old_writer);
    IOStreamReader old_reader(old_stream);
    canonical_destination.Deserialize(old_reader);
    RequireBufferContent(canonical_destination, data);

    BufferIO canonical_source(dir.GenerateRandomFile(false), allocator.get());
    BufferIO compatibility_destination(dir.GenerateRandomFile(false), allocator.get());
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

TEST_CASE("BufferIO releases acquire buffer after a short read", "[ut][BufferIO]") {
    BufferCountingAllocator allocator;
    fixtures::TempDir dir("buffer_io_canonical_short_read");
    auto path = dir.GenerateRandomFile(false);
    BufferIO io(path, &allocator);
    std::array<uint8_t, 64> data{};
    io.WriteAt(0, data.data(), data.size());
    std::filesystem::resize_file(path, 8);

    allocator.Reset();
    REQUIRE_THROWS_AS(io.Acquire(0, data.size()), VsagException);
    REQUIRE(allocator.Allocations() == 1);
    REQUIRE(allocator.Deallocations() == 1);
}

TEST_CASE("BufferIO cache configuration preserves content and invalidation",
          "[ut][BufferIO][cache]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("buffer_io_canonical_cache");
    BufferIO compatibility(dir.GenerateRandomFile(false), allocator.get());
    BufferIO canonical(dir.GenerateRandomFile(false), allocator.get());
    auto parameter = std::make_shared<BufferIOParameter>();
    parameter->enable_read_cache_ = true;
    parameter->read_cache_total_size_ = 4 * Page::DEFAULT_PAGE_SIZE;
    compatibility.EnableReadCache(parameter);
    canonical.EnableReadCache(parameter);

    std::vector<uint8_t> data(3 * Page::DEFAULT_PAGE_SIZE + 117);
    for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i * 29);
    }
    compatibility.Write(data.data(), data.size(), 0);
    canonical.WriteAt(0, data.data(), data.size());
    RequireSameBufferContent(compatibility, canonical);

    std::vector<uint8_t> replacement(Page::DEFAULT_PAGE_SIZE + 31, 0xD3);
    uint64_t write_offset = Page::DEFAULT_PAGE_SIZE - 13;
    compatibility.Write(replacement.data(), replacement.size(), write_offset);
    canonical.WriteAt(write_offset, replacement.data(), replacement.size());
    RequireSameBufferContent(compatibility, canonical);

    canonical.InitIO(nullptr);
    RequireSameBufferContent(compatibility, canonical);
}

TEST_CASE("BufferIO shared cache namespaces remain isolated", "[ut][BufferIO][cache]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("buffer_io_canonical_shared_cache");
    BufferIO first(dir.GenerateRandomFile(false), allocator.get());
    BufferIO second(dir.GenerateRandomFile(false), allocator.get());
    auto cache = std::make_shared<LRUPageCache>(8);
    first.SetReadCache(cache, 100);
    second.SetReadCache(cache, 1000);

    std::vector<uint8_t> first_data(Page::DEFAULT_PAGE_SIZE + 3, 0x19);
    std::vector<uint8_t> second_data(Page::DEFAULT_PAGE_SIZE + 3, 0xE7);
    first.WriteAt(0, first_data.data(), first_data.size());
    second.WriteAt(0, second_data.data(), second_data.size());
    std::vector<uint8_t> first_read(first_data.size());
    std::vector<uint8_t> second_read(second_data.size());
    REQUIRE(first.ReadAt(0, first_read.size(), first_read.data()));
    REQUIRE(second.ReadAt(0, second_read.size(), second_read.data()));
    REQUIRE(first_read == first_data);
    REQUIRE(second_read == second_data);

    first.Resize(first.Size());
    std::fill(second_read.begin(), second_read.end(), 0);
    REQUIRE(second.ReadAt(0, second_read.size(), second_read.data()));
    REQUIRE(second_read == second_data);
}

}  // namespace vsag
