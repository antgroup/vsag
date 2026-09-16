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
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/async_io/async_io.h"
#include "io/read_cache/lru_page_cache.h"
#include "storage/serialization.h"
#include "unittest.h"

namespace vsag {
namespace {

std::vector<uint8_t>
MakeAsyncData(uint64_t size) {
    std::vector<uint8_t> data(size);
    for (uint64_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(i * 47 + 11);
    }
    return data;
}

void
RequireAsyncContent(const AsyncIO& io, const std::vector<uint8_t>& expected) {
    REQUIRE(io.Size() == expected.size());
    std::vector<uint8_t> actual(expected.size());
    REQUIRE(io.ReadAt(0, actual.size(), actual.data()));
    REQUIRE(actual == expected);
}

}  // namespace

TEST_CASE("AsyncIO direct single and sliced scatter batch", "[ut][AsyncIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("async_io_canonical_contract");
    AsyncIO io(dir.GenerateRandomFile(false), allocator.get());
    auto data = MakeAsyncData(512 * 1024 + 37);
    io.WriteAt(0, data.data(), data.size());

    std::array<uint8_t, 777> output{};
    REQUIRE(io.ReadAt(123, output.size(), output.data()));
    REQUIRE(std::memcmp(output.data(), data.data() + 123, output.size()) == 0);

    auto lease = io.Acquire(509, 1027);
    REQUIRE(lease);
    REQUIRE(lease.Size() == 1027);
    REQUIRE(std::memcmp(lease.Data(), data.data() + 509, lease.Size()) == 0);

    bool need_release = false;
    const uint8_t* compatibility_data = io.Read(1027, 509, need_release);
    REQUIRE(compatibility_data != nullptr);
    REQUIRE(need_release);
    REQUIRE(std::memcmp(compatibility_data, data.data() + 509, 1027) == 0);
    io.Release(compatibility_data);

    constexpr uint64_t request_count = 237;
    std::mt19937_64 random(20260824);
    std::vector<std::vector<uint8_t>> destinations(request_count);
    std::vector<ReadRequest> requests(request_count);
    for (uint64_t i = 0; i < request_count; ++i) {
        uint64_t size = i % 29 == 0 ? 0 : random() % 1024 + 1;
        uint64_t offset = random() % (data.size() - size + 1);
        destinations[i].resize(size);
        requests[i] = ReadRequest{size == 0 ? nullptr : destinations[i].data(), offset, size};
    }
    REQUIRE(io.ReadMany(requests.data(), requests.size()));
    for (uint64_t i = 0; i < request_count; ++i) {
        if (requests[i].size > 0) {
            REQUIRE(std::memcmp(destinations[i].data(),
                                data.data() + requests[i].offset,
                                requests[i].size) == 0);
        }
    }
}

TEST_CASE("AsyncIO randomized read differential and compatibility",
          "[ut][AsyncIO][differential][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("async_io_canonical_differential");
    AsyncIO compatibility(dir.GenerateRandomFile(false), allocator.get());
    AsyncIO canonical(dir.GenerateRandomFile(false), allocator.get());
    auto data = MakeAsyncData(2 * 1024 * 1024 + 91);
    compatibility.Write(data.data(), data.size(), 0);
    canonical.WriteAt(0, data.data(), data.size());
    std::mt19937_64 random(20260825);
    for (uint64_t i = 0; i < 1000; ++i) {
        uint64_t size = random() % 4096 + 1;
        uint64_t offset = random() % (data.size() - size + 1);
        std::vector<uint8_t> compatibility_data(size);
        std::vector<uint8_t> canonical_data(size);
        REQUIRE(compatibility.Read(size, offset, compatibility_data.data()));
        REQUIRE(canonical.ReadAt(offset, size, canonical_data.data()));
        REQUIRE(compatibility_data == canonical_data);
    }

    std::stringstream old_stream;
    IOStreamWriter old_writer(old_stream);
    compatibility.Serialize(old_writer);
    AsyncIO canonical_destination(dir.GenerateRandomFile(false), allocator.get());
    IOStreamReader old_reader(old_stream);
    canonical_destination.Deserialize(old_reader);
    RequireAsyncContent(canonical_destination, data);

    std::stringstream new_stream;
    IOStreamWriter new_writer(new_stream);
    canonical.Serialize(new_writer);
    AsyncIO compatibility_destination(dir.GenerateRandomFile(false), allocator.get());
    IOStreamReader new_reader(new_stream);
    compatibility_destination.Deserialize(new_reader);
    std::vector<uint8_t> actual(data.size());
    REQUIRE(compatibility_destination.Read(actual.size(), 0, actual.data()));
    REQUIRE(actual == data);
}

TEST_CASE("AsyncIO normalizes filesystem inspection errors", "[ut][AsyncIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("async_io_filesystem_error");
    const auto regular_file = dir.GenerateRandomFile();
    const auto invalid_path = regular_file + "/child";

    REQUIRE_THROWS_AS(AsyncIO(invalid_path, allocator.get()), VsagException);
    REQUIRE_FALSE(std::filesystem::exists(invalid_path));
}

#if HAVE_LIBAIO
TEST_CASE("AsyncIO shared environment is concurrent", "[ut][AsyncIO][concurrency]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IOContextPool pool(2, allocator.get());
    IOEnvironment environment{allocator.get(), &pool};
    fixtures::TempDir dir("async_io_canonical_concurrency");
    AsyncIO io(dir.GenerateRandomFile(false), environment);
    auto data = MakeAsyncData(1024 * 1024);
    io.WriteAt(0, data.data(), data.size());

    std::atomic<bool> success{true};
    std::vector<std::thread> threads;
    for (uint64_t thread_id = 0; thread_id < 8; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            std::mt19937_64 random(20260826 + thread_id);
            for (uint64_t iteration = 0; iteration < 100; ++iteration) {
                std::array<std::array<uint8_t, 257>, 16> destinations{};
                std::array<ReadRequest, 16> requests{};
                for (uint64_t i = 0; i < requests.size(); ++i) {
                    uint64_t offset = random() % (data.size() - destinations[i].size() + 1);
                    requests[i] = {destinations[i].data(), offset, destinations[i].size()};
                }
                if (not io.ReadMany(requests.data(), requests.size())) {
                    success.store(false, std::memory_order_relaxed);
                    return;
                }
                for (uint64_t i = 0; i < requests.size(); ++i) {
                    if (std::memcmp(destinations[i].data(),
                                    data.data() + requests[i].offset,
                                    requests[i].size) != 0) {
                        success.store(false, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    REQUIRE(success.load(std::memory_order_relaxed));
}
#endif

TEST_CASE("AsyncIO detects direct short reads", "[ut][AsyncIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("async_io_canonical_short_read");
    auto path = dir.GenerateRandomFile(false);
    AsyncIO io(path, allocator.get());
    auto data = MakeAsyncData(4096);
    io.WriteAt(0, data.data(), data.size());
    std::filesystem::resize_file(path, 64);
    REQUIRE_THROWS_AS(io.Acquire(0, data.size()), VsagException);
}

TEST_CASE("AsyncIO optional cache preserves native and cached ownership", "[ut][AsyncIO][cache]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("async_io_canonical_cache");
    AsyncIO io(dir.GenerateRandomFile(false), allocator.get());
    auto data = MakeAsyncData(2 * Page::DEFAULT_PAGE_SIZE + 79);
    io.WriteAt(0, data.data(), data.size());

    {
        auto native_lease = io.Acquire(37, 911);
        REQUIRE(native_lease);
        REQUIRE(std::memcmp(native_lease.Data(), data.data() + 37, 911) == 0);
    }

    io.SetReadCache(std::make_shared<LRUPageCache>(4));
    {
        auto cached_lease = io.Acquire(Page::DEFAULT_PAGE_SIZE - 19, 211);
        REQUIRE(cached_lease);
        REQUIRE(std::memcmp(cached_lease.Data(), data.data() + Page::DEFAULT_PAGE_SIZE - 19, 211) ==
                0);
    }
    std::array<uint8_t, 33> first{};
    std::array<uint8_t, 47> second{};
    std::array<ReadRequest, 2> requests{{
        {first.data(), 17, first.size()},
        {second.data(), Page::DEFAULT_PAGE_SIZE + 31, second.size()},
    }};
    auto operation = io.SubmitReads(requests.data(), requests.size());
    REQUIRE(operation.Poll());
    REQUIRE(operation.Wait());
    REQUIRE(std::memcmp(first.data(), data.data() + 17, first.size()) == 0);
    REQUIRE(std::memcmp(second.data(), data.data() + Page::DEFAULT_PAGE_SIZE + 31, second.size()) ==
            0);

    uint8_t replacement = 0xA7;
    io.WriteAt(17, &replacement, 1);
    uint8_t actual = 0;
    REQUIRE(io.ReadAt(17, 1, &actual));
    REQUIRE(actual == replacement);
}

}  // namespace vsag
