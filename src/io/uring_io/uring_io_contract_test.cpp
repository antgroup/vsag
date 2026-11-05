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
#include <type_traits>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/read_cache/lru_page_cache.h"
#include "io/uring_io/uring_io.h"
#include "storage/serialization.h"
#include "unittest.h"

namespace vsag {
namespace {

std::vector<uint8_t>
MakeUringData(uint64_t size) {
    std::vector<uint8_t> data(size);
    for (uint64_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(i * 61 + 23);
    }
    return data;
}

void
RequireUringContent(const UringIO& io, const std::vector<uint8_t>& expected) {
    REQUIRE(io.Size() == expected.size());
    std::vector<uint8_t> actual(expected.size());
    REQUIRE(io.ReadAt(0, actual.size(), actual.data()));
    REQUIRE(actual == expected);
}

void
ExerciseUringProfile(const std::string& path, Allocator* allocator, bool direct_read) {
    UringIO io(path, allocator, direct_read);
    auto data = MakeUringData(4 * 1024 * 1024 + 137);
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

    constexpr uint64_t request_count = 1037;
    std::mt19937_64 random(20260827 + static_cast<uint64_t>(direct_read));
    std::vector<std::vector<uint8_t>> destinations(request_count);
    std::vector<ReadRequest> requests(request_count);
    std::vector<uint64_t> sizes(request_count);
    std::vector<uint64_t> offsets(request_count);
    uint64_t contiguous_size = 0;
    for (uint64_t i = 0; i < request_count; ++i) {
        uint64_t size = i % 43 == 0 ? 0 : random() % 1024 + 1;
        uint64_t offset = random() % (data.size() - size + 1);
        destinations[i].resize(size);
        requests[i] = {size == 0 ? nullptr : destinations[i].data(), offset, size};
        sizes[i] = size;
        offsets[i] = offset;
        contiguous_size += size;
    }

    auto operation = io.SubmitReads(requests.data(), requests.size());
#if HAVE_LIBURING
    static_assert(not std::is_copy_constructible_v<decltype(operation)>);
    static_assert(std::is_move_constructible_v<decltype(operation)>);
#endif
    REQUIRE(operation.Poll());
    REQUIRE(operation.Wait());
    for (uint64_t i = 0; i < request_count; ++i) {
        if (requests[i].size > 0) {
            REQUIRE(std::memcmp(destinations[i].data(),
                                data.data() + requests[i].offset,
                                requests[i].size) == 0);
        }
    }

    std::vector<uint8_t> contiguous(contiguous_size);
    REQUIRE(io.MultiRead(contiguous.data(), sizes.data(), offsets.data(), request_count));
    uint64_t cursor = 0;
    for (uint64_t i = 0; i < request_count; ++i) {
        REQUIRE(std::memcmp(contiguous.data() + cursor, data.data() + offsets[i], sizes[i]) == 0);
        cursor += sizes[i];
    }
}

}  // namespace

TEST_CASE("UringIO buffered and direct contracts", "[ut][UringIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("uring_io_canonical_contract");
    ExerciseUringProfile(dir.GenerateRandomFile(false), allocator.get(), false);
    ExerciseUringProfile(dir.GenerateRandomFile(false), allocator.get(), true);
}

TEST_CASE("UringIO randomized differential and compatibility",
          "[ut][UringIO][differential][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("uring_io_canonical_differential");
#if HAVE_LIBURING
    UringIO compatibility(dir.GenerateRandomFile(false), allocator.get(), false);
#else
    UringIO compatibility(dir.GenerateRandomFile(false), allocator.get());
#endif
    UringIO canonical(dir.GenerateRandomFile(false), allocator.get(), false);
    auto data = MakeUringData(2 * 1024 * 1024 + 91);
    compatibility.Write(data.data(), data.size(), 0);
    canonical.WriteAt(0, data.data(), data.size());
    std::mt19937_64 random(20260828);
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
    UringIO canonical_destination(dir.GenerateRandomFile(false), allocator.get());
    IOStreamReader old_reader(old_stream);
    canonical_destination.Deserialize(old_reader);
    RequireUringContent(canonical_destination, data);

    std::stringstream new_stream;
    IOStreamWriter new_writer(new_stream);
    canonical.Serialize(new_writer);
#if HAVE_LIBURING
    UringIO compatibility_destination(dir.GenerateRandomFile(false), allocator.get(), false);
#else
    UringIO compatibility_destination(dir.GenerateRandomFile(false), allocator.get());
#endif
    IOStreamReader new_reader(new_stream);
    compatibility_destination.Deserialize(new_reader);
    std::vector<uint8_t> actual(data.size());
    REQUIRE(compatibility_destination.Read(actual.size(), 0, actual.data()));
    REQUIRE(actual == data);
}

TEST_CASE("UringIO normalizes filesystem inspection errors", "[ut][UringIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("uring_io_filesystem_error");
    const auto regular_file = dir.GenerateRandomFile();
    const auto invalid_path = regular_file + "/child";

    REQUIRE_THROWS_AS(UringIO(invalid_path, allocator.get()), VsagException);
    REQUIRE_FALSE(std::filesystem::exists(invalid_path));
}

#if !HAVE_LIBURING
TEST_CASE("UringIO preserves BufferIO compatibility reads without liburing",
          "[ut][UringIO][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("uring_io_no_liburing_compatibility");
    UringIO io(dir.GenerateRandomFile(false), allocator.get());
    std::array<uint8_t, 4> data{1, 2, 3, 4};
    io.Write(data.data(), data.size(), 0);

    std::array<uint8_t, 5> destination{};
    REQUIRE_THROWS_AS(io.Read(destination.size(), 0, destination.data()), VsagException);
    REQUIRE_FALSE(io.ReadAt(0, destination.size(), destination.data()));

    std::array<uint64_t, 1> sizes{destination.size()};
    std::array<uint64_t, 1> offsets{0};
    REQUIRE_THROWS_AS(io.MultiRead(destination.data(), sizes.data(), offsets.data(), sizes.size()),
                      VsagException);
    std::array<ReadRequest, 1> requests{{{destination.data(), 0, destination.size()}}};
    REQUIRE_FALSE(io.ReadMany(requests.data(), requests.size()));
}
#endif

#if HAVE_LIBURING
TEST_CASE("UringIO shared environment is concurrent", "[ut][UringIO][concurrency]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    UringIOContextPool pool(2, allocator.get());
    IOEnvironment environment = MakeDefaultIOEnvironment(allocator.get());
    environment.uring_context_pool = &pool;
    fixtures::TempDir dir("uring_io_canonical_concurrency");
    UringIO io(dir.GenerateRandomFile(false), environment);
    auto data = MakeUringData(1024 * 1024);
    io.WriteAt(0, data.data(), data.size());

    std::atomic<bool> success{true};
    std::vector<std::thread> threads;
    for (uint64_t thread_id = 0; thread_id < 8; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            std::mt19937_64 random(20260829 + thread_id);
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

TEST_CASE("UringIO detects short reads", "[ut][UringIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("uring_io_canonical_short_read");
    auto path = dir.GenerateRandomFile(false);
    UringIO io(path, allocator.get(), true);
    auto data = MakeUringData(4096);
    io.WriteAt(0, data.data(), data.size());
    std::filesystem::resize_file(path, 64);
    REQUIRE_THROWS_AS(io.Acquire(0, data.size()), VsagException);
}

TEST_CASE("UringIO optional cache adapts leases and operations", "[ut][UringIO][cache]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("uring_io_canonical_cache");
    UringIO io(dir.GenerateRandomFile(false), allocator.get(), true);
    auto data = MakeUringData(2 * Page::DEFAULT_PAGE_SIZE + 101);
    io.WriteAt(0, data.data(), data.size());

    {
        auto native_lease = io.Acquire(43, 997);
        REQUIRE(native_lease);
        REQUIRE(std::memcmp(native_lease.Data(), data.data() + 43, 997) == 0);
    }

    io.SetReadCache(std::make_shared<LRUPageCache>(4));
    {
        auto cached_lease = io.Acquire(Page::DEFAULT_PAGE_SIZE - 23, 257);
        REQUIRE(cached_lease);
        REQUIRE(std::memcmp(cached_lease.Data(), data.data() + Page::DEFAULT_PAGE_SIZE - 23, 257) ==
                0);
    }
    std::array<uint8_t, 29> first{};
    std::array<uint8_t, 53> second{};
    std::array<ReadRequest, 2> requests{{
        {first.data(), 19, first.size()},
        {second.data(), Page::DEFAULT_PAGE_SIZE + 37, second.size()},
    }};
    auto operation = io.SubmitReads(requests.data(), requests.size());
    REQUIRE(operation.Poll());
    REQUIRE(operation.Wait());
    REQUIRE(std::memcmp(first.data(), data.data() + 19, first.size()) == 0);
    REQUIRE(std::memcmp(second.data(), data.data() + Page::DEFAULT_PAGE_SIZE + 37, second.size()) ==
            0);

    uint8_t replacement = 0x5D;
    io.WriteAt(19, &replacement, 1);
    uint8_t actual = 0;
    REQUIRE(io.ReadAt(19, 1, &actual));
    REQUIRE(actual == replacement);
}

}  // namespace vsag
