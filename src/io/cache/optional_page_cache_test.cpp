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

#include "io/cache/optional_page_cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "io/core/byte_io.h"
#include "unittest.h"

namespace vsag {
namespace {

class CacheTestAllocator : public Allocator {
public:
    std::string
    Name() override {
        return "CacheTestAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        allocations_.fetch_add(1, std::memory_order_relaxed);
        if (fail_next_.exchange(false, std::memory_order_relaxed)) {
            return nullptr;
        }
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

    [[nodiscard]] uint64_t
    Allocations() const {
        return allocations_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t
    Deallocations() const {
        return deallocations_.load(std::memory_order_relaxed);
    }

    void
    FailNextAllocation() {
        fail_next_.store(true, std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> allocations_{0};
    std::atomic<uint64_t> deallocations_{0};
    std::atomic<bool> fail_next_{false};
};

struct CountingBackendState {
    explicit CountingBackendState(uint64_t size, uint8_t seed = 0) : data(size) {
        for (uint64_t i = 0; i < size; ++i) {
            data[i] = static_cast<uint8_t>(i * 37 + seed);
        }
    }

    std::vector<uint8_t> data;
    std::mutex data_mutex;
    std::atomic<uint64_t> read_at_calls{0};
    std::atomic<uint64_t> read_many_calls{0};
    std::atomic<uint64_t> read_many_requests{0};
    std::atomic<bool> fail_next_read{false};

    std::atomic<bool> block_reads{false};
    std::atomic<bool> read_is_entered{false};
    std::atomic<bool> release_read{false};
};

class CountingBackend {
public:
    struct Capabilities {
        static constexpr bool InMemory = false;
        static constexpr bool RequiresInitialization = false;
        static constexpr bool CanBindSerializedRange = false;
        static constexpr bool LegacyBatchRangeThrows = false;
        static constexpr bool LegacyUncheckedReadable = false;
        static constexpr bool AsyncReadable = true;
    };

    using Lease = AllocatorLease;
    using Operation = ImmediateOperation;

    CountingBackend(std::shared_ptr<CountingBackendState> state, Allocator* allocator)
        : state_(std::move(state)), allocator_(allocator) {
    }

    [[nodiscard]] uint64_t
    InitialLogicalSize() const {
        std::scoped_lock<std::mutex> lock(state_->data_mutex);
        return state_->data.size();
    }

    [[nodiscard]] bool
    ReadAt(uint64_t offset, uint64_t size, uint8_t* destination) const {
        state_->read_at_calls.fetch_add(1, std::memory_order_relaxed);
        BlockReadIfRequested();
        if (state_->fail_next_read.exchange(false, std::memory_order_relaxed)) {
            return false;
        }
        return Copy(offset, size, destination);
    }

    [[nodiscard]] bool
    ReadMany(const ReadRequest* requests, uint64_t count) const {
        state_->read_many_calls.fetch_add(1, std::memory_order_relaxed);
        state_->read_many_requests.fetch_add(count, std::memory_order_relaxed);
        BlockReadIfRequested();
        if (state_->fail_next_read.exchange(false, std::memory_order_relaxed)) {
            return false;
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (not Copy(requests[i].offset, requests[i].size, requests[i].destination)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool
    ReadManyContiguous(uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        std::vector<ReadRequest> requests;
        requests.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            requests.emplace_back(ReadRequest{destination, offsets[i], sizes[i]});
            destination += sizes[i];
        }
        return ReadMany(requests.data(), requests.size());
    }

    [[nodiscard]] ImmediateOperation
    SubmitReads(const ReadRequest* requests, uint64_t count) const {
        return ImmediateOperation(ReadMany(requests, count));
    }

    [[nodiscard]] AllocatorLease
    Acquire(uint64_t offset, uint64_t size) const {
        auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        AllocatorOwner owner(allocator_, data);
        if (data == nullptr or not ReadAt(offset, size, data)) {
            return {};
        }
        return AllocatorLease(data, size, std::move(owner));
    }

    [[nodiscard]] const uint8_t*
    LegacyRead(uint64_t offset, uint64_t size, bool& need_release) const {
        auto lease = Acquire(offset, size);
        if (not lease) {
            need_release = false;
            return nullptr;
        }
        auto* data = static_cast<uint8_t*>(allocator_->Allocate(size));
        if (data == nullptr) {
            need_release = false;
            return nullptr;
        }
        std::memcpy(data, lease.Data(), size);
        need_release = true;
        return data;
    }

    void
    Release(const uint8_t* data) const {
        allocator_->Deallocate(const_cast<uint8_t*>(data));
    }

    void
    WriteAt(uint64_t offset, const uint8_t* source, uint64_t size) {
        std::scoped_lock<std::mutex> lock(state_->data_mutex);
        if (offset + size > state_->data.size()) {
            state_->data.resize(offset + size);
        }
        std::memcpy(state_->data.data() + offset, source, size);
    }

    void
    ResizePhysical(uint64_t size) {
        std::scoped_lock<std::mutex> lock(state_->data_mutex);
        state_->data.resize(size);
    }

    void
    ShrinkPhysical(uint64_t size) {
        ResizePhysical(size);
    }

    void
    Prefetch(uint64_t, uint64_t) {
    }

    [[nodiscard]] int64_t
    MemoryUsage(uint64_t logical_size) const {
        return logical_size;
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return nullptr;
    }

    [[nodiscard]] Allocator*
    AllocatorPtr() const {
        return allocator_;
    }

private:
    void
    BlockReadIfRequested() const {
        if (state_->block_reads.load(std::memory_order_acquire)) {
            state_->read_is_entered.store(true, std::memory_order_release);
            while (not state_->release_read.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
    }

    [[nodiscard]] bool
    Copy(uint64_t offset, uint64_t size, uint8_t* destination) const {
        std::scoped_lock<std::mutex> lock(state_->data_mutex);
        if (offset > state_->data.size() or size > state_->data.size() - offset) {
            return false;
        }
        std::memcpy(destination, state_->data.data() + offset, size);
        return true;
    }

    std::shared_ptr<CountingBackendState> state_;
    Allocator* allocator_{nullptr};
};

class SequentialCountingBackend final : public CountingBackend {
public:
    struct Capabilities : CountingBackend::Capabilities {
        static constexpr bool AsyncReadable = false;
    };

    using CountingBackend::CountingBackend;
};

using CachedTestIO = ByteIO<CountingBackend, OptionalPageCache>;
using SequentialCachedTestIO = ByteIO<SequentialCountingBackend, OptionalPageCache>;

void
ReleaseBlockedRead(const std::shared_ptr<CountingBackendState>& state) {
    state->release_read.store(true, std::memory_order_release);
}

void
WaitForBlockedRead(const std::shared_ptr<CountingBackendState>& state) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (not state->read_is_entered.load(std::memory_order_acquire) and
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    REQUIRE(state->read_is_entered.load(std::memory_order_acquire));
}

}  // namespace

TEST_CASE("OptionalPageCache disabled path is an exact backend passthrough",
          "[ut][OptionalPageCache]") {
    CacheTestAllocator allocator;
    auto state = std::make_shared<CountingBackendState>(2 * Page::DEFAULT_PAGE_SIZE);
    CachedTestIO io(state, &allocator);
    uint8_t value = 0;
    REQUIRE(io.ReadAt(17, 1, &value));
    REQUIRE(value == state->data[17]);
    REQUIRE(state->read_at_calls == 1);
    REQUIRE(state->read_many_calls == 0);

    ReadRequest request{&value, 31, 1};
    REQUIRE(io.ReadMany(&request, 1));
    REQUIRE(value == state->data[31]);
    REQUIRE(state->read_at_calls == 1);
    REQUIRE(state->read_many_calls == 1);
    REQUIRE(state->read_many_requests == 1);
}

TEST_CASE("OptionalPageCache batches unique misses and serves subsequent hits",
          "[ut][OptionalPageCache]") {
    CacheTestAllocator allocator;
    constexpr uint64_t PAGE_COUNT = 37;
    auto state = std::make_shared<CountingBackendState>(PAGE_COUNT * Page::DEFAULT_PAGE_SIZE);
    CachedTestIO io(state, &allocator);
    io.SetReadCache(std::make_shared<LRUPageCache>(PAGE_COUNT));

    std::vector<uint8_t> values(100);
    std::vector<ReadRequest> requests;
    requests.reserve(values.size());
    for (uint64_t i = 0; i < values.size(); ++i) {
        uint64_t offset = (i % PAGE_COUNT) * Page::DEFAULT_PAGE_SIZE + i % 251;
        requests.emplace_back(ReadRequest{values.data() + i, offset, 1});
    }
    REQUIRE(io.ReadMany(requests.data(), requests.size()));
    REQUIRE(state->read_at_calls == 0);
    REQUIRE(state->read_many_calls == 1);
    REQUIRE(state->read_many_requests == PAGE_COUNT);
    for (uint64_t i = 0; i < values.size(); ++i) {
        REQUIRE(values[i] == state->data[requests[i].offset]);
    }

    std::fill(values.begin(), values.end(), 0);
    REQUIRE(io.ReadMany(requests.data(), requests.size()));
    REQUIRE(state->read_many_calls == 1);
    REQUIRE(state->read_many_requests == PAGE_COUNT);
}

TEST_CASE("OptionalPageCache deduplicates batch misses for sequential backends",
          "[ut][OptionalPageCache]") {
    CacheTestAllocator allocator;
    constexpr uint64_t UNIQUE_PAGE_COUNT = 8;
    constexpr uint64_t REQUEST_COUNT = 32;

    SECTION("ReadMany") {
        auto state =
            std::make_shared<CountingBackendState>(UNIQUE_PAGE_COUNT * Page::DEFAULT_PAGE_SIZE);
        SequentialCachedTestIO io(state, &allocator);
        io.SetReadCache(std::make_shared<LRUPageCache>(UNIQUE_PAGE_COUNT));

        std::array<uint8_t, REQUEST_COUNT> values{};
        std::array<ReadRequest, REQUEST_COUNT> requests{};
        for (uint64_t i = 0; i < REQUEST_COUNT; ++i) {
            const uint64_t offset = (i % UNIQUE_PAGE_COUNT) * Page::DEFAULT_PAGE_SIZE + i % 251;
            requests[i] = ReadRequest{values.data() + i, offset, 1};
        }

        REQUIRE(io.ReadMany(requests.data(), requests.size()));
        REQUIRE(state->read_at_calls == 0);
        REQUIRE(state->read_many_calls == 1);
        REQUIRE(state->read_many_requests == UNIQUE_PAGE_COUNT);
        for (uint64_t i = 0; i < REQUEST_COUNT; ++i) {
            REQUIRE(values[i] == state->data[requests[i].offset]);
        }

        std::fill(values.begin(), values.end(), 0);
        REQUIRE(io.ReadMany(requests.data(), requests.size()));
        REQUIRE(state->read_many_calls == 1);
        REQUIRE(state->read_many_requests == UNIQUE_PAGE_COUNT);
    }

    SECTION("MultiRead") {
        auto state =
            std::make_shared<CountingBackendState>(UNIQUE_PAGE_COUNT * Page::DEFAULT_PAGE_SIZE);
        SequentialCachedTestIO io(state, &allocator);
        io.SetReadCache(std::make_shared<LRUPageCache>(UNIQUE_PAGE_COUNT));

        std::array<uint8_t, REQUEST_COUNT> values{};
        std::array<uint64_t, REQUEST_COUNT> sizes{};
        std::array<uint64_t, REQUEST_COUNT> offsets{};
        sizes.fill(1);
        for (uint64_t i = 0; i < REQUEST_COUNT; ++i) {
            offsets[i] = (i % UNIQUE_PAGE_COUNT) * Page::DEFAULT_PAGE_SIZE + i % 251;
        }

        REQUIRE(io.MultiRead(values.data(), sizes.data(), offsets.data(), offsets.size()));
        REQUIRE(state->read_at_calls == 0);
        REQUIRE(state->read_many_calls == 1);
        REQUIRE(state->read_many_requests == UNIQUE_PAGE_COUNT);
        for (uint64_t i = 0; i < REQUEST_COUNT; ++i) {
            REQUIRE(values[i] == state->data[offsets[i]]);
        }

        std::fill(values.begin(), values.end(), 0);
        REQUIRE(io.MultiRead(values.data(), sizes.data(), offsets.data(), offsets.size()));
        REQUIRE(state->read_many_calls == 1);
        REQUIRE(state->read_many_requests == UNIQUE_PAGE_COUNT);
    }
}

TEST_CASE("OptionalPageCache coalesces concurrent same-page misses",
          "[ut][OptionalPageCache][concurrency]") {
    for (uint64_t thread_count : {8ULL, 32ULL}) {
        CacheTestAllocator allocator;
        auto state = std::make_shared<CountingBackendState>(Page::DEFAULT_PAGE_SIZE);
        state->block_reads = true;
        CachedTestIO io(state, &allocator);
        io.SetReadCache(std::make_shared<LRUPageCache>(4), thread_count * 10);

        std::atomic<uint64_t> ready{0};
        std::atomic<bool> start{false};
        std::vector<uint8_t> values(thread_count);
        std::vector<uint8_t> results(thread_count);
        std::vector<std::thread> threads;
        for (uint64_t i = 0; i < thread_count; ++i) {
            threads.emplace_back([&, i]() {
                ready.fetch_add(1, std::memory_order_relaxed);
                while (not start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                results[i] = io.ReadAt(123, 1, values.data() + i);
            });
        }
        while (ready.load(std::memory_order_relaxed) != thread_count) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        WaitForBlockedRead(state);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ReleaseBlockedRead(state);
        for (auto& thread : threads) {
            thread.join();
        }
        REQUIRE(state->read_at_calls == 1);
        REQUIRE(state->read_many_calls == 0);
        for (uint64_t i = 0; i < thread_count; ++i) {
            REQUIRE(results[i]);
            REQUIRE(values[i] == state->data[123]);
        }
    }
}

TEST_CASE("OptionalPageCache propagates a failed load and permits retry",
          "[ut][OptionalPageCache][concurrency]") {
    CacheTestAllocator allocator;
    auto state = std::make_shared<CountingBackendState>(Page::DEFAULT_PAGE_SIZE);
    state->block_reads = true;
    state->fail_next_read = true;
    CachedTestIO io(state, &allocator);
    io.SetReadCache(std::make_shared<LRUPageCache>(4));

    constexpr uint64_t THREAD_COUNT = 8;
    std::vector<uint8_t> results(THREAD_COUNT);
    std::vector<uint8_t> values(THREAD_COUNT);
    std::vector<std::thread> threads;
    for (uint64_t i = 0; i < THREAD_COUNT; ++i) {
        threads.emplace_back([&, i]() { results[i] = io.ReadAt(7, 1, values.data() + i); });
    }
    WaitForBlockedRead(state);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ReleaseBlockedRead(state);
    for (auto& thread : threads) {
        thread.join();
    }
    REQUIRE(state->read_at_calls == 1);
    for (uint8_t result : results) {
        REQUIRE_FALSE(result);
    }

    uint8_t value = 0;
    REQUIRE(io.ReadAt(7, 1, &value));
    REQUIRE(value == state->data[7]);
    REQUIRE(state->read_at_calls == 2);
}

TEST_CASE("OptionalPageCache reserves reverse-order batches without deadlock",
          "[ut][OptionalPageCache][concurrency]") {
    CacheTestAllocator allocator;
    auto state = std::make_shared<CountingBackendState>(2 * Page::DEFAULT_PAGE_SIZE);
    state->block_reads = true;
    CachedTestIO io(state, &allocator);
    io.SetReadCache(std::make_shared<LRUPageCache>(4));

    std::array<uint8_t, 2> first_values{};
    std::array<uint8_t, 2> second_values{};
    std::array<ReadRequest, 2> first_requests{{
        {first_values.data(), 3, 1},
        {first_values.data() + 1, Page::DEFAULT_PAGE_SIZE + 5, 1},
    }};
    std::array<ReadRequest, 2> second_requests{{
        {second_values.data(), Page::DEFAULT_PAGE_SIZE + 5, 1},
        {second_values.data() + 1, 3, 1},
    }};
    bool first_result = false;
    bool second_result = false;
    std::thread first_thread(
        [&]() { first_result = io.ReadMany(first_requests.data(), first_requests.size()); });
    WaitForBlockedRead(state);
    std::thread second_thread(
        [&]() { second_result = io.ReadMany(second_requests.data(), second_requests.size()); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ReleaseBlockedRead(state);
    first_thread.join();
    second_thread.join();

    REQUIRE(first_result);
    REQUIRE(second_result);
    REQUIRE(state->read_many_calls == 1);
    REQUIRE(state->read_many_requests == 2);
    REQUIRE(first_values[0] == state->data[3]);
    REQUIRE(first_values[1] == state->data[Page::DEFAULT_PAGE_SIZE + 5]);
    REQUIRE(second_values[0] == first_values[1]);
    REQUIRE(second_values[1] == first_values[0]);
}

TEST_CASE("OptionalPageCache retries an in-flight single-page load invalidated by a write",
          "[ut][OptionalPageCache][concurrency]") {
    CacheTestAllocator allocator;
    auto state = std::make_shared<CountingBackendState>(Page::DEFAULT_PAGE_SIZE);
    state->block_reads = true;
    CachedTestIO io(state, &allocator);
    io.SetReadCache(std::make_shared<LRUPageCache>(4));

    uint8_t value = 0;
    bool result = false;
    std::thread reader([&]() { result = io.ReadAt(17, 1, &value); });
    WaitForBlockedRead(state);
    uint8_t replacement = static_cast<uint8_t>(state->data[17] + 1);
    io.WriteAt(17, &replacement, 1);
    ReleaseBlockedRead(state);
    reader.join();

    REQUIRE(result);
    REQUIRE(value == replacement);
    REQUIRE(state->read_at_calls == 2);
}

TEST_CASE("OptionalPageCache retries only stale pages from an in-flight batch",
          "[ut][OptionalPageCache][concurrency]") {
    CacheTestAllocator allocator;
    auto state = std::make_shared<CountingBackendState>(2 * Page::DEFAULT_PAGE_SIZE);
    state->block_reads = true;
    CachedTestIO io(state, &allocator);
    io.SetReadCache(std::make_shared<LRUPageCache>(4));

    std::array<uint8_t, 2> values{};
    std::array<ReadRequest, 2> requests{{
        {values.data(), 17, 1},
        {values.data() + 1, Page::DEFAULT_PAGE_SIZE + 23, 1},
    }};
    bool result = false;
    std::thread reader([&]() { result = io.ReadMany(requests.data(), requests.size()); });
    WaitForBlockedRead(state);
    uint8_t replacement = static_cast<uint8_t>(state->data[17] + 1);
    io.WriteAt(17, &replacement, 1);
    ReleaseBlockedRead(state);
    reader.join();

    REQUIRE(result);
    REQUIRE(values[0] == replacement);
    REQUIRE(values[1] == state->data[Page::DEFAULT_PAGE_SIZE + 23]);
    REQUIRE(state->read_many_calls == 2);
    REQUIRE(state->read_many_requests == 3);
}

TEST_CASE("OptionalPageCache allocation failure leaves a retryable miss",
          "[ut][OptionalPageCache]") {
    CacheTestAllocator allocator;
    auto state = std::make_shared<CountingBackendState>(Page::DEFAULT_PAGE_SIZE);
    CachedTestIO io(state, &allocator);
    io.SetReadCache(std::make_shared<LRUPageCache>(2));
    allocator.FailNextAllocation();
    uint8_t value = 0;
    REQUIRE_FALSE(io.ReadAt(11, 1, &value));
    REQUIRE(state->read_at_calls == 0);
    REQUIRE(io.ReadAt(11, 1, &value));
    REQUIRE(value == state->data[11]);
    REQUIRE(state->read_at_calls == 1);
}

TEST_CASE("OptionalPageCache isolates shared namespaces and invalidates exact ranges",
          "[ut][OptionalPageCache]") {
    CacheTestAllocator allocator;
    auto first_state = std::make_shared<CountingBackendState>(2 * Page::DEFAULT_PAGE_SIZE, 3);
    auto second_state = std::make_shared<CountingBackendState>(2 * Page::DEFAULT_PAGE_SIZE, 91);
    auto cache = std::make_shared<LRUPageCache>(8);
    CachedTestIO first(first_state, &allocator);
    CachedTestIO second(second_state, &allocator);
    first.SetReadCache(cache, 100);
    second.SetReadCache(cache, 1000);

    uint8_t first_value = 0;
    uint8_t second_value = 0;
    REQUIRE(first.ReadAt(9, 1, &first_value));
    REQUIRE(second.ReadAt(9, 1, &second_value));
    REQUIRE(first_value != second_value);
    REQUIRE(cache->Size() == 2);

    first.Resize(first.Size());
    REQUIRE(cache->Size() == 1);
    REQUIRE(second.ReadAt(9, 1, &second_value));
    REQUIRE(second_state->read_at_calls == 1);
    REQUIRE(first.ReadAt(9, 1, &first_value));
    REQUIRE(first_state->read_at_calls == 2);

    std::vector<uint8_t> initial(2);
    std::vector<ReadRequest> requests{{initial.data(), 17, 1},
                                      {initial.data() + 1, Page::DEFAULT_PAGE_SIZE + 17, 1}};
    REQUIRE(first.ReadMany(requests.data(), requests.size()));
    uint8_t replacement = static_cast<uint8_t>(initial[0] + 1);
    first.WriteAt(17, &replacement, 1);
    std::vector<uint8_t> after_write(2);
    requests[0].destination = after_write.data();
    requests[1].destination = after_write.data() + 1;
    uint64_t calls_before = first_state->read_many_calls.load(std::memory_order_relaxed);
    uint64_t requests_before = first_state->read_many_requests.load(std::memory_order_relaxed);
    REQUIRE(first.ReadMany(requests.data(), requests.size()));
    REQUIRE(after_write[0] == replacement);
    REQUIRE(after_write[1] == initial[1]);
    REQUIRE(first_state->read_many_calls == calls_before + 1);
    REQUIRE(first_state->read_many_requests == requests_before + 1);
}

TEST_CASE("OptionalPageCache preserves acquire and legacy release ownership",
          "[ut][OptionalPageCache]") {
    CacheTestAllocator allocator;
    {
        auto state = std::make_shared<CountingBackendState>(Page::DEFAULT_PAGE_SIZE);
        CachedTestIO io(state, &allocator);
        io.SetReadCache(std::make_shared<LRUPageCache>(2));
        {
            auto lease = io.Acquire(41, 73);
            REQUIRE(lease);
            REQUIRE(lease.Size() == 73);
            REQUIRE(std::memcmp(lease.Data(), state->data.data() + 41, 73) == 0);
        }
        bool need_release = false;
        const uint8_t* data = io.Read(89, 57, need_release);
        REQUIRE(data != nullptr);
        REQUIRE(need_release);
        REQUIRE(std::memcmp(data, state->data.data() + 57, 89) == 0);
        io.Release(data);
    }
    REQUIRE(allocator.Allocations() == allocator.Deallocations());
}

TEST_CASE("Read leases clear moved-from views", "[ut][OptionalPageCache]") {
    std::array<uint8_t, 4> data{1, 2, 3, 4};
    BorrowedLease borrowed(data.data(), data.size(), BorrowedOwner{});
    BorrowedLease moved(std::move(borrowed));
    REQUIRE_FALSE(borrowed);
    REQUIRE(borrowed.Size() == 0);
    REQUIRE(moved.Data() == data.data());

    CacheTestAllocator allocator;
    auto state = std::make_shared<CountingBackendState>(Page::DEFAULT_PAGE_SIZE);
    CachedTestIO io(state, &allocator);
    auto cached = io.Acquire(41, 73);
    decltype(cached) cached_moved(std::move(cached));
    REQUIRE_FALSE(cached);
    REQUIRE(cached.Size() == 0);
    REQUIRE(cached_moved);
    decltype(cached) cached_assigned;
    cached_assigned = std::move(cached_moved);
    REQUIRE_FALSE(cached_moved);
    REQUIRE(cached_moved.Size() == 0);
    REQUIRE(cached_assigned.Size() == 73);
}

}  // namespace vsag
