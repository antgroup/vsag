
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

#include "lock_strategy.h"

#include <mutex>
#include <thread>
#include <unordered_map>

#include "impl/allocator/default_allocator.h"
#include "unittest.h"

using namespace vsag;

namespace {
class CountingAllocator : public DefaultAllocator {
public:
    void*
    Allocate(uint64_t size) override {
        auto* ptr = DefaultAllocator::Allocate(size);
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = size;
        live_bytes_ += size;
        ++allocation_count_;
        return ptr;
    }

    void
    Deallocate(void* p) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            live_bytes_ -= allocations_.at(p);
            allocations_.erase(p);
            ++deallocation_count_;
        }
        DefaultAllocator::Deallocate(p);
    }

    uint64_t
    LiveBytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return live_bytes_;
    }

    uint64_t allocation_count_{0};
    uint64_t deallocation_count_{0};

private:
    mutable std::mutex mutex_;
    std::unordered_map<void*, uint64_t> allocations_;
    uint64_t live_bytes_{0};
};
}  // namespace

TEST_CASE("LockStrategy Basic", "[ut][LockStrategy]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("points mutex lock and shared lock") {
        PointsMutex mutex_array(8, allocator.get());
        mutex_array.Lock(0);
        mutex_array.Unlock(0);
        mutex_array.SharedLock(0);
        mutex_array.SharedUnlock(0);
        REQUIRE(mutex_array.GetMemoryUsage() > 0);
    }

    SECTION("points mutex initializes empty and resizes") {
        PointsMutex mutex_array(0, allocator.get());
        REQUIRE(mutex_array.GetMemoryUsage() == 0);

        mutex_array.Resize(1);
        const auto resized_memory = mutex_array.GetMemoryUsage();
        REQUIRE(resized_memory > 0);

        mutex_array.Resize(0);
        REQUIRE(mutex_array.GetMemoryUsage() < resized_memory);
    }

    SECTION("resize updates memory usage") {
        const auto block_size = PointsMutex::kMutexesPerBlock;
        PointsMutex mutex_array(block_size + 1, allocator.get());
        auto before = mutex_array.GetMemoryUsage();
        mutex_array.Resize(block_size * 2 + 1);
        auto after = mutex_array.GetMemoryUsage();
        REQUIRE(after > before);
        mutex_array.Resize(2);
        REQUIRE(mutex_array.GetMemoryUsage() < after);
    }

    SECTION("locks across block boundaries") {
        const auto block_size = PointsMutex::kMutexesPerBlock;
        PointsMutex mutex_array(block_size + 1, allocator.get());
        mutex_array.Lock(block_size - 1);
        mutex_array.Unlock(block_size - 1);
        mutex_array.SharedLock(block_size);
        mutex_array.SharedUnlock(block_size);
    }

    SECTION("block allocation reduces allocator overhead") {
        CountingAllocator counting_allocator;
        const auto element_count = PointsMutex::kMutexesPerBlock * 2;
        {
            PointsMutex mutex_array(element_count, &counting_allocator);
            REQUIRE(counting_allocator.allocation_count_ == 3);
            REQUIRE(counting_allocator.LiveBytes() == mutex_array.GetMemoryUsage());
            REQUIRE(mutex_array.GetMemoryUsage() <
                    element_count *
                        (sizeof(std::shared_ptr<std::shared_mutex>) + sizeof(std::shared_mutex)));
        }
        REQUIRE(counting_allocator.LiveBytes() == 0);
        REQUIRE(counting_allocator.deallocation_count_ == 3);
    }

    SECTION("lock guards protect concurrent increment") {
        auto mutex_impl = std::make_shared<PointsMutex>(1, allocator.get());
        int counter = 0;
        constexpr int thread_num = 8;
        constexpr int loops = 1000;
        std::vector<std::thread> threads;
        threads.reserve(thread_num);
        for (int i = 0; i < thread_num; ++i) {
            threads.emplace_back([&]() {
                for (int j = 0; j < loops; ++j) {
                    LockGuard guard(mutex_impl, 0);
                    ++counter;
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        REQUIRE(counter == thread_num * loops);
    }

    SECTION("empty mutex no-op") {
        EmptyMutex mutex_array;
        mutex_array.Lock(0);
        mutex_array.Unlock(0);
        mutex_array.SharedLock(0);
        mutex_array.SharedUnlock(0);
        mutex_array.Resize(100);
        REQUIRE(mutex_array.GetMemoryUsage() == 0);
    }
}
