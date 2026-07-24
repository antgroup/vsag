
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

#include "io/noncontinuous_io/noncontinuous_io.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/async_io/async_io.h"
#include "io/buffer_io/buffer_io.h"
#include "io/common/basic_io_test.h"
#include "io/mmap_io/mmap_io.h"
#include "unittest.h"
namespace vsag {
template <typename IOTmpl>
class NonContinuousIOTest {
public:
    NonContinuousIOTest() = default;
    ~NonContinuousIOTest() = default;

    template <typename... Args>
    NonContinuousIO<IOTmpl>*
    CreateNonContinuousIO(NonContinuousAllocator* non_continuous_allocator,
                          Allocator* allocator,
                          Args&&... args) {
        return new NonContinuousIO<IOTmpl>(
            non_continuous_allocator, allocator, std::forward<Args>(args)...);
    }
};
}  // namespace vsag

using namespace vsag;
template <typename T>
void
NonContinuousIOTestBasic() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    {
        NonContinuousIOTest<T> test;
        auto non_continuous_allocator = std::make_unique<NonContinuousAllocator>(allocator.get());
        auto io = test.CreateNonContinuousIO(non_continuous_allocator.get(),
                                             allocator.get(),
                                             "/tmp/test_noncontinuous_io",
                                             allocator.get());
        TestBasicReadWrite(*io);
        delete io;
    }
}

TEST_CASE("NonContinuousIO Basic Test", "[NonContinuousIO][ut]") {
    NonContinuousIOTestBasic<MMapIO>();
    NonContinuousIOTestBasic<BufferIO>();
    NonContinuousIOTestBasic<AsyncIO>();
}

template <typename T>
void
NonContinuousIOTestSerialize() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    {
        NonContinuousIOTest<T> test;
        auto non_continuous_allocator1 = std::make_unique<NonContinuousAllocator>(allocator.get());
        auto io1 = test.CreateNonContinuousIO(non_continuous_allocator1.get(),
                                              allocator.get(),
                                              "/tmp/test_noncontinuous_io1",
                                              allocator.get());
        auto non_continuous_allocator2 = std::make_unique<NonContinuousAllocator>(allocator.get());
        auto io2 = test.CreateNonContinuousIO(non_continuous_allocator2.get(),
                                              allocator.get(),
                                              "/tmp/test_noncontinuous_io2",
                                              allocator.get());
        TestSerializeAndDeserialize(*io1, *io2);
        delete io1;
        delete io2;
    }
}

TEST_CASE("NonContinuousIO Serialize Test", "[NonContinuousIO][ut]") {
    NonContinuousIOTestSerialize<MMapIO>();
    NonContinuousIOTestSerialize<BufferIO>();
    NonContinuousIOTestSerialize<AsyncIO>();
}

TEST_CASE("NonContinuousAllocator allocates unique regions concurrently",
          "[NonContinuousIO][ut][concurrent]") {
    constexpr uint64_t thread_count = 16;
    constexpr uint64_t allocations_per_thread = 4096;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    NonContinuousAllocator non_continuous_allocator(allocator.get());
    std::atomic<uint64_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::vector<NonContinuousArea>> thread_areas(thread_count);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (uint64_t thread_id = 0; thread_id < thread_count; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            auto& areas = thread_areas[thread_id];
            areas.reserve(allocations_per_thread);
            ready.fetch_add(1, std::memory_order_release);
            while (not start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint64_t i = 0; i < allocations_per_thread; ++i) {
                areas.emplace_back(non_continuous_allocator.Require(1));
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != thread_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
        thread.join();
    }

    std::vector<NonContinuousArea> areas;
    areas.reserve(thread_count * allocations_per_thread);
    for (const auto& thread_area : thread_areas) {
        areas.insert(areas.end(), thread_area.begin(), thread_area.end());
    }
    std::sort(areas.begin(), areas.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.offset < rhs.offset;
    });
    for (uint64_t i = 1; i < areas.size(); ++i) {
        REQUIRE(areas[i - 1].offset + areas[i - 1].size <= areas[i].offset);
    }
}
