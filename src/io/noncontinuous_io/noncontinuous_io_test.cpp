
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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/async_io/async_io.h"
#include "io/buffer_io/buffer_io.h"
#include "io/common/io_contract_test.h"
#include "io/memory_io/memory_io.h"
#include "io/mmap_io/mmap_io.h"
#include "unittest.h"
namespace vsag {

class ToggleFailAllocator : public Allocator {
public:
    std::string
    Name() override {
        return "ToggleFailAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        return fail_allocations_ ? nullptr : std::malloc(size);
    }

    void
    Deallocate(void* data) override {
        std::free(data);
    }

    void*
    Reallocate(void* data, uint64_t size) override {
        return fail_allocations_ ? nullptr : std::realloc(data, size);
    }

    void
    SetFailAllocations(bool fail) {
        fail_allocations_ = fail;
    }

private:
    bool fail_allocations_{false};
};

struct TrackingIOState {
    std::vector<uint8_t> data_;
    uint64_t multi_read_calls_{0};
    std::vector<uint64_t> last_sizes_;
    std::vector<uint64_t> last_offsets_;
};

class TrackingIO : public MemoryIO {
public:
    TrackingIO(std::shared_ptr<TrackingIOState> state, Allocator* allocator)
        : MemoryIO(allocator), state_(std::move(state)) {
    }

    void
    Write(const uint8_t* data, uint64_t size, uint64_t offset) {
        state_->data_.resize(std::max<uint64_t>(state_->data_.size(), offset + size));
        std::memcpy(state_->data_.data() + offset, data, size);
        MemoryIO::Write(data, size, offset);
    }

    bool
    MultiRead(uint8_t* datas,
              const uint64_t* sizes,
              const uint64_t* offsets,
              uint64_t count) const {
        ++state_->multi_read_calls_;
        state_->last_sizes_.assign(sizes, sizes + count);
        state_->last_offsets_.assign(offsets, offsets + count);
        for (uint64_t i = 0; i < count; ++i) {
            if (offsets[i] > state_->data_.size() or sizes[i] > state_->data_.size() - offsets[i]) {
                return false;
            }
            std::memcpy(datas, state_->data_.data() + offsets[i], sizes[i]);
            datas += sizes[i];
        }
        return true;
    }

    bool
    ReadMany(const ReadRequest* requests, uint64_t count) const {
        ++state_->multi_read_calls_;
        state_->last_sizes_.clear();
        state_->last_offsets_.clear();
        state_->last_sizes_.reserve(count);
        state_->last_offsets_.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            state_->last_sizes_.emplace_back(requests[i].size);
            state_->last_offsets_.emplace_back(requests[i].offset);
            if (requests[i].offset > state_->data_.size() or
                requests[i].size > state_->data_.size() - requests[i].offset) {
                return false;
            }
            std::memcpy(requests[i].destination,
                        state_->data_.data() + requests[i].offset,
                        requests[i].size);
        }
        return true;
    }

    bool
    ReadManyPrevalidated(const ReadRequest* requests, uint64_t count) const {
        return ReadMany(requests, count);
    }

private:
    std::shared_ptr<TrackingIOState> state_;
};

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
    fixtures::TempDir temp_dir("NonContinuousIOTestSerialize");
    {
        NonContinuousIOTest<T> test;
        auto non_continuous_allocator1 = std::make_unique<NonContinuousAllocator>(allocator.get());
        auto io1 =
            test.CreateNonContinuousIO(non_continuous_allocator1.get(),
                                       allocator.get(),
                                       (std::filesystem::path(temp_dir.path) / "io1").string(),
                                       allocator.get());
        auto non_continuous_allocator2 = std::make_unique<NonContinuousAllocator>(allocator.get());
        auto io2 =
            test.CreateNonContinuousIO(non_continuous_allocator2.get(),
                                       allocator.get(),
                                       (std::filesystem::path(temp_dir.path) / "io2").string(),
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

TEST_CASE("NonContinuousIO batches physical fragments", "[NonContinuousIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto non_continuous_allocator = std::make_unique<NonContinuousAllocator>(allocator.get());
    auto state = std::make_shared<TrackingIOState>();
    NonContinuousIOTest<TrackingIO> test;
    std::unique_ptr<NonContinuousIO<TrackingIO>> io(test.CreateNonContinuousIO(
        non_continuous_allocator.get(), allocator.get(), state, allocator.get()));
    std::unique_ptr<NonContinuousIO<TrackingIO>> spacer(test.CreateNonContinuousIO(
        non_continuous_allocator.get(), allocator.get(), state, allocator.get()));

    constexpr uint64_t page_size = 4096;
    std::vector<uint8_t> logical_data(page_size * 3);
    for (uint64_t i = 0; i < logical_data.size(); ++i) {
        logical_data[i] = static_cast<uint8_t>(i % 251);
    }
    std::vector<uint8_t> spacer_data(page_size, 0xFF);
    io->Write(logical_data.data(), page_size, 0);
    spacer->Write(spacer_data.data(), page_size, 0);
    io->Write(logical_data.data() + page_size, page_size, page_size);
    spacer->Write(spacer_data.data(), page_size, page_size);
    io->Write(logical_data.data() + page_size * 2, page_size, page_size * 2);

    std::vector<uint64_t> sizes{12, 0, 5, 12, 8};
    std::vector<uint64_t> offsets{
        page_size * 2 - 4, page_size * 3, 7, page_size * 2 - 4, page_size - 2};
    std::vector<uint8_t> output(37);
    REQUIRE(io->MultiRead(output.data(), sizes.data(), offsets.data(), sizes.size()));

    std::vector<uint8_t> expected;
    expected.reserve(output.size());
    for (uint64_t i = 0; i < sizes.size(); ++i) {
        expected.insert(expected.end(),
                        logical_data.begin() + offsets[i],
                        logical_data.begin() + offsets[i] + sizes[i]);
    }
    REQUIRE(output == expected);
    REQUIRE(state->multi_read_calls_ == 1);
    REQUIRE(state->last_sizes_ == std::vector<uint64_t>{4, 8, 5, 4, 8, 2, 6});
    REQUIRE(state->last_offsets_ == std::vector<uint64_t>{page_size * 3 - 4,
                                                          page_size * 4,
                                                          7,
                                                          page_size * 3 - 4,
                                                          page_size * 4,
                                                          page_size - 2,
                                                          page_size * 2});
}

TEST_CASE("NonContinuousIO spills large physical fragment batches", "[NonContinuousIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto non_continuous_allocator = std::make_unique<NonContinuousAllocator>(allocator.get());
    auto state = std::make_shared<TrackingIOState>();
    NonContinuousIOTest<TrackingIO> test;
    std::unique_ptr<NonContinuousIO<TrackingIO>> io(test.CreateNonContinuousIO(
        non_continuous_allocator.get(), allocator.get(), state, allocator.get()));
    std::unique_ptr<NonContinuousIO<TrackingIO>> spacer(test.CreateNonContinuousIO(
        non_continuous_allocator.get(), allocator.get(), state, allocator.get()));

    constexpr uint64_t page_size = 4096;
    constexpr uint64_t request_count = 80;
    std::vector<uint8_t> logical_data(page_size * (request_count + 1));
    for (uint64_t i = 0; i < logical_data.size(); ++i) {
        logical_data[i] = static_cast<uint8_t>(i % 251);
    }
    std::vector<uint8_t> spacer_data(page_size, 0xFF);
    for (uint64_t page_id = 0; page_id <= request_count; ++page_id) {
        io->Write(logical_data.data() + page_id * page_size, page_size, page_id * page_size);
        spacer->Write(spacer_data.data(), page_size, page_id * page_size);
    }

    std::vector<uint64_t> sizes(request_count, 128);
    std::vector<uint64_t> offsets(request_count);
    std::vector<uint8_t> output(request_count * 128);
    std::vector<uint8_t> expected;
    expected.reserve(output.size());
    for (uint64_t i = 0; i < request_count; ++i) {
        offsets[i] = (i + 1) * page_size - 64;
        expected.insert(expected.end(),
                        logical_data.begin() + offsets[i],
                        logical_data.begin() + offsets[i] + sizes[i]);
    }

    REQUIRE(io->MultiRead(output.data(), sizes.data(), offsets.data(), request_count));
    REQUIRE(output == expected);
    REQUIRE(state->multi_read_calls_ == 1);
    REQUIRE(state->last_sizes_.size() == request_count * 2);
}

TEST_CASE("NonContinuousIO validates batch ranges and empty reads", "[NonContinuousIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto non_continuous_allocator = std::make_unique<NonContinuousAllocator>(allocator.get());
    auto state = std::make_shared<TrackingIOState>();
    NonContinuousIOTest<TrackingIO> test;
    std::unique_ptr<NonContinuousIO<TrackingIO>> io(test.CreateNonContinuousIO(
        non_continuous_allocator.get(), allocator.get(), state, allocator.get()));

    constexpr uint64_t page_size = 4096;
    std::vector<uint8_t> data(page_size, 0x5A);
    io->Write(data.data(), data.size(), 0);

    REQUIRE(io->MultiRead(nullptr, nullptr, nullptr, 0));
    REQUIRE(state->multi_read_calls_ == 0);

    std::vector<uint64_t> empty_sizes{0, 0, 0};
    std::vector<uint64_t> empty_offsets{0, page_size / 2, page_size};
    REQUIRE(io->MultiRead(nullptr, empty_sizes.data(), empty_offsets.data(), empty_sizes.size()));
    REQUIRE(state->multi_read_calls_ == 0);

    std::vector<uint64_t> invalid_sizes{4, 1};
    std::vector<uint64_t> invalid_offsets{0, page_size};
    std::vector<uint8_t> output(5, 0xA5);
    REQUIRE_FALSE(io->MultiRead(
        output.data(), invalid_sizes.data(), invalid_offsets.data(), invalid_sizes.size()));
    REQUIRE(output == std::vector<uint8_t>(5, 0xA5));
    REQUIRE(state->multi_read_calls_ == 0);

    uint64_t overflow_size = 16;
    uint64_t overflow_offset = std::numeric_limits<uint64_t>::max() - 7;
    REQUIRE_FALSE(io->MultiRead(&output[0], &overflow_size, &overflow_offset, 1));
    REQUIRE(state->multi_read_calls_ == 0);

    uint64_t zero_size = 0;
    uint64_t past_end_offset = page_size + 1;
    REQUIRE_FALSE(io->MultiRead(nullptr, &zero_size, &past_end_offset, 1));
    REQUIRE_FALSE(io->MultiRead(nullptr, &overflow_size, &zero_size, 1));
    REQUIRE_FALSE(io->MultiRead(output.data(), nullptr, &zero_size, 1));
    REQUIRE_FALSE(io->MultiRead(output.data(), &zero_size, nullptr, 1));
    REQUIRE(state->multi_read_calls_ == 0);
}

TEST_CASE("NonContinuousIO canonical acquire reports allocation failure",
          "[NonContinuousIO][ut][exception-safety]") {
    ToggleFailAllocator allocator;
    NonContinuousAllocator extent_allocator(&allocator);
    NonContinuousIO<MemoryIO> io(&extent_allocator, &allocator, &allocator);
    std::vector<uint8_t> source(4096, 0x5A);
    io.WriteAt(0, source.data(), source.size());

    allocator.SetFailAllocations(true);
    REQUIRE_THROWS_AS(io.Acquire(0, 128), VsagException);

    bool need_release = true;
    REQUIRE(io.Read(128, 0, need_release) == nullptr);
    REQUIRE_FALSE(need_release);
    allocator.SetFailAllocations(false);
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
