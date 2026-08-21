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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/buffer_io/buffer_io.h"
#include "io/buffer_io/buffer_io_parameter.h"
#include "io/common/basic_io_test.h"
#include "io/common/io_parameter.h"
#include "io/memory_io/memory_io_parameter.h"
#include "io/read_cache/page.h"
#include "io/reader_io/reader_io.h"
#include "io/reader_io/reader_io_parameter.h"
#include "unittest.h"

using namespace vsag;

namespace {

IOParamPtr
MakeReadCacheParam(uint64_t page_count = 4) {
    auto param = std::make_shared<BufferIOParameter>();
    param->enable_read_cache_ = true;
    param->read_cache_total_size_ = Page::DEFAULT_PAGE_SIZE * page_count;
    return param;
}

class TestReader : public Reader {
public:
    explicit TestReader(const std::vector<uint8_t>& data) : data_(data) {
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::memcpy(dest, data_.data() + offset, len);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        Read(offset, len, dest);
        callback(IOErrorCode::IO_SUCCESS, "success");
    }

    uint64_t
    Size() const override {
        return data_.size();
    }

private:
    const std::vector<uint8_t>& data_;
};

class CountingIO : public BasicIO<CountingIO> {
public:
    static constexpr bool InMemory = false;
    static constexpr bool SkipDeserialize = false;

    explicit CountingIO(Allocator* allocator) : BasicIO<CountingIO>(allocator) {
    }

    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        std::scoped_lock<std::mutex> lock(multi_read_mutex_);
        if (data_.size() < offset + size) {
            data_.resize(offset + size);
            size_ = offset + size;
        }
        std::memcpy(data_.data() + offset, data, size);
    }

    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        ++read_count;
        {
            std::unique_lock<std::mutex> lock(read_mutex_);
            if (block_read_) {
                read_blocked_ = true;
                read_blocked_cv_.notify_all();
                read_resume_cv_.wait(lock, [this] { return not block_read_; });
            }
        }
        std::scoped_lock<std::mutex> lock(multi_read_mutex_);
        std::memcpy(data, data_.data() + offset, size);
        return true;
    }

    const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        std::scoped_lock<std::mutex> lock(multi_read_mutex_);
        need_release = true;
        if (offset > data_.size() or size > data_.size() - offset) {
            return nullptr;
        }
        return data_.data() + offset;
    }

    void
    ReleaseImpl(const uint8_t*) const {
        ++release_count;
    }

    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        ++multi_read_count;
        {
            std::unique_lock<std::mutex> lock(multi_read_mutex_);
            if (block_multi_read_) {
                multi_read_blocked_ = true;
                multi_read_blocked_cv_.notify_all();
                multi_read_resume_cv_.wait(lock, [this] { return not block_multi_read_; });
            }
        }
        if (fail_next_multi_read.exchange(false)) {
            return false;
        }
        std::scoped_lock<std::mutex> lock(multi_read_mutex_);
        for (uint64_t i = 0; i < count; ++i) {
            std::memcpy(datas, data_.data() + offsets[i], sizes[i]);
            datas += sizes[i];
        }
        return true;
    }

    void
    BlockMultiRead() {
        std::scoped_lock<std::mutex> lock(multi_read_mutex_);
        block_multi_read_ = true;
        multi_read_blocked_ = false;
    }

    bool
    WaitForMultiReadBlock() const {
        std::unique_lock<std::mutex> lock(multi_read_mutex_);
        return multi_read_blocked_cv_.wait_for(
            lock, std::chrono::seconds(5), [this] { return multi_read_blocked_; });
    }

    void
    UnblockMultiRead() {
        {
            std::scoped_lock<std::mutex> lock(multi_read_mutex_);
            block_multi_read_ = false;
        }
        multi_read_resume_cv_.notify_all();
    }

    void
    BlockRead() {
        std::scoped_lock<std::mutex> lock(read_mutex_);
        block_read_ = true;
        read_blocked_ = false;
    }

    bool
    WaitForReadBlock() const {
        std::unique_lock<std::mutex> lock(read_mutex_);
        return read_blocked_cv_.wait_for(
            lock, std::chrono::seconds(5), [this] { return read_blocked_; });
    }

    void
    UnblockRead() {
        {
            std::scoped_lock<std::mutex> lock(read_mutex_);
            block_read_ = false;
        }
        read_resume_cv_.notify_all();
    }

    mutable std::atomic<uint64_t> read_count{0};
    mutable std::atomic<uint64_t> multi_read_count{0};
    mutable std::atomic<bool> fail_next_multi_read{false};

    mutable std::atomic<uint64_t> release_count{0};

private:
    std::vector<uint8_t> data_;
    mutable std::mutex multi_read_mutex_;
    mutable std::condition_variable multi_read_blocked_cv_;
    mutable std::condition_variable multi_read_resume_cv_;
    mutable bool block_multi_read_{false};
    mutable bool multi_read_blocked_{false};
    mutable std::mutex read_mutex_;
    mutable std::condition_variable read_blocked_cv_;
    mutable std::condition_variable read_resume_cv_;
    mutable bool block_read_{false};
    mutable bool read_blocked_{false};
};

}  // namespace

TEST_CASE("BasicIO cache component basic test", "[ReadCache][ut]") {
    fixtures::TempDir dir("read_cache_basic");
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    BufferIO io(dir.GenerateRandomFile(false), allocator.get());
    io.EnableReadCache(MakeReadCacheParam());
    TestBasicReadWrite(io);
}

TEST_CASE("BasicIO cache component file backend test", "[ReadCache][ut]") {
    fixtures::TempDir dir("read_cache_buffer");
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    BufferIO io(dir.GenerateRandomFile(false), allocator.get());
    io.EnableReadCache(MakeReadCacheParam());
    TestBasicReadWrite(io);
}

TEST_CASE("BasicIO cache component invalidates writes", "[ReadCache][ut]") {
    fixtures::TempDir dir("read_cache_invalidate");
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    BufferIO io(dir.GenerateRandomFile(false), allocator.get());
    io.EnableReadCache(MakeReadCacheParam());

    std::vector<uint8_t> old_data(128, 0x11);
    std::vector<uint8_t> new_data(64, 0x22);
    std::vector<uint8_t> read_buf(64);
    io.Write(old_data.data(), old_data.size(), 0);
    REQUIRE(io.Read(read_buf.size(), 32, read_buf.data()));

    io.Write(new_data.data(), new_data.size(), 32);
    REQUIRE(io.Read(read_buf.size(), 32, read_buf.data()));
    REQUIRE(std::memcmp(read_buf.data(), new_data.data(), read_buf.size()) == 0);
}

TEST_CASE("BasicIO cache component direct and multi read", "[ReadCache][ut]") {
    fixtures::TempDir dir("read_cache_multi_read");
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    BufferIO io(dir.GenerateRandomFile(false), allocator.get());
    io.EnableReadCache(MakeReadCacheParam());

    uint64_t offset = Page::DEFAULT_PAGE_SIZE - 30;
    std::vector<uint8_t> data(100, 0xCD);
    io.Write(data.data(), data.size(), offset);

    bool need_release = false;
    const auto* ptr = io.Read(data.size(), offset, need_release);
    REQUIRE(ptr != nullptr);
    REQUIRE(need_release);
    REQUIRE(std::memcmp(ptr, data.data(), data.size()) == 0);
    io.Release(ptr);

    uint64_t sizes[] = {30, 40};
    uint64_t offsets[] = {offset, offset + 30};
    std::vector<uint8_t> result(70);
    REQUIRE(io.MultiRead(result.data(), sizes, offsets, 2));
    REQUIRE(std::memcmp(result.data(), data.data(), result.size()) == 0);
}

TEST_CASE("BasicIO cache releases direct-read buffers by their allocation source",
          "[ReadCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CountingIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam());

    std::vector<uint8_t> source(128, 0x5A);
    io.Write(source.data(), source.size(), 0);

    bool need_release = false;
    const auto* cached = io.Read(source.size(), 0, need_release);
    REQUIRE(cached != nullptr);
    REQUIRE(need_release);
    REQUIRE(std::memcmp(cached, source.data(), source.size()) == 0);

    io.SetReadCache(nullptr);
    io.Release(cached);
    REQUIRE(io.release_count == 0);

    const auto* direct = io.Read(source.size(), 0, need_release);
    REQUIRE(direct != nullptr);
    REQUIRE(need_release);
    io.Release(direct);
    REQUIRE(io.release_count == 1);
}

TEST_CASE("BasicIO cache validates multi-read pointers", "[ReadCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CountingIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam());

    std::vector<uint8_t> source(Page::DEFAULT_PAGE_SIZE, 0x5A);
    io.Write(source.data(), source.size(), 0);

    uint64_t zero = 0;
    uint64_t one = 1;
    REQUIRE(io.MultiRead(nullptr, nullptr, nullptr, 0));
    REQUIRE_FALSE(io.MultiRead(nullptr, nullptr, &zero, 1));
    REQUIRE_FALSE(io.MultiRead(nullptr, &zero, nullptr, 1));
    REQUIRE(io.MultiRead(nullptr, &zero, &zero, 1));
    REQUIRE_FALSE(io.MultiRead(nullptr, &one, &zero, 1));
}

TEST_CASE("BasicIO cache batches missing pages", "[ReadCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CountingIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam(4));

    std::vector<uint8_t> source(Page::DEFAULT_PAGE_SIZE * 3);
    for (uint64_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>(i);
    }
    io.Write(source.data(), source.size(), 0);

    uint64_t sizes[] = {32, 64, 48};
    uint64_t offsets[] = {Page::DEFAULT_PAGE_SIZE - 16,
                          Page::DEFAULT_PAGE_SIZE + 20,
                          Page::DEFAULT_PAGE_SIZE * 2 + 40};
    std::vector<uint8_t> result(144);
    REQUIRE(io.MultiRead(result.data(), sizes, offsets, 3));
    REQUIRE(io.multi_read_count == 1);
    REQUIRE(io.read_count == 0);
    REQUIRE(std::memcmp(result.data(), source.data() + offsets[0], sizes[0]) == 0);
    REQUIRE(std::memcmp(result.data() + sizes[0], source.data() + offsets[1], sizes[1]) == 0);
    REQUIRE(std::memcmp(
                result.data() + sizes[0] + sizes[1], source.data() + offsets[2], sizes[2]) == 0);
}

TEST_CASE("BasicIO cache bounds sparse multi-read batches", "[ReadCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CountingIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam(4));

    constexpr uint64_t temp_budget = 8ULL * 1024ULL * 1024ULL;
    const uint64_t page_count = temp_budget / (2 * Page::DEFAULT_PAGE_SIZE) + 1;
    std::vector<uint8_t> source(page_count * Page::DEFAULT_PAGE_SIZE);
    std::vector<uint64_t> sizes(page_count, 1);
    std::vector<uint64_t> offsets(page_count);
    std::vector<uint8_t> result(page_count);
    for (uint64_t i = 0; i < page_count; ++i) {
        offsets[i] = i * Page::DEFAULT_PAGE_SIZE;
        source[offsets[i]] = static_cast<uint8_t>(i);
    }
    io.Write(source.data(), source.size(), 0);

    REQUIRE(io.MultiRead(result.data(), sizes.data(), offsets.data(), page_count));
    REQUIRE(io.multi_read_count == 2);
    for (uint64_t i = 0; i < page_count; ++i) {
        REQUIRE(result[i] == source[offsets[i]]);
    }
}

TEST_CASE("BasicIO cache uses single-read backend for direct reads", "[ReadCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CountingIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam());

    std::vector<uint8_t> source(Page::DEFAULT_PAGE_SIZE, 0xA5);
    std::vector<uint8_t> result(source.size());
    io.Write(source.data(), source.size(), 0);

    REQUIRE(io.Read(result.size(), 0, result.data()));
    REQUIRE(io.read_count == 1);
    REQUIRE(io.multi_read_count == 0);
    REQUIRE(result == source);
}

TEST_CASE("BasicIO cache shares in-flight page loads and retries failures", "[ReadCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CountingIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam());

    std::vector<uint8_t> source(Page::DEFAULT_PAGE_SIZE, 0xA5);
    io.Write(source.data(), source.size(), 0);
    io.BlockMultiRead();
    std::mutex start_mutex;
    std::condition_variable start_cv;
    uint64_t readers_ready = 0;
    bool start = false;
    std::vector<uint8_t> first(source.size());
    std::vector<uint8_t> second(source.size());
    std::atomic<bool> first_succeeded{false};
    std::atomic<bool> second_succeeded{false};
    auto read_page = [&](std::vector<uint8_t>& result, std::atomic<bool>& succeeded) {
        {
            std::unique_lock<std::mutex> lock(start_mutex);
            ++readers_ready;
            start_cv.notify_all();
            if (not start_cv.wait_for(lock, std::chrono::seconds(5), [&] { return start; })) {
                return;
            }
        }
        uint64_t size = result.size();
        uint64_t offset = 0;
        succeeded = io.MultiRead(result.data(), &size, &offset, 1);
    };
    std::thread first_reader(read_page, std::ref(first), std::ref(first_succeeded));
    std::thread second_reader(read_page, std::ref(second), std::ref(second_succeeded));
    bool readers_ready_in_time = false;
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        readers_ready_in_time =
            start_cv.wait_for(lock, std::chrono::seconds(5), [&] { return readers_ready == 2; });
        start = true;
    }
    start_cv.notify_all();
    const bool multi_read_blocked = io.WaitForMultiReadBlock();
    io.UnblockMultiRead();
    first_reader.join();
    second_reader.join();

    REQUIRE(readers_ready_in_time);
    REQUIRE(multi_read_blocked);
    REQUIRE(first_succeeded);
    REQUIRE(second_succeeded);
    REQUIRE(io.multi_read_count == 1);
    REQUIRE(first == source);
    REQUIRE(second == source);

    CountingIO failing_io(allocator.get());
    failing_io.EnableReadCache(MakeReadCacheParam());
    failing_io.Write(source.data(), source.size(), 0);
    failing_io.fail_next_multi_read = true;
    uint64_t size = source.size();
    uint64_t offset = 0;
    std::vector<uint8_t> result(source.size());
    REQUIRE_FALSE(failing_io.MultiRead(result.data(), &size, &offset, 1));
    REQUIRE(failing_io.MultiRead(result.data(), &size, &offset, 1));
    REQUIRE(failing_io.multi_read_count == 2);
    REQUIRE(result == source);
}

TEST_CASE("BasicIO cache keeps an in-flight multi-read snapshot", "[ReadCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CountingIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam());

    std::vector<uint8_t> source(Page::DEFAULT_PAGE_SIZE, 0xA5);
    std::vector<uint8_t> result(source.size());
    io.Write(source.data(), source.size(), 0);
    io.BlockMultiRead();

    std::atomic<bool> read_succeeded{false};
    std::thread reader([&] {
        uint64_t size = result.size();
        uint64_t offset = 0;
        read_succeeded = io.MultiRead(result.data(), &size, &offset, 1);
    });
    const bool multi_read_blocked = io.WaitForMultiReadBlock();
    io.SetReadCache(nullptr);
    io.UnblockMultiRead();
    reader.join();

    REQUIRE(multi_read_blocked);
    REQUIRE(read_succeeded);
    REQUIRE(result == source);
    REQUIRE(io.multi_read_count == 1);
}

TEST_CASE("BasicIO cache reloads invalidated in-flight pages", "[ReadCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CountingIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam());

    std::vector<uint8_t> source(Page::DEFAULT_PAGE_SIZE, 0xA5);
    std::vector<uint8_t> result(source.size());
    io.Write(source.data(), source.size(), 0);
    io.BlockMultiRead();

    std::atomic<bool> read_succeeded{false};
    std::thread reader([&] {
        uint64_t size = result.size();
        uint64_t offset = 0;
        read_succeeded = io.MultiRead(result.data(), &size, &offset, 1);
    });
    const bool multi_read_blocked = io.WaitForMultiReadBlock();
    source[0] = 0x5A;
    io.Write(source.data(), 1, 0);
    io.UnblockMultiRead();
    reader.join();

    REQUIRE(multi_read_blocked);
    REQUIRE(read_succeeded);
    REQUIRE(result == source);
    REQUIRE(io.multi_read_count == 2);
}

TEST_CASE("BasicIO cache reloads invalidated direct reads", "[ReadCache][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CountingIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam());

    std::vector<uint8_t> source(Page::DEFAULT_PAGE_SIZE, 0xA5);
    std::vector<uint8_t> result(source.size());
    io.Write(source.data(), source.size(), 0);
    io.BlockRead();

    std::atomic<bool> read_succeeded{false};
    std::thread reader([&] { read_succeeded = io.Read(result.size(), 0, result.data()); });
    const bool read_blocked = io.WaitForReadBlock();
    source[0] = 0x5A;
    io.Write(source.data(), 1, 0);
    io.UnblockRead();
    reader.join();

    REQUIRE(read_blocked);
    REQUIRE(read_succeeded);
    REQUIRE(result == source);
    REQUIRE(io.read_count == 2);
}

TEST_CASE("BasicIO cache component initializes ReaderIO", "[ReadCache][ut]") {
    std::vector<uint8_t> data(1024);
    for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i);
    }

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    ReaderIO io(allocator.get());
    io.EnableReadCache(MakeReadCacheParam());
    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = std::make_shared<TestReader>(data);
    io.InitIO(reader_param);

    std::vector<uint8_t> read_buf(data.size());
    REQUIRE(io.Read(read_buf.size(), 0, read_buf.data()));
    REQUIRE(read_buf == data);
}

TEST_CASE("ReadCache configuration test", "[ReadCache][ut]") {
    JsonType json;
    json["type"].SetString("memory_io");
    json["enable_read_cache"].SetBool(true);
    json["total_cache_size"].SetUint64(4096);

    auto param = IOParameter::GetIOParameterByJson(json);
    REQUIRE(param != nullptr);
    REQUIRE(param->enable_read_cache_);
    REQUIRE(param->read_cache_total_size_ == 4096);
    REQUIRE(param->ToJson()["enable_read_cache"].GetBool());
    REQUIRE(param->ToJson()["total_cache_size"].GetUint64() == 4096);
}
