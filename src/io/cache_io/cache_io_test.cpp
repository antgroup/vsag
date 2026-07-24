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

#include "io/cache_io/cache_io.h"

#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/buffer_io/buffer_io.h"
#include "io/cache_io/cache_io_parameter.h"
#include "io/common/basic_io_test.h"
#include "io/memory_io/memory_io.h"
#include "io/mmap_io/mmap_io.h"
#include "io/reader_io/reader_io.h"
#include "io/reader_io/reader_io_parameter.h"
#include "unittest.h"

using namespace vsag;

template <typename InnerIO>
void
TestCacheIOFileBackend(const std::string& path) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CacheIOParameter param;
    param.total_cache_size_ = Page::DEFAULT_PAGE_SIZE * 4;

    CacheIO<InnerIO> io(param, allocator.get(), path, allocator.get());
    TestBasicReadWrite(io);

    std::vector<uint8_t> data(300);
    for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i % 251);
    }
    io.Write(data.data(), data.size(), 32);

    std::vector<uint8_t> read_buf(data.size());
    REQUIRE(io.Read(data.size(), 32, read_buf.data()));
    REQUIRE(read_buf == data);
}

TEST_CASE("CacheIO Basic Test", "[CacheIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CacheIOParameter param;
    param.total_cache_size_ = Page::DEFAULT_PAGE_SIZE * 8;

    CacheIO<MemoryIO> io(param, allocator.get(), allocator.get());
    TestBasicReadWrite(io);
}

TEST_CASE("CacheIO BufferIO Backend Test", "[CacheIO][ut]") {
    fixtures::TempDir dir("cache_io_buffer");
    TestCacheIOFileBackend<BufferIO>(dir.GenerateRandomFile(false));
}

TEST_CASE("CacheIO MMapIO Backend Test", "[CacheIO][ut]") {
    fixtures::TempDir dir("cache_io_mmap");
    TestCacheIOFileBackend<MMapIO>(dir.GenerateRandomFile(false));
}

TEST_CASE("CacheIO Eviction Test", "[CacheIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CacheIOParameter param;
    param.total_cache_size_ = Page::DEFAULT_PAGE_SIZE * 2;

    CacheIO<MemoryIO> io(param, allocator.get(), allocator.get());

    std::vector<uint8_t> data(64, 0xAA);
    io.Write(data.data(), 64, 0);
    std::vector<uint8_t> data2(64, 0xBB);
    io.Write(data2.data(), 64, Page::DEFAULT_PAGE_SIZE);
    std::vector<uint8_t> data3(64, 0xCC);
    io.Write(data3.data(), 64, Page::DEFAULT_PAGE_SIZE * 2);
    std::vector<uint8_t> data4(64, 0xDD);
    io.Write(data4.data(), 64, Page::DEFAULT_PAGE_SIZE * 3);

    std::vector<uint8_t> read_buf(64);
    REQUIRE(io.Read(64, Page::DEFAULT_PAGE_SIZE * 3, read_buf.data()));
    REQUIRE(memcmp(read_buf.data(), data4.data(), 64) == 0);

    REQUIRE(io.Read(64, Page::DEFAULT_PAGE_SIZE * 2, read_buf.data()));
    REQUIRE(memcmp(read_buf.data(), data3.data(), 64) == 0);

    REQUIRE(io.Read(64, 0, read_buf.data()));
    REQUIRE(memcmp(read_buf.data(), data.data(), 64) == 0);

    REQUIRE(io.Read(64, Page::DEFAULT_PAGE_SIZE, read_buf.data()));
    REQUIRE(memcmp(read_buf.data(), data2.data(), 64) == 0);
}

TEST_CASE("CacheIO DirectRead Single Page Test", "[CacheIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CacheIOParameter param;
    param.total_cache_size_ = Page::DEFAULT_PAGE_SIZE * 4;

    CacheIO<MemoryIO> io(param, allocator.get(), allocator.get());

    std::vector<uint8_t> data(100, 0xAB);
    io.Write(data.data(), 100, 50);

    bool need_release = false;
    const auto* ptr = io.Read(100, 50, need_release);
    REQUIRE(ptr != nullptr);
    REQUIRE(need_release == true);
    REQUIRE(memcmp(ptr, data.data(), 100) == 0);
    io.Release(ptr);
}

TEST_CASE("CacheIO DirectRead Cross Page Test", "[CacheIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CacheIOParameter param;
    param.total_cache_size_ = Page::DEFAULT_PAGE_SIZE * 4;

    CacheIO<MemoryIO> io(param, allocator.get(), allocator.get());

    uint64_t offset = Page::DEFAULT_PAGE_SIZE - 30;
    std::vector<uint8_t> data(100, 0xCD);
    io.Write(data.data(), data.size(), offset);

    bool need_release = false;
    const auto* ptr = io.Read(data.size(), offset, need_release);
    REQUIRE(ptr != nullptr);
    REQUIRE(need_release == true);
    REQUIRE(memcmp(ptr, data.data(), data.size()) == 0);
    io.Release(ptr);
}

TEST_CASE("CacheIO Write Invalidate Test", "[CacheIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CacheIOParameter param;
    param.total_cache_size_ = Page::DEFAULT_PAGE_SIZE * 4;

    CacheIO<MemoryIO> io(param, allocator.get(), allocator.get());

    std::vector<uint8_t> old_data(128, 0x11);
    std::vector<uint8_t> new_data(64, 0x22);
    std::vector<uint8_t> read_buf(64);

    io.Write(old_data.data(), old_data.size(), 0);
    REQUIRE(io.Read(read_buf.size(), 32, read_buf.data()));
    REQUIRE(std::memcmp(read_buf.data(), old_data.data() + 32, read_buf.size()) == 0);

    io.Write(new_data.data(), new_data.size(), 32);
    REQUIRE(io.Read(read_buf.size(), 32, read_buf.data()));
    REQUIRE(std::memcmp(read_buf.data(), new_data.data(), read_buf.size()) == 0);
}

TEST_CASE("CacheIO MultiRead Test", "[CacheIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CacheIOParameter param;
    param.total_cache_size_ = Page::DEFAULT_PAGE_SIZE * 4;

    CacheIO<MemoryIO> io(param, allocator.get(), allocator.get());

    std::vector<uint8_t> data1(30, 0x11);
    std::vector<uint8_t> data2(40, 0x22);
    io.Write(data1.data(), 30, 10);
    io.Write(data2.data(), 40, 100);

    uint64_t sizes[] = {30, 40};
    uint64_t offsets[] = {10, 100};
    std::vector<uint8_t> result(70);
    bool ret = io.MultiRead(result.data(), sizes, offsets, 2);
    REQUIRE(ret);
    REQUIRE(memcmp(result.data(), data1.data(), 30) == 0);
    REQUIRE(memcmp(result.data() + 30, data2.data(), 40) == 0);
}

TEST_CASE("CacheIO Parameter Test", "[CacheIO][ut]") {
    CacheIOParameter param;
    REQUIRE(param.total_cache_size_ == 268435456);
    REQUIRE(param.eviction_strategy_ == "lru");

    JsonType json;
    json["type"].SetString("cache_io");
    json["total_cache_size"].SetUint64(4096);
    json["eviction_strategy"].SetString("lru");

    CacheIOParameter param2(json);
    REQUIRE(param2.total_cache_size_ == 4096);
    REQUIRE(param2.eviction_strategy_ == "lru");

    auto json2 = param2.ToJson();
    REQUIRE(json2["total_cache_size"].GetUint64() == 4096);
}

TEST_CASE("CacheIO ReaderIO Backend Test", "[CacheIO][ut]") {
    const uint64_t kTestSize = 1024;
    std::vector<uint8_t> all_data(kTestSize);
    for (uint64_t i = 0; i < kTestSize; ++i) {
        all_data[i] = static_cast<uint8_t>(i % 256);
    }

    class TestReader : public vsag::Reader {
    public:
        TestReader(const uint8_t* data, uint64_t size) : data_(data), size_(size) {
        }
        void
        Read(uint64_t offset, uint64_t len, void* dest) override {
            memcpy(dest, data_ + offset, len);
        }
        void
        AsyncRead(uint64_t offset, uint64_t len, void* dest, vsag::CallBack callback) override {
            Read(offset, len, dest);
            callback(vsag::IOErrorCode::IO_SUCCESS, "success");
        }
        uint64_t
        Size() const override {
            return size_;
        }

    private:
        const uint8_t* data_;
        uint64_t size_;
    };

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CacheIOParameter param;
    param.total_cache_size_ = Page::DEFAULT_PAGE_SIZE * 8;

    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = std::make_shared<TestReader>(all_data.data(), kTestSize);

    CacheIO<ReaderIO> io(param, allocator.get(), allocator.get());
    io.InitIO(reader_param);

    std::vector<uint8_t> read_buf(kTestSize);
    REQUIRE(io.Read(kTestSize, 0, read_buf.data()));
    REQUIRE(read_buf == all_data);

    // DirectRead
    bool need_release = false;
    const auto* ptr = io.Read(128, 0, need_release);
    REQUIRE(ptr != nullptr);
    REQUIRE(need_release == true);
    io.Release(ptr);

    // MultiRead
    uint64_t sizes[] = {64, 64};
    uint64_t offsets[] = {0, 128};
    std::vector<uint8_t> multi_buf(128);
    REQUIRE(io.MultiRead(multi_buf.data(), sizes, offsets, 2));
    REQUIRE(std::memcmp(multi_buf.data(), all_data.data(), 64) == 0);
    REQUIRE(std::memcmp(multi_buf.data() + 64, all_data.data() + 128, 64) == 0);
}

TEST_CASE("CacheIO Concurrent Read Test", "[CacheIO][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    CacheIOParameter param;
    param.total_cache_size_ = Page::DEFAULT_PAGE_SIZE * 4;

    CacheIO<MemoryIO> io(param, allocator.get(), allocator.get());

    const uint64_t num_pages = 4;
    const uint64_t page_size = Page::DEFAULT_PAGE_SIZE;
    const uint64_t total_size = num_pages * page_size;

    std::vector<uint8_t> all_data(total_size);
    for (uint64_t i = 0; i < total_size; ++i) {
        all_data[i] = static_cast<uint8_t>(i % 256);
    }
    io.Write(all_data.data(), total_size, 0);

    const int num_threads = 8;
    const int reads_per_thread = 100;
    std::atomic<int> errors{0};

    auto worker = [&](int thread_id) {
        std::vector<uint8_t> buf(page_size);
        for (int r = 0; r < reads_per_thread; ++r) {
            uint64_t page_idx = (thread_id + r) % num_pages;
            uint64_t offset = page_idx * page_size;
            if (not io.Read(page_size, offset, buf.data())) {
                errors.fetch_add(1);
                continue;
            }
            if (std::memcmp(buf.data(), all_data.data() + offset, page_size) != 0) {
                errors.fetch_add(1);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) {
        th.join();
    }

    REQUIRE(errors.load() == 0);
}
