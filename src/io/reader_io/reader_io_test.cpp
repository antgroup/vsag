
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

#include "io/reader_io/reader_io.h"

#include <memory>
#include <sstream>

#include "index_common_param.h"
#include "io/common/io_contract_test.h"
#include "io/reader_io/reader_io_parameter.h"
#include "unittest.h"

class TestReader : public vsag::Reader, public vsag::ReaderPrefetcher {
public:
    TestReader(uint8_t* data, uint64_t size) : data_(data), size_(size) {
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

    void
    Prefetch(uint64_t offset, uint64_t len) override {
        if (throw_on_prefetch_) {
            throw std::runtime_error("prefetch failed");
        }
        prefetch_offset_ = offset;
        prefetch_len_ = len;
        ++prefetch_count_;
    }

    uint64_t
    Size() const override {
        return size_;
    }

    uint64_t prefetch_offset_{0};
    uint64_t prefetch_len_{0};
    uint64_t prefetch_count_{0};
    bool throw_on_prefetch_{false};

private:
    const uint8_t* data_{nullptr};
    uint64_t size_{0};
};

TEST_CASE("ReaderIO forwards enabled prefetch hints", "[ut][ReaderIO]") {
    std::vector<uint8_t> data(1024);
    auto reader = std::make_shared<TestReader>(data.data(), data.size());
    auto parameter = std::make_shared<vsag::ReaderIOParameter>();
    parameter->reader = reader;
    parameter->enable_prefetch_hint_ = true;

    vsag::IndexCommonParam common_param;
    common_param.allocator_ = vsag::Engine::CreateDefaultAllocator();
    IOParamPtr io_parameter = parameter;
    ReaderIO io(io_parameter, common_param);

    io.Prefetch(128, 64);
    REQUIRE(reader->prefetch_count_ == 1);
    REQUIRE(reader->prefetch_offset_ == 128);
    REQUIRE(reader->prefetch_len_ == 64);
}

TEST_CASE("ReaderIO clamps prefetch and suppresses reader failures", "[ut][ReaderIO]") {
    std::vector<uint8_t> data(1024);
    auto reader = std::make_shared<TestReader>(data.data(), data.size());
    auto parameter = std::make_shared<vsag::ReaderIOParameter>();
    parameter->reader = reader;
    parameter->enable_prefetch_hint_ = true;

    vsag::IndexCommonParam common_param;
    common_param.allocator_ = vsag::Engine::CreateDefaultAllocator();
    IOParamPtr io_parameter = parameter;
    ReaderIO io(io_parameter, common_param);

    io.Prefetch(1000, 64);
    REQUIRE(reader->prefetch_count_ == 1);
    REQUIRE(reader->prefetch_offset_ == 1000);
    REQUIRE(reader->prefetch_len_ == 24);

    reader->throw_on_prefetch_ = true;
    REQUIRE_NOTHROW(io.Prefetch(128, 64));
}

TEST_CASE("ReaderIO ignores hints for Readers without ReaderPrefetcher", "[ut][ReaderIO]") {
    class ReaderWithoutPrefetch final : public vsag::Reader {
    public:
        explicit ReaderWithoutPrefetch(std::vector<uint8_t> data) : data_(std::move(data)) {
        }

        void
        Read(uint64_t offset, uint64_t len, void* dest) override {
            std::memcpy(dest, data_.data() + offset, len);
        }

        void
        AsyncRead(uint64_t offset, uint64_t len, void* dest, vsag::CallBack callback) override {
            Read(offset, len, dest);
            callback(vsag::IOErrorCode::IO_SUCCESS, "success");
        }

        [[nodiscard]] uint64_t
        Size() const override {
            return data_.size();
        }

    private:
        std::vector<uint8_t> data_;
    };

    auto parameter = std::make_shared<vsag::ReaderIOParameter>();
    parameter->reader = std::make_shared<ReaderWithoutPrefetch>(std::vector<uint8_t>(1024));
    parameter->enable_prefetch_hint_ = true;

    vsag::IndexCommonParam common_param;
    common_param.allocator_ = vsag::Engine::CreateDefaultAllocator();
    IOParamPtr io_parameter = parameter;
    ReaderIO io(io_parameter, common_param);

    REQUIRE_NOTHROW(io.Prefetch(128, 64));
    std::vector<uint8_t> result(64);
    REQUIRE(io.ReadAt(128, 64, result.data()));
}

TEST_CASE("ReaderIO keeps prefetch disabled by default", "[ut][ReaderIO]") {
    std::vector<uint8_t> data(1024);
    auto reader = std::make_shared<TestReader>(data.data(), data.size());
    auto parameter = std::make_shared<vsag::ReaderIOParameter>();
    parameter->reader = reader;

    vsag::IndexCommonParam common_param;
    common_param.allocator_ = vsag::Engine::CreateDefaultAllocator();
    IOParamPtr io_parameter = parameter;
    ReaderIO io(io_parameter, common_param);

    io.Prefetch(128, 64);
    REQUIRE(reader->prefetch_count_ == 0);
}

TEST_CASE("ReaderIO Read Test", "[ut][ReaderIO]") {
    const uint64_t kTestSize = 1024;
    std::vector<uint8_t> all_data(kTestSize);
    for (uint64_t i = 0; i < kTestSize; ++i) {
        all_data[i] = static_cast<uint8_t>(i % 256);
    }

    vsag::IndexCommonParam common_param;
    common_param.allocator_ = vsag::Engine::CreateDefaultAllocator();
    auto reader_param = std::make_shared<vsag::ReaderIOParameter>();
    reader_param->reader = std::make_shared<TestReader>(all_data.data(), all_data.size());
    IOParamPtr io_param = reader_param;

    ReaderIO reader_io(reader_param, common_param);
    REQUIRE(reader_io.Size() == kTestSize);

    ReaderIO generic_reader_io(io_param, common_param);
    REQUIRE(generic_reader_io.Size() == kTestSize);

    SECTION("Test Read normal case") {
        const uint64_t offset = 100;
        const uint64_t size = 256;
        std::vector<uint8_t> buffer(size);
        bool result = reader_io.Read(size, offset, buffer.data());
        REQUIRE(result == true);
        for (uint64_t i = 0; i < size; ++i) {
            REQUIRE(buffer[i] == all_data[offset + i]);
        }
    }

    SECTION("Test Read out of bounds") {
        const uint64_t offset = kTestSize;
        const uint64_t size = 1;
        std::vector<uint8_t> buffer(size);
        bool result = reader_io.Read(size, offset, buffer.data());
        REQUIRE(result == false);
    }

    SECTION("Test direct Read normal case") {
        const uint64_t offset = 100;
        const uint64_t size = 256;
        bool need_release = false;
        const uint8_t* data = reader_io.Read(size, offset, need_release);
        REQUIRE(need_release == true);
        REQUIRE(data != nullptr);
        for (uint64_t i = 0; i < size; ++i) {
            REQUIRE(data[i] == all_data[offset + i]);
        }
        reader_io.Release(data);
    }

    SECTION("Test direct Read out of bounds") {
        const uint64_t offset = kTestSize;
        const uint64_t size = 1;
        bool need_release = false;
        const uint8_t* data = reader_io.Read(size, offset, need_release);
        REQUIRE(data == nullptr);
    }

    SECTION("Test MultiRead multiple reads") {
        const uint64_t count = 2;
        uint64_t offsets[] = {100, 200};
        uint64_t sizes[] = {256, 256};
        std::vector<uint8_t> buffer(sizes[0] + sizes[1]);
        bool result = reader_io.MultiRead(buffer.data(), sizes, offsets, count);
        REQUIRE(result == true);

        for (uint64_t i = 0; i < sizes[0]; ++i) {
            REQUIRE(buffer[i] == all_data[offsets[0] + i]);
        }
        for (uint64_t i = 0; i < sizes[1]; ++i) {
            REQUIRE(buffer[sizes[0] + i] == all_data[offsets[1] + i]);
        }
    }

    SECTION("Test MultiRead with error") {
        const uint64_t count = 1;
        uint64_t offsets[] = {kTestSize};
        uint64_t sizes[] = {1};
        std::vector<uint8_t> buffer(1);
        REQUIRE_THROWS(reader_io.MultiRead(buffer.data(), sizes, offsets, count));
    }

    SECTION("Test cached MultiRead with error") {
        reader_param->enable_read_cache_ = true;
        reader_io.EnableReadCache(reader_param);
        const uint64_t count = 1;
        uint64_t offsets[] = {kTestSize};
        uint64_t sizes[] = {1};
        std::vector<uint8_t> buffer(1);
        REQUIRE_FALSE(reader_io.MultiRead(buffer.data(), sizes, offsets, count));
    }
}

TEST_CASE("SkipDeserialize updates size without writing null data", "[ut][ReaderIO]") {
    const uint64_t kTestSize = 1024;
    std::vector<uint8_t> all_data(kTestSize);
    for (uint64_t i = 0; i < kTestSize; ++i) {
        all_data[i] = static_cast<uint8_t>(i % 256);
    }

    std::stringstream ss;
    vsag::IOStreamWriter writer(ss);
    vsag::StreamWriter::WriteObj(writer, kTestSize);
    writer.Write(reinterpret_cast<const char*>(all_data.data()), kTestSize);
    ss.seekg(0, std::ios::beg);

    vsag::IOStreamReader reader(ss);
    auto allocator = vsag::Engine::CreateDefaultAllocator();
    ReaderIO io(allocator.get());

    io.Deserialize(reader);

    REQUIRE(io.Size() == kTestSize);
    REQUIRE(reader.GetCursor() == sizeof(kTestSize) + kTestSize);
}
