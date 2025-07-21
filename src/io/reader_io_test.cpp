
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

#include "reader_io.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "basic_io_test.h"
#include "reader_io_parameter.h"

class TestReader : public vsag::Reader {
public:
    TestReader(uint8_t* data, size_t size) : data_(data), size_(size) {
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
    const uint8_t* data_{nullptr};
    size_t size_{0};
};

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

    ReaderIO reader_io(reader_param, common_param);
    reader_io.InitIOImpl(reader_param);
    reader_io.start_ = 0;
    reader_io.size_ = kTestSize;

    SECTION("Test ReadImpl normal case") {
        const uint64_t offset = 100;
        const uint64_t size = 256;
        std::vector<uint8_t> buffer(size);
        bool result = reader_io.ReadImpl(size, offset, buffer.data());
        REQUIRE(result == true);
        for (uint64_t i = 0; i < size; ++i) {
            REQUIRE(buffer[i] == all_data[offset + i]);
        }
    }

    SECTION("Test ReadImpl out of bounds") {
        const uint64_t offset = kTestSize;
        const uint64_t size = 1;
        std::vector<uint8_t> buffer(size);
        bool result = reader_io.ReadImpl(size, offset, buffer.data());
        REQUIRE(result == false);
    }

    SECTION("Test DirectReadImpl normal case") {
        const uint64_t offset = 100;
        const uint64_t size = 256;
        bool need_release = false;
        const uint8_t* data = reader_io.DirectReadImpl(size, offset, need_release);
        REQUIRE(need_release == true);
        REQUIRE(data != nullptr);
        for (uint64_t i = 0; i < size; ++i) {
            REQUIRE(data[i] == all_data[offset + i]);
        }
        reader_io.ReleaseImpl(data);  // 释放内存
    }

    SECTION("Test DirectReadImpl out of bounds") {
        const uint64_t offset = kTestSize;
        const uint64_t size = 1;
        bool need_release = false;
        const uint8_t* data = reader_io.DirectReadImpl(size, offset, need_release);
        REQUIRE(data == nullptr);
    }

    SECTION("Test MultiReadImpl multiple reads") {
        const uint64_t count = 2;
        uint64_t offsets[] = {100, 200};
        uint64_t sizes[] = {256, 256};
        std::vector<uint8_t> buffer(sizes[0] + sizes[1]);
        bool result = reader_io.MultiReadImpl(buffer.data(), sizes, offsets, count);
        REQUIRE(result == true);

        for (uint64_t i = 0; i < sizes[0]; ++i) {
            REQUIRE(buffer[i] == all_data[offsets[0] + i]);
        }
        for (uint64_t i = 0; i < sizes[1]; ++i) {
            REQUIRE(buffer[sizes[0] + i] == all_data[offsets[1] + i]);
        }
    }

    SECTION("Test MultiReadImpl with error") {
        const uint64_t count = 1;
        uint64_t offsets[] = {kTestSize};
        uint64_t sizes[] = {1};
        std::vector<uint8_t> buffer(1);
        REQUIRE_THROWS(reader_io.MultiReadImpl(buffer.data(), sizes, offsets, count));
    }
}
