
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

class TestReader : public vsag::Reader {
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

    uint64_t
    Size() const override {
        return size_;
    }

private:
    const uint8_t* data_{nullptr};
    uint64_t size_{0};
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
