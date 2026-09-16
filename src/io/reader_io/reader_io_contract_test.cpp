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

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "io/reader_io/reader_io.h"
#include "storage/serialization.h"
#include "unittest.h"

namespace vsag {
namespace {

class OwnedTestReader : public Reader {
public:
    explicit OwnedTestReader(std::vector<uint8_t> data) : data_(std::move(data)) {
    }

    void
    Read(uint64_t offset, uint64_t len, void* destination) override {
        std::memcpy(destination, data_.data() + offset, len);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* destination, CallBack callback) override {
        Read(offset, len, destination);
        callback(IOErrorCode::IO_SUCCESS, "success");
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return data_.size();
    }

private:
    std::vector<uint8_t> data_;
};

std::vector<uint8_t>
MakeReaderData(uint64_t size, uint8_t seed = 0) {
    std::vector<uint8_t> data(size);
    for (uint64_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(i * 53 + seed);
    }
    return data;
}

ReaderIOParamPtr
MakeReaderParam(std::vector<uint8_t> data, bool enable_cache = false) {
    auto parameter = std::make_shared<ReaderIOParameter>();
    parameter->reader = std::make_shared<OwnedTestReader>(std::move(data));
    parameter->enable_read_cache_ = enable_cache;
    parameter->read_cache_total_size_ = 4 * Page::DEFAULT_PAGE_SIZE;
    return parameter;
}

}  // namespace

TEST_CASE("ReaderIO rejects an invalid generic parameter", "[ut][ReaderIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    IOParamPtr invalid_param;

    REQUIRE_THROWS_AS(ReaderIO(invalid_param, common_param), VsagException);
}

TEST_CASE("ReaderIO direct, batch, cache, and compatibility adapters", "[ut][ReaderIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data = MakeReaderData(2 * Page::DEFAULT_PAGE_SIZE + 117);
    auto parameter = MakeReaderParam(data, true);
    ReaderIO canonical(allocator.get());
    canonical.InitIO(parameter);
    REQUIRE(canonical.Size() == data.size());

    std::array<uint8_t, 211> direct{};
    REQUIRE(canonical.ReadAt(Page::DEFAULT_PAGE_SIZE - 17, direct.size(), direct.data()));
    REQUIRE(std::memcmp(direct.data(), data.data() + Page::DEFAULT_PAGE_SIZE - 17, direct.size()) ==
            0);

    auto lease = canonical.Acquire(31, 701);
    REQUIRE(lease);
    REQUIRE(std::memcmp(lease.Data(), data.data() + 31, lease.Size()) == 0);

    bool need_release = false;
    const uint8_t* compatibility_data = canonical.Read(509, 43, need_release);
    REQUIRE(compatibility_data != nullptr);
    REQUIRE(need_release);
    REQUIRE(std::memcmp(compatibility_data, data.data() + 43, 509) == 0);
    canonical.Release(compatibility_data);

    std::array<uint8_t, 29> first{};
    std::array<uint8_t, 47> second{};
    std::array<ReadRequest, 2> requests{{
        {first.data(), 19, first.size()},
        {second.data(), Page::DEFAULT_PAGE_SIZE + 23, second.size()},
    }};
    REQUIRE(canonical.ReadMany(requests.data(), requests.size()));
    REQUIRE(std::memcmp(first.data(), data.data() + 19, first.size()) == 0);
    REQUIRE(std::memcmp(second.data(), data.data() + Page::DEFAULT_PAGE_SIZE + 23, second.size()) ==
            0);

    std::array<uint64_t, 2> sizes{31, 53};
    std::array<uint64_t, 2> offsets{7, Page::DEFAULT_PAGE_SIZE + 41};
    std::array<uint8_t, 84> contiguous{};
    REQUIRE(canonical.MultiRead(contiguous.data(), sizes.data(), offsets.data(), offsets.size()));
    REQUIRE(std::memcmp(contiguous.data(), data.data() + offsets[0], sizes[0]) == 0);
    REQUIRE(std::memcmp(contiguous.data() + sizes[0], data.data() + offsets[1], sizes[1]) == 0);
}

TEST_CASE("ReaderIO binds skipped serialized ranges and rebinds without stale cache",
          "[ut][ReaderIO][compatibility][cache]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto payload = MakeReaderData(Page::DEFAULT_PAGE_SIZE + 73, 9);
    std::stringstream stream;
    IOStreamWriter writer(stream);
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(payload.size()));
    writer.Write(reinterpret_cast<const char*>(payload.data()), payload.size());
    std::string serialized = stream.str();
    std::vector<uint8_t> serialized_bytes(serialized.begin(), serialized.end());

    stream.seekg(0, std::ios::beg);
    IOStreamReader stream_reader(stream);
    ReaderIO io(allocator.get());
    io.Deserialize(stream_reader);
    auto first_parameter = MakeReaderParam(serialized_bytes, true);
    io.InitIO(first_parameter);
    REQUIRE(io.Size() == payload.size());
    REQUIRE(stream_reader.GetCursor() == sizeof(uint64_t) + payload.size());
    std::vector<uint8_t> actual(payload.size());
    REQUIRE(io.ReadAt(0, actual.size(), actual.data()));
    REQUIRE(actual == payload);

    auto replacement = MakeReaderData(payload.size(), 103);
    auto replacement_serialized = serialized_bytes;
    std::copy(
        replacement.begin(), replacement.end(), replacement_serialized.begin() + sizeof(uint64_t));
    auto second_parameter = MakeReaderParam(std::move(replacement_serialized), true);
    io.InitIO(second_parameter);
    std::fill(actual.begin(), actual.end(), 0);
    REQUIRE(io.ReadAt(0, actual.size(), actual.data()));
    REQUIRE(actual == replacement);
}

TEST_CASE("ReaderIO deserialization replaces its initialized reader size",
          "[ut][ReaderIO][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto payload = MakeReaderData(Page::DEFAULT_PAGE_SIZE + 73, 9);
    std::stringstream stream;
    IOStreamWriter writer(stream);
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(payload.size()));
    writer.Write(reinterpret_cast<const char*>(payload.data()), payload.size());
    std::string serialized = stream.str();
    std::vector<uint8_t> serialized_bytes(serialized.begin(), serialized.end());

    ReaderIO io(allocator.get());
    io.InitIO(MakeReaderParam(serialized_bytes, true));
    REQUIRE(io.Size() == serialized_bytes.size());

    stream.seekg(0, std::ios::beg);
    IOStreamReader stream_reader(stream);
    io.Deserialize(stream_reader);
    REQUIRE(io.Size() == payload.size());
    REQUIRE(stream_reader.GetCursor() == serialized_bytes.size());

    std::vector<uint8_t> actual(payload.size());
    REQUIRE(io.ReadAt(0, actual.size(), actual.data()));
    REQUIRE(actual == payload);
}

TEST_CASE("ReaderIO releases compatibility buffer when an unbound read throws",
          "[ut][ReaderIO][exception-safety]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto payload = MakeReaderData(257, 17);
    std::stringstream stream;
    IOStreamWriter writer(stream);
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(payload.size()));
    writer.Write(reinterpret_cast<const char*>(payload.data()), payload.size());

    stream.seekg(0, std::ios::beg);
    IOStreamReader stream_reader(stream);
    ReaderIO io(allocator.get());
    io.Deserialize(stream_reader);

    bool need_release = false;
    REQUIRE_THROWS(io.Read(31, 7, need_release));
    REQUIRE_FALSE(need_release);
}

TEST_CASE("ReaderIO overflow does not commit deserialized state",
          "[ut][ReaderIO][exception-safety][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::stringstream stream;
    IOStreamWriter writer(stream);
    StreamWriter::WriteObj(writer, std::numeric_limits<uint64_t>::max());

    stream.seekg(0, std::ios::beg);
    IOStreamReader stream_reader(stream);
    ReaderIO io(allocator.get());
    REQUIRE_FALSE(io.HasDeserialized());
    REQUIRE(io.Size() == 0);
    REQUIRE_THROWS_AS(io.Deserialize(stream_reader), VsagException);
    REQUIRE_FALSE(io.HasDeserialized());
    REQUIRE(io.Size() == 0);
}

}  // namespace vsag
