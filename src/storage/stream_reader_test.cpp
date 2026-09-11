

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

#include "stream_reader.h"

#include <cstdint>
#include <limits>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"
#include "vsag_exception.h"

TEST_CASE("StreamReader Skip consumes forward input", "[ut][stream_reader]") {
    std::istringstream input(std::string(20000, 'x') + "tail");
    vsag::ForwardStreamReader reader(input);
    reader.Skip(0);
    REQUIRE(reader.GetCursor() == 0);
    reader.Skip(20000);
    REQUIRE(reader.GetCursor() == 20000);
    char tail[4];
    reader.Read(tail, sizeof(tail));
    REQUIRE(std::string(tail, sizeof(tail)) == "tail");
    REQUIRE_THROWS_AS(reader.Skip(1), vsag::VsagException);
}

TEST_CASE("StreamReader Skip respects bounds without reading random-access data",
          "[ut][stream_reader]") {
    uint64_t read_count = 0;
    vsag::ReadFuncStreamReader reader(
        [&](uint64_t, uint64_t size, void* dest) {
            ++read_count;
            std::memset(dest, 'x', size);
        },
        0,
        100);
    auto slice = reader.Slice(50);
    slice.Skip(20);
    REQUIRE(slice.GetCursor() == 20);
    REQUIRE(reader.GetCursor() == 20);
    REQUIRE_THROWS_AS(slice.Skip(31), vsag::VsagException);
    REQUIRE(slice.GetCursor() == 20);
    slice.Skip(30);
    reader.Skip(50);
    reader.Skip(0);
    REQUIRE(read_count == 0);
    REQUIRE(reader.GetCursor() == 100);
    REQUIRE_THROWS_AS(reader.Skip(1), vsag::VsagException);
    REQUIRE_THROWS_AS(reader.Skip(std::numeric_limits<uint64_t>::max()), vsag::VsagException);
    REQUIRE(reader.GetCursor() == 100);
}

TEST_CASE("BufferStreamReader Skip preserves buffered cursor and delegates unread bytes",
          "[ut][stream_reader]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    const std::string data = "0123456789";
    uint64_t read_count = 0;
    vsag::ReadFuncStreamReader source(
        [&](uint64_t offset, uint64_t size, void* dest) {
            ++read_count;
            std::memcpy(dest, data.data() + offset, size);
        },
        0,
        data.size());
    vsag::BufferStreamReader reader(&source, data.size(), allocator.get());
    reader.Skip(3);
    REQUIRE(read_count == 0);
    REQUIRE(reader.GetCursor() == 3);
    char value = 0;
    reader.Read(&value, 1);
    REQUIRE(value == '3');
    reader.Skip(2);
    REQUIRE(reader.GetCursor() == 6);
    reader.Read(&value, 1);
    REQUIRE(value == '6');
    REQUIRE_THROWS_AS(reader.Skip(4), vsag::VsagException);
    REQUIRE(reader.GetCursor() == 7);
    reader.Skip(3);
    reader.Skip(0);
    REQUIRE(reader.GetCursor() == data.size());
    REQUIRE(read_count == 1);
}

TEST_CASE("BoundedForwardReader Skip consumes only its payload", "[ut][stream_reader]") {
    std::istringstream input("payloadsuffix");
    vsag::ForwardStreamReader source(input);
    vsag::BoundedForwardReader reader(&source, 7);
    REQUIRE_THROWS_AS(reader.Skip(8), vsag::VsagException);
    REQUIRE(source.GetCursor() == 0);
    reader.Skip(7);
    REQUIRE(reader.GetCursor() == 7);
    REQUIRE(source.GetCursor() == 7);
}

// fill buffer with below and return a wrappered StreamReader object:
// ['1' '1' ... repeats 1024 times]
// ['2' '2' ... repeats 1024 times]
// ['3' '3' ... repeats 1024 times]
// ['4' '4' ... repeats 1024 times]
vsag::ReadFuncStreamReader
gen_4k_data_and_return_stream_reader(char* buffer) {
    memset(buffer, '1', 1024);
    memset(buffer + 1024, '2', 1024);
    memset(buffer + 2048, '3', 1024);
    memset(buffer + 3072, '4', 1024);

    auto reader = vsag::ReadFuncStreamReader(
        /*read_func=*/[=](uint64_t offset,
                          uint64_t size,
                          void* dest) { memcpy(dest, buffer + offset, size); },
        /*cursor=*/0,
        /*length=*/4096);

    return reader;
}

TEST_CASE("StreamReader", "[ut][stream_reader]") {
    char buffer[4096]{};
    auto reader = gen_4k_data_and_return_stream_reader(buffer);
    char ch{'0'};

    // PushSeek, Read and Check
    // Expected: 1, 4, 3, 2, 1
    reader.Read(&ch, 1);
    REQUIRE(ch == '1');

    reader.PushSeek(3072);
    reader.Read(&ch, 1);
    REQUIRE(ch == '4');

    reader.PushSeek(2048);
    reader.Read(&ch, 1);
    REQUIRE(ch == '3');

    reader.PushSeek(1024);
    reader.Read(&ch, 1);
    REQUIRE(ch == '2');

    reader.PushSeek(0);
    reader.Read(&ch, 1);
    REQUIRE(ch == '1');

    // PopSeek, Check in Reverse Order
    // Expected: 2, 3, 4, 1 (cursor moved back to the first position)
    reader.PopSeek();
    reader.Read(&ch, 1);
    REQUIRE(ch == '2');

    reader.PopSeek();
    reader.Read(&ch, 1);
    REQUIRE(ch == '3');

    reader.PopSeek();
    reader.Read(&ch, 1);
    REQUIRE(ch == '4');

    reader.PopSeek();
    reader.Read(&ch, 1);
    REQUIRE(ch == '1');
}

TEST_CASE("SliceStreamReader", "[ut][stream_reader]") {
    char buffer[4096]{};
    auto reader = gen_4k_data_and_return_stream_reader(buffer);

    reader.Seek(1024 + 1000);
    auto reader_slice = reader.Slice(48);

    auto check_func = [](const char* array, char expected_char, uint64_t length) -> bool {
        for (uint64_t i = 0; i < length; ++i) {
            if (array[i] != expected_char) {
                return false;
            }
        }
        return true;
    };

    char read_buffer2[24];
    reader_slice.Read(read_buffer2, 24);
    // std::cout << std::string(read_buffer2, 24) << std::endl;
    REQUIRE(check_func(read_buffer2, '2', 24));

    char read_buffer3[24];
    reader_slice.Read(read_buffer3, 24);
    // std::cout << std::string(read_buffer3, 24) << std::endl;
    REQUIRE(check_func(read_buffer3, '3', 24));
}
