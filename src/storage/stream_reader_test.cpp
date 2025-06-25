

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

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

// fill buffer with below and return a wrappered StreamReader object:
// ['1' '1' ... repeats 1024 times]
// ['2' '2' ... repeats 1024 times]
// ['3' '3' ... repeats 1024 times]
// ['4' '4' ... repeats 1024 times]
ReadFuncStreamReader
gen_4k_data_and_return_stream_reader(char* buffer) {
    memset(buffer, '1', 1024);
    memset(buffer + 1024, '2', 1024);
    memset(buffer + 2048, '3', 1024);
    memset(buffer + 3072, '4', 1024);

    auto reader = ReadFuncStreamReader(
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
