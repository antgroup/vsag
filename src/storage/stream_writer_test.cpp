
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

#include "stream_writer.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <iostream>

#include "fixtures.h"
#include "impl/allocator/default_allocator.h"

TEST_CASE("BufferStreamWriter", "[ut][stream_reader]") {
    auto* buffer = new char[4096]{};
    BufferStreamWriter writer(buffer);

    SECTION("float") {
        float number = 1.234567;
        StreamWriter::WriteObj(writer, number);
        float number2 = 0.0F;
        memcpy(&number2, buffer, sizeof(float));
        REQUIRE(number == number2);
    }

    SECTION("string") {
        std::string text{"hello world!"};
        StreamWriter::WriteString(writer, text);

        struct {
            uint64_t size;
            char objs[12];
        } payload;
        memcpy(&payload, buffer, sizeof(payload));

        REQUIRE(payload.size == 12);
        CHECK(0 == memcmp("hello world!", payload.objs, 12));
    }

    SECTION("empty string") {
        std::string empty_text;
        CHECK_NOTHROW(StreamWriter::WriteString(writer, empty_text));
    }

    SECTION("std::vector") {
        std::vector<float> numbers{1.1, 2.2, 3.3, 4.4, 5.5};
        StreamWriter::WriteVector(writer, numbers);

        struct {
            uint64_t size;
            float objs[5];
        } payload;
        memcpy(&payload, buffer, sizeof(payload));

        REQUIRE(payload.size == 5);
        CHECK(fixtures::dist_t(payload.objs[0]) == 1.1);
        CHECK(fixtures::dist_t(payload.objs[1]) == 2.2);
        CHECK(fixtures::dist_t(payload.objs[2]) == 3.3);
        CHECK(fixtures::dist_t(payload.objs[3]) == 4.4);
        CHECK(fixtures::dist_t(payload.objs[4]) == 5.5);
    }

    SECTION("empty std::vector") {
        std::vector<float> numbers{};
        CHECK_NOTHROW(StreamWriter::WriteVector(writer, numbers));
    }

    SECTION("vsag::Vector") {
        vsag::DefaultAllocator allocator;
        vsag::Vector<int64_t> numbers(4, &allocator);
        numbers[0] = 2;
        numbers[1] = 3;
        numbers[2] = 5;
        numbers[3] = 7;
        StreamWriter::WriteVector(writer, numbers);

        struct {
            uint64_t size;
            int64_t objs[4];
        } payload;
        memcpy(&payload, buffer, sizeof(payload));

        REQUIRE(payload.size == 4);
        CHECK(payload.objs[0] == 2);
        CHECK(payload.objs[1] == 3);
        CHECK(payload.objs[2] == 5);
        CHECK(payload.objs[3] == 7);
    }

    SECTION("empty vsag::Vector") {
        std::vector<float> numbers{};
        CHECK_NOTHROW(StreamWriter::WriteVector(writer, numbers));
    }

    delete[] buffer;
}
