
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

#include "vsag/attribute.h"

#include <catch2/catch_all.hpp>

#include "fixtures.h"

using namespace vsag;

TEST_CASE("Attribute Equal Test", "[ut][Attribute]") {
    auto test_equal = [](const auto& attr1, const auto& attr2, bool expected) {
        REQUIRE(attr1->Equal(attr2.get()) == expected);
        REQUIRE(attr2->Equal(attr1.get()) == expected);
    };

    SECTION("INT32") {
        auto a1 = std::make_shared<AttributeValue<int32_t>>();
        a1->GetValue().push_back(123);
        auto a2 = std::make_shared<AttributeValue<int32_t>>();
        a2->GetValue().push_back(123);
        auto a3 = std::make_shared<AttributeValue<int32_t>>();
        a3->GetValue().push_back(456);

        test_equal(a1, a2, true);
        test_equal(a1, a3, false);
    }

    SECTION("UINT32") {
        auto a1 = std::make_shared<AttributeValue<uint32_t>>();
        a1->GetValue().push_back(100);
        auto a2 = std::make_shared<AttributeValue<uint32_t>>();
        a2->GetValue().push_back(100);
        auto a3 = std::make_shared<AttributeValue<uint32_t>>();
        a3->GetValue().push_back(200);

        test_equal(a1, a2, true);
        test_equal(a1, a3, false);
    }

    SECTION("INT64") {
        auto a1 = std::make_shared<AttributeValue<int64_t>>();
        a1->GetValue().push_back(123456789012345);
        auto a2 = std::make_shared<AttributeValue<int64_t>>();
        a2->GetValue().push_back(123456789012345);
        auto a3 = std::make_shared<AttributeValue<int64_t>>();
        a3->GetValue().push_back(987654321098765);

        test_equal(a1, a2, true);
        test_equal(a1, a3, false);
    }

    SECTION("UINT64") {
        auto a1 = std::make_shared<AttributeValue<uint64_t>>();
        a1->GetValue().push_back(1000000000000000000ULL);
        auto a2 = std::make_shared<AttributeValue<uint64_t>>();
        a2->GetValue().push_back(1000000000000000000ULL);
        auto a3 = std::make_shared<AttributeValue<uint64_t>>();
        a3->GetValue().push_back(2000000000000000000ULL);

        test_equal(a1, a2, true);
        test_equal(a1, a3, false);
    }

    SECTION("INT8") {
        auto a1 = std::make_shared<AttributeValue<int8_t>>();
        a1->GetValue().push_back(-128);
        auto a2 = std::make_shared<AttributeValue<int8_t>>();
        a2->GetValue().push_back(-128);
        auto a3 = std::make_shared<AttributeValue<int8_t>>();
        a3->GetValue().push_back(0);

        test_equal(a1, a2, true);
        test_equal(a1, a3, false);
    }

    SECTION("UINT8") {
        auto a1 = std::make_shared<AttributeValue<uint8_t>>();
        a1->GetValue().push_back(255);
        auto a2 = std::make_shared<AttributeValue<uint8_t>>();
        a2->GetValue().push_back(255);
        auto a3 = std::make_shared<AttributeValue<uint8_t>>();
        a3->GetValue().push_back(0);

        test_equal(a1, a2, true);
        test_equal(a1, a3, false);
    }

    SECTION("INT16") {
        auto a1 = std::make_shared<AttributeValue<int16_t>>();
        a1->GetValue().push_back(-32768);
        auto a2 = std::make_shared<AttributeValue<int16_t>>();
        a2->GetValue().push_back(-32768);
        auto a3 = std::make_shared<AttributeValue<int16_t>>();
        a3->GetValue().push_back(0);

        test_equal(a1, a2, true);
        test_equal(a1, a3, false);
    }

    SECTION("UINT16") {
        auto a1 = std::make_shared<AttributeValue<uint16_t>>();
        a1->GetValue().push_back(65535);
        auto a2 = std::make_shared<AttributeValue<uint16_t>>();
        a2->GetValue().push_back(65535);
        auto a3 = std::make_shared<AttributeValue<uint16_t>>();
        a3->GetValue().push_back(0);

        test_equal(a1, a2, true);
        test_equal(a1, a3, false);
    }

    SECTION("STRING") {
        auto a1 = std::make_shared<AttributeValue<std::string>>();
        a1->GetValue().push_back("hello");
        auto a2 = std::make_shared<AttributeValue<std::string>>();
        a2->GetValue().push_back("hello");
        auto a3 = std::make_shared<AttributeValue<std::string>>();
        a3->GetValue().push_back("world");

        test_equal(a1, a2, true);
        test_equal(a1, a3, false);
    }

    SECTION("Different Type") {
        auto a1 = std::make_shared<AttributeValue<int32_t>>();
        a1->GetValue().push_back(123);
        auto a2 = std::make_shared<AttributeValue<uint32_t>>();
        a2->GetValue().push_back(123);

        REQUIRE(a1->Equal(a2.get()) == false);
        REQUIRE(a2->Equal(a1.get()) == false);
    }

    SECTION("Different Count") {
        auto a1 = std::make_shared<AttributeValue<int32_t>>();
        a1->GetValue().push_back(1);
        a1->GetValue().push_back(2);
        auto a2 = std::make_shared<AttributeValue<int32_t>>();
        a2->GetValue().push_back(1);

        REQUIRE(a1->Equal(a2.get()) == false);
        REQUIRE(a2->Equal(a1.get()) == false);
    }
}
