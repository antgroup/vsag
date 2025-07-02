
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

#include "fast_bitset.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "fixtures.h"

using namespace vsag;

TEST_CASE("FastBitset basic operations", "[ut][FastBitset]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    FastBitset bs(allocator.get());

    SECTION("Initial state") {
        REQUIRE_FALSE(bs.Test(0));
        REQUIRE(bs.Count() == 0);
        REQUIRE(bs.Dump() == "{}");
    }

    SECTION("Single bit set") {
        bs.Set(10, true);
        REQUIRE(bs.Test(10));
        REQUIRE(bs.Count() == 1);
        REQUIRE(bs.Dump() == "{10}");

        bs.Set(10, false);
        REQUIRE_FALSE(bs.Test(0));
        REQUIRE(bs.Count() == 0);
        REQUIRE(bs.Dump() == "{}");
    }

    SECTION("Multiple bits in same word") {
        bs.Set(1, true);
        bs.Set(3, true);
        bs.Set(5, true);
        REQUIRE(bs.Count() == 3);
        REQUIRE(bs.Dump() == "{1,3,5}");
    }

    SECTION("Multiple bit set in different word") {
        bs.Set(0, true);
        bs.Set(103, true);
        REQUIRE(bs.Test(0));
        REQUIRE(bs.Test(103));
        REQUIRE(bs.Count() == 2);
        REQUIRE(bs.Dump() == "{0,103}");

        bs.Set(0, false);
        REQUIRE_FALSE(bs.Test(0));
        REQUIRE(bs.Test(103));
        REQUIRE(bs.Count() == 1);
        REQUIRE(bs.Dump() == "{103}");
    }

    SECTION("Resize on large position") {
        bs.Set(12800, true);
        REQUIRE(bs.Test(12800));
        REQUIRE(bs.Count() == 1);
        REQUIRE(bs.Dump() == "{12800}");
    }

    SECTION("Serialize and deserialize") {
        fixtures::TempDir dir("fast_bitset");
        auto path = dir.GenerateRandomFile();
        std::ofstream ofs(path, std::ios::binary);
        IOStreamWriter writer(ofs);

        bs.Set(101, true);
        bs.Serialize(writer);
        ofs.close();

        std::ifstream ifs(path, std::ios::binary);
        IOStreamReader reader(ifs);
        FastBitset bitset2(allocator.get());
        bitset2.Deserialize(reader);
        ifs.close();
        auto dump = bitset2.Dump();
        REQUIRE(dump == "{101}");
    }
}

TEST_CASE("FastBitset bitwise operations", "[ut][FastBitset]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    FastBitset a(allocator.get());
    FastBitset b(allocator.get());

    SECTION("OR operation") {
        a.Set(10, true);
        b.Set(111, true);
        a.Or(b);

        REQUIRE(a.Test(10));
        REQUIRE(a.Test(111));
        REQUIRE(a.Count() == 2);
        REQUIRE(a.Dump() == "{10,111}");

        FastBitset c(allocator.get());
        c.Set(64, true);
        c.Set(111, true);
        a.Or(c);
        REQUIRE(a.Test(64));
        REQUIRE(a.Count() == 3);
    }

    SECTION("AND operation") {
        a.Set(2, true);
        a.Set(215, true);
        b.Set(215, true);
        b.Set(1928, true);
        a.And(b);

        REQUIRE_FALSE(a.Test(2));
        REQUIRE(a.Test(215));
        REQUIRE_FALSE(a.Test(1928));
        REQUIRE(a.Count() == 1);
        REQUIRE(a.Dump() == "{215}");
    }

    SECTION("XOR operation") {
        a.Set(100, true);
        a.Set(1001, true);
        b.Set(1001, true);
        b.Set(2025, true);
        a.Xor(b);

        REQUIRE(a.Test(100));
        REQUIRE_FALSE(a.Test(1001));
        REQUIRE(a.Test(2025));
        REQUIRE(a.Count() == 2);
        REQUIRE(a.Dump() == "{100,2025}");
    }

    SECTION("NOT operation") {
        a.Set(100, true);
        a.Set(1001, true);
        a.Not();
        REQUIRE_FALSE(a.Test(100));
        REQUIRE_FALSE(a.Test(1001));
        a.Not();
        REQUIRE(a.Test(100));
        REQUIRE(a.Test(1001));
        REQUIRE(a.Count() == 2);
    }

    SECTION("AND Operation With Pointer") {
        auto ptr1 = std::make_shared<FastBitset>(allocator.get());
        auto ptr2 = std::make_shared<FastBitset>(allocator.get());
        ptr1->Set(2, true);
        ptr1->Set(215, true);
        ptr2->Set(215, true);
        ptr2->Set(1929, true);
        ptr1->And(ptr2);

        REQUIRE_FALSE(ptr1->Test(2));
        REQUIRE(ptr1->Test(215));
        REQUIRE_FALSE(ptr1->Test(1929));
        REQUIRE(ptr1->Count() == 1);
        REQUIRE(ptr1->Dump() == "{215}");

        ptr2 = nullptr;
        ptr1->And(ptr2);
        REQUIRE_FALSE(ptr1->Test(215));
        REQUIRE(ptr1->Count() == 0);
        REQUIRE(ptr1->Dump() == "{}");
    }

    SECTION("OR Operation With Pointer") {
        auto ptr1 = std::make_shared<FastBitset>(allocator.get());
        auto ptr2 = std::make_shared<FastBitset>(allocator.get());
        ptr1->Set(10, true);
        ptr2->Set(111, true);
        ptr1->Or(ptr2);

        REQUIRE(ptr1->Test(10));
        REQUIRE(ptr1->Test(111));
        REQUIRE(ptr1->Count() == 2);
        REQUIRE(ptr1->Dump() == "{10,111}");

        auto ptr3 = std::make_shared<FastBitset>(allocator.get());
        ptr3->Set(64, true);
        ptr3->Set(111, true);
        ptr1->Or(ptr3);
        REQUIRE(ptr1->Test(64));
        REQUIRE(ptr1->Count() == 3);

        ptr2 = nullptr;
        ptr1->Or(ptr2);
        REQUIRE(ptr1->Count() == 3);
        REQUIRE(ptr1->Test(64));
        REQUIRE(ptr1->Dump() == "{10,64,111}");
    }

    SECTION("XOR Operation With Pointer") {
        auto ptr1 = std::make_shared<FastBitset>(allocator.get());
        auto ptr2 = std::make_shared<FastBitset>(allocator.get());
        ptr1->Set(100, true);
        ptr1->Set(1001, true);
        ptr2->Set(1001, true);
        ptr2->Set(2025, true);
        ptr1->Xor(ptr2);

        REQUIRE(ptr1->Test(100));
        REQUIRE_FALSE(ptr1->Test(1001));
        REQUIRE(ptr1->Test(2025));
        REQUIRE(ptr1->Count() == 2);
        REQUIRE(ptr1->Dump() == "{100,2025}");

        ptr2 = nullptr;
        ptr1->Xor(ptr2);
        REQUIRE(ptr1->Count() == 2);
        REQUIRE(ptr1->Dump() == "{100,2025}");
    }

    SECTION("AND Operation With Vector Pointer") {
        ComputableBitsetPtr ptr1 = std::make_shared<FastBitset>(allocator.get());
        auto ptr2 = std::make_shared<FastBitset>(allocator.get());
        auto ptr3 = std::make_shared<FastBitset>(allocator.get());
        std::vector<ComputableBitsetPtr> vec(2);
        vec[0] = ptr2;
        vec[1] = ptr3;
        ptr1->Set(100, true);
        ptr1->Set(1001, true);
        ptr2->Set(1001, true);
        ptr2->Set(2025, true);
        ptr3->Set(1001, true);
        ptr3->Set(2020, true);
        ptr1->And(vec);
        REQUIRE(ptr1->Test(1001));
        REQUIRE_FALSE(ptr1->Test(2020));
        REQUIRE_FALSE(ptr1->Test(2025));
        REQUIRE_FALSE(ptr1->Test(100));
        REQUIRE(ptr1->Count() == 1);
        REQUIRE(ptr1->Dump() == "{1001}");
    }

    SECTION("OR Operation With Vector Pointer") {
        ComputableBitsetPtr ptr1 = std::make_shared<FastBitset>(allocator.get());
        auto ptr2 = std::make_shared<FastBitset>(allocator.get());
        auto ptr3 = std::make_shared<FastBitset>(allocator.get());
        std::vector<ComputableBitsetPtr> vec(2);
        vec[0] = ptr2;
        vec[1] = ptr3;
        ptr1->Set(100, true);
        ptr1->Set(1001, true);
        ptr2->Set(1001, true);
        ptr2->Set(2025, true);
        ptr3->Set(1001, true);
        ptr3->Set(2020, true);
        ptr1->Or(vec);
        REQUIRE(ptr1->Test(100));
        REQUIRE(ptr1->Test(1001));
        REQUIRE(ptr1->Test(2020));
        REQUIRE(ptr1->Test(2025));
        REQUIRE(ptr1->Count() == 4);
        REQUIRE(ptr1->Dump() == "{100,1001,2020,2025}");
    }

    SECTION("XOR Operation With Vector Pointer") {
        ComputableBitsetPtr ptr1 = std::make_shared<FastBitset>(allocator.get());
        auto ptr2 = std::make_shared<FastBitset>(allocator.get());
        auto ptr3 = std::make_shared<FastBitset>(allocator.get());
        std::vector<ComputableBitsetPtr> vec(2);
        vec[0] = ptr2;
        vec[1] = ptr3;
        ptr1->Set(100, true);
        ptr1->Set(1001, true);
        ptr2->Set(1001, true);
        ptr2->Set(2025, true);
        ptr3->Set(100, true);
        ptr3->Set(2020, true);
        ptr1->Xor(vec);
        REQUIRE_FALSE(ptr1->Test(100));
        REQUIRE_FALSE(ptr1->Test(1001));
        REQUIRE(ptr1->Test(2020));
        REQUIRE(ptr1->Test(2025));
        REQUIRE(ptr1->Dump() == "{2020,2025}");
    }
}
