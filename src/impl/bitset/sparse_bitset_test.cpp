
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

#include "sparse_bitset.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <roaring.hh>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"

using namespace roaring;
using namespace vsag;

TEST_CASE("SparseBitset Test", "[ut][bitset]") {
    SparseBitset bitset;
    // empty
    REQUIRE(bitset.Count() == 0);

    // set to true
    bitset.Set(100, true);
    REQUIRE(bitset.Test(100));
    REQUIRE(bitset.Count() == 1);

    // set to false
    bitset.Set(100, false);
    REQUIRE_FALSE(bitset.Test(100));
    REQUIRE(bitset.Count() == 0);

    // not set
    REQUIRE_FALSE(bitset.Test(1234567890));

    // dump
    bitset.Set(100, false);
    REQUIRE(bitset.Dump() == "{}");
    bitset.Set(100, true);
    auto dumped = bitset.Dump();
    REQUIRE(dumped == "{100}");

    fixtures::TempDir dir("sparse_bitset");
    auto path = dir.GenerateRandomFile();
    std::ofstream ofs(path, std::ios::binary);
    IOStreamWriter writer(ofs);
    bitset.Serialize(writer);
    ofs.close();

    std::ifstream ifs(path, std::ios::binary);
    IOStreamReader reader(ifs);
    SparseBitset bitset2;
    bitset2.Deserialize(reader);
    ifs.close();
    dumped = bitset.Dump();
    REQUIRE(dumped == "{100}");
}

TEST_CASE("SparseBitset Or Test", "[ut][bitset]") {
    SECTION("both empty") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset1.Or(bitset2);
        REQUIRE(bitset1.Count() == 0);
        REQUIRE(bitset1.Dump() == "{}");
    }

    SECTION("empty and non-empty") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset2.Set(100, true);
        bitset1.Or(bitset2);
        REQUIRE(bitset1.Test(100));
        REQUIRE(bitset1.Count() == 1);
        REQUIRE(bitset1.Dump() == "{100}");
    }

    SECTION("disjoint sets") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset1.Set(100, true);
        bitset2.Set(200, true);
        bitset1.Or(bitset2);
        REQUIRE(bitset1.Test(100));
        REQUIRE(bitset1.Test(200));
        REQUIRE(bitset1.Count() == 2);
        REQUIRE(bitset1.Dump() == "{100,200}");
    }

    SECTION("overlapping sets") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset1.Set(100, true);
        bitset2.Set(100, true);
        bitset1.Or(bitset2);
        REQUIRE(bitset1.Count() == 1);
        REQUIRE(bitset1.Dump() == "{100}");
    }
}

TEST_CASE("SparseBitset And Test", "[ut][bitset]") {
    SECTION("both empty") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset1.And(bitset2);
        REQUIRE(bitset1.Count() == 0);
        REQUIRE(bitset1.Dump() == "{}");
    }

    SECTION("empty and non-empty") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset2.Set(100, true);
        bitset1.And(bitset2);
        REQUIRE(bitset1.Count() == 0);
    }

    SECTION("common elements") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset1.Set(100, true);
        bitset1.Set(200, true);
        bitset2.Set(200, true);
        bitset2.Set(300, true);
        bitset1.And(bitset2);
        REQUIRE(bitset1.Count() == 1);
        REQUIRE(bitset1.Test(200));
        REQUIRE_FALSE(bitset1.Test(100));
        REQUIRE_FALSE(bitset1.Test(300));
        REQUIRE(bitset1.Dump() == "{200}");
    }

    SECTION("no common elements") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset1.Set(100, true);
        bitset2.Set(200, true);
        bitset1.And(bitset2);
        REQUIRE(bitset1.Count() == 0);
        REQUIRE(bitset1.Dump() == "{}");
    }
}

TEST_CASE("SparseBitset Xor Test", "[ut][bitset]") {
    SECTION("both empty") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset1.Xor(bitset2);
        REQUIRE(bitset1.Count() == 0);
        REQUIRE(bitset1.Dump() == "{}");
    }

    SECTION("empty and non-empty") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset2.Set(100, true);
        bitset1.Xor(bitset2);
        REQUIRE(bitset1.Test(100));
        REQUIRE(bitset1.Count() == 1);
        REQUIRE(bitset1.Dump() == "{100}");
    }

    SECTION("partial overlap") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset1.Set(100, true);
        bitset1.Set(200, true);
        bitset2.Set(200, true);
        bitset2.Set(300, true);
        bitset1.Xor(bitset2);
        REQUIRE(bitset1.Count() == 2);
        REQUIRE(bitset1.Test(100));
        REQUIRE_FALSE(bitset1.Test(200));
        REQUIRE(bitset1.Test(300));
        REQUIRE(bitset1.Dump() == "{100,300}");
    }

    SECTION("identical sets") {
        SparseBitset bitset1;
        SparseBitset bitset2;
        bitset1.Set(100, true);
        bitset2.Set(100, true);
        bitset1.Xor(bitset2);
        REQUIRE(bitset1.Count() == 0);
        REQUIRE(bitset1.Dump() == "{}");
    }
}

TEST_CASE("SparseBitset Bitwise Operations", "[ut][SparseBitset]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    SparseBitset a(allocator.get());
    SparseBitset b(allocator.get());

    SECTION("OR operation") {
        a.Set(10, true);
        b.Set(111, true);
        a.Or(b);

        REQUIRE(a.Test(10));
        REQUIRE(a.Test(111));
        REQUIRE(a.Count() == 2);
        REQUIRE(a.Dump() == "{10,111}");

        SparseBitset c(allocator.get());
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
        REQUIRE(a.Test(100));
        REQUIRE(a.Test(1001));
        a.Not();
        REQUIRE_FALSE(a.Test(100));
        REQUIRE_FALSE(a.Test(1001));
    }

    SECTION("AND Operation With Pointer") {
        auto ptr1 = std::make_shared<SparseBitset>(allocator.get());
        auto ptr2 = std::make_shared<SparseBitset>(allocator.get());
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
        auto ptr1 = std::make_shared<SparseBitset>(allocator.get());
        auto ptr2 = std::make_shared<SparseBitset>(allocator.get());
        ptr1->Set(10, true);
        ptr2->Set(111, true);
        ptr1->Or(ptr2);

        REQUIRE(ptr1->Test(10));
        REQUIRE(ptr1->Test(111));
        REQUIRE(ptr1->Count() == 2);
        REQUIRE(ptr1->Dump() == "{10,111}");

        auto ptr3 = std::make_shared<SparseBitset>(allocator.get());
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
        auto ptr1 = std::make_shared<SparseBitset>(allocator.get());
        auto ptr2 = std::make_shared<SparseBitset>(allocator.get());
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
        ComputableBitsetPtr ptr1 = std::make_shared<SparseBitset>(allocator.get());
        auto ptr2 = std::make_shared<SparseBitset>(allocator.get());
        auto ptr3 = std::make_shared<SparseBitset>(allocator.get());
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
        ComputableBitsetPtr ptr1 = std::make_shared<SparseBitset>(allocator.get());
        auto ptr2 = std::make_shared<SparseBitset>(allocator.get());
        auto ptr3 = std::make_shared<SparseBitset>(allocator.get());
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
        ComputableBitsetPtr ptr1 = std::make_shared<SparseBitset>(allocator.get());
        auto ptr2 = std::make_shared<SparseBitset>(allocator.get());
        auto ptr3 = std::make_shared<SparseBitset>(allocator.get());
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

TEST_CASE("Roaring Bitmap Test", "[ut][bitset]") {
    Roaring r1;
    for (uint32_t i = 100; i < 1000; i++) {
        r1.add(i);
    }

    // check whether a value is contained
    assert(r1.contains(500));

    // compute how many bits there are:
    uint32_t cardinality = r1.cardinality();

    // if your bitmaps have long runs, you can compress them by calling
    // run_optimize
    uint32_t size = r1.getSizeInBytes();
    r1.runOptimize();

    // you can enable "copy-on-write" for fast and shallow copies
    r1.setCopyOnWrite(true);

    uint32_t compact_size = r1.getSizeInBytes();
    // std::cout << "size before run optimize " << size << " bytes, and after " << compact_size
    //           << " bytes." << std::endl;

    // create a new bitmap with varargs
    Roaring r2 = Roaring::bitmapOf(5, 1, 2, 3, 5, 6);

    // r2.printf();
    // printf("\n");

    // create a new bitmap with initializer list
    Roaring r2i = Roaring::bitmapOfList({1, 2, 3, 5, 6});

    assert(r2i == r2);

    // we can also create a bitmap from a pointer to 32-bit integers
    const uint32_t values[] = {2, 3, 4};
    Roaring r3(3, values);

    // we can also go in reverse and go from arrays to bitmaps
    uint64_t card1 = r1.cardinality();
    uint32_t* arr1 = new uint32_t[card1];
    r1.toUint32Array(arr1);
    Roaring r1f(card1, arr1);
    delete[] arr1;

    // bitmaps shall be equal
    assert(r1 == r1f);

    // we can copy and compare bitmaps
    Roaring z(r3);
    assert(r3 == z);

    // we can compute union two-by-two
    Roaring r1_2_3 = r1 | r2;
    r1_2_3 |= r3;

    // we can compute a big union
    const Roaring* allmybitmaps[] = {&r1, &r2, &r3};
    Roaring bigunion = Roaring::fastunion(3, allmybitmaps);
    assert(r1_2_3 == bigunion);

    // we can compute intersection two-by-two
    Roaring i1_2 = r1 & r2;

    // we can write a bitmap to a pointer and recover it later
    uint32_t expectedsize = r1.getSizeInBytes();
    char* serializedbytes = new char[expectedsize];
    r1.write(serializedbytes);
    // readSafe will not overflow, but the resulting bitmap
    // is only valid and usable if the input follows the
    // Roaring specification: https://github.com/RoaringBitmap/RoaringFormatSpec/
    Roaring t = Roaring::readSafe(serializedbytes, expectedsize);
    assert(r1 == t);
    delete[] serializedbytes;

    // we can iterate over all values using custom functions
    uint32_t counter = 0;
    r1.iterate(
        [](uint32_t value, void* param) {
            *(uint32_t*)param += value;
            return true;
        },
        &counter);

    // we can also iterate the C++ way
    counter = 0;
    for (Roaring::const_iterator i = t.begin(); i != t.end(); i++) {
        ++counter;
    }
    // counter == t.cardinality()

    // we can move iterators to skip values
    const uint32_t manyvalues[] = {2, 3, 4, 7, 8};
    Roaring rogue(5, manyvalues);
    Roaring::const_iterator j = rogue.begin();
    j.equalorlarger(4);  // *j == 4
}
