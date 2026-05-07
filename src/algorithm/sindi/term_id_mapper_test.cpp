
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

#include "term_id_mapper.h"

#include <random>
#include <set>

#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"
#include "vsag_exception.h"

using namespace vsag;

TEST_CASE("TermIdMapper Basic Mapping", "[ut][TermIdMapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TermIdMapper mapper(1000, allocator.get());

    SECTION("first-come-first-served assignment") {
        REQUIRE(mapper.Map(50000) == 0);
        REQUIRE(mapper.Map(12) == 1);
        REQUIRE(mapper.Map(999999) == 2);
        REQUIRE(mapper.Map(42) == 3);
        REQUIRE(mapper.Size() == 4);
    }

    SECTION("duplicate returns existing compact ID") {
        REQUIRE(mapper.Map(50000) == 0);
        REQUIRE(mapper.Map(12) == 1);
        REQUIRE(mapper.Map(50000) == 0);  // duplicate
        REQUIRE(mapper.Map(12) == 1);     // duplicate
        REQUIRE(mapper.Size() == 2);
    }

    SECTION("sequential input maps to same IDs") {
        REQUIRE(mapper.Map(0) == 0);
        REQUIRE(mapper.Map(1) == 1);
        REQUIRE(mapper.Map(2) == 2);
        REQUIRE(mapper.Map(3) == 3);
        REQUIRE(mapper.Size() == 4);
    }
}

TEST_CASE("TermIdMapper Empty", "[ut][TermIdMapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TermIdMapper mapper(100, allocator.get());
    REQUIRE(mapper.Size() == 0);
}

TEST_CASE("TermIdMapper TryMap", "[ut][TermIdMapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TermIdMapper mapper(100, allocator.get());

    mapper.Map(100);
    mapper.Map(200);

    SECTION("known term returns compact ID") {
        auto result = mapper.TryMap(100);
        REQUIRE(result.has_value());
        REQUIRE(result.value() == 0);

        result = mapper.TryMap(200);
        REQUIRE(result.has_value());
        REQUIRE(result.value() == 1);
    }

    SECTION("unknown term returns nullopt") {
        auto result = mapper.TryMap(999);
        REQUIRE_FALSE(result.has_value());
    }
}

TEST_CASE("TermIdMapper ReverseMap", "[ut][TermIdMapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TermIdMapper mapper(100, allocator.get());

    mapper.Map(50000);
    mapper.Map(12);
    mapper.Map(999999);

    REQUIRE(mapper.ReverseMap(0) == 50000);
    REQUIRE(mapper.ReverseMap(1) == 12);
    REQUIRE(mapper.ReverseMap(2) == 999999);

    REQUIRE_THROWS_AS(mapper.ReverseMap(3), VsagException);
}

TEST_CASE("TermIdMapper Limit Reached", "[ut][TermIdMapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TermIdMapper mapper(3, allocator.get());

    REQUIRE(mapper.Map(10) == 0);
    REQUIRE(mapper.Map(20) == 1);
    REQUIRE(mapper.Map(30) == 2);

    // Limit reached, next new term should throw
    REQUIRE_THROWS_AS(mapper.Map(40), VsagException);

    // Existing terms still work
    REQUIRE(mapper.Map(10) == 0);
    REQUIRE(mapper.Map(20) == 1);
    REQUIRE(mapper.Size() == 3);
}

TEST_CASE("TermIdMapper Serialize and Deserialize", "[ut][TermIdMapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    // Build a mapper with some data
    TermIdMapper mapper1(1000, allocator.get());
    mapper1.Map(50000);
    mapper1.Map(12);
    mapper1.Map(999999);
    mapper1.Map(42);

    // Serialize to buffer
    std::vector<char> buffer(1024 * 1024);
    BufferStreamWriter writer(buffer.data());
    mapper1.Serialize(writer);

    // Deserialize into a new mapper
    TermIdMapper mapper2(1000, allocator.get());
    auto reader_func = [&buffer](uint64_t offset, uint64_t size, void* dest) {
        memcpy(dest, buffer.data() + offset, size);
    };
    ReadFuncStreamReader reader(reader_func, 0, writer.GetCursor());
    mapper2.Deserialize(reader);

    REQUIRE(mapper2.Size() == mapper1.Size());
    REQUIRE(mapper2.TryMap(50000).value() == 0);
    REQUIRE(mapper2.TryMap(12).value() == 1);
    REQUIRE(mapper2.TryMap(999999).value() == 2);
    REQUIRE(mapper2.TryMap(42).value() == 3);
    REQUIRE(mapper2.ReverseMap(0) == 50000);
    REQUIRE(mapper2.ReverseMap(1) == 12);
    REQUIRE(mapper2.ReverseMap(2) == 999999);
    REQUIRE(mapper2.ReverseMap(3) == 42);
}

TEST_CASE("TermIdMapper Deserialize Then Continue Add", "[ut][TermIdMapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    TermIdMapper mapper1(1000, allocator.get());
    mapper1.Map(100);
    mapper1.Map(200);

    // Serialize
    std::vector<char> buffer(1024 * 1024);
    BufferStreamWriter writer(buffer.data());
    mapper1.Serialize(writer);

    // Deserialize
    TermIdMapper mapper2(1000, allocator.get());
    auto reader_func = [&buffer](uint64_t offset, uint64_t size, void* dest) {
        memcpy(dest, buffer.data() + offset, size);
    };
    ReadFuncStreamReader reader(reader_func, 0, writer.GetCursor());
    mapper2.Deserialize(reader);

    // Continue adding new terms — should get next sequential IDs
    REQUIRE(mapper2.Map(300) == 2);
    REQUIRE(mapper2.Map(400) == 3);
    REQUIRE(mapper2.Size() == 4);

    // Old terms still work
    REQUIRE(mapper2.Map(100) == 0);
    REQUIRE(mapper2.Map(200) == 1);
}

TEST_CASE("TermIdMapper Large Scale", "[ut][TermIdMapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    TermIdMapper mapper(200000, allocator.get());

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX / 2);
    std::set<uint32_t> unique_terms;

    // Insert 100K random term IDs
    for (int i = 0; i < 100000; ++i) {
        uint32_t term = dist(rng);
        unique_terms.insert(term);
        mapper.Map(term);
    }

    REQUIRE(mapper.Size() == static_cast<uint32_t>(unique_terms.size()));

    // Verify all mapped terms can be looked up
    for (auto term : unique_terms) {
        auto result = mapper.TryMap(term);
        REQUIRE(result.has_value());
        REQUIRE(mapper.ReverseMap(result.value()) == term);
    }
}
