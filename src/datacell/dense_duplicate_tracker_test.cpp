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

#include "dense_duplicate_tracker.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <sstream>

#include "impl/allocator/default_allocator.h"
#include "unittest.h"
#include "vsag_exception.h"
using namespace vsag;

namespace {

auto
sorted_duplicates(std::vector<InnerIdType> ids) -> std::vector<InnerIdType> {
    std::sort(ids.begin(), ids.end());
    return ids;
}

}  // namespace

TEST_CASE("DenseDuplicateTracker tracks duplicate groups", "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    DenseDuplicateTracker tracker(allocator.get());
    tracker.Resize(8);

    tracker.SetDuplicateId(0, 1);
    tracker.SetDuplicateId(0, 2);
    tracker.SetDuplicateId(4, 5);

    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(0)) == std::vector<InnerIdType>{1, 2});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(1)) == std::vector<InnerIdType>{0, 2});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(2)) == std::vector<InnerIdType>{0, 1});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(4)) == std::vector<InnerIdType>{5});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(5)) == std::vector<InnerIdType>{4});
    REQUIRE(tracker.GetGroupId(0) == 0);
    REQUIRE(tracker.GetGroupId(1) == 0);
    REQUIRE(tracker.GetGroupId(2) == 0);
    REQUIRE(tracker.GetGroupId(4) == 4);
    REQUIRE(tracker.GetGroupId(5) == 4);
    REQUIRE(tracker.GetGroupId(7) == 7);
}

TEST_CASE("DenseDuplicateTracker ignores duplicate reinsertion", "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    DenseDuplicateTracker tracker(allocator.get());
    tracker.Resize(4);

    tracker.SetDuplicateId(0, 1);
    tracker.SetDuplicateId(0, 1);
    tracker.SetDuplicateId(1, 1);

    REQUIRE(tracker.GetDuplicateIds(0) == std::vector<InnerIdType>{1});
}

TEST_CASE("DenseDuplicateTracker serialize and deserialize", "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    DenseDuplicateTracker tracker(allocator.get());
    tracker.Resize(8);
    tracker.SetDuplicateId(0, 1);
    tracker.SetDuplicateId(0, 2);
    tracker.SetDuplicateId(4, 5);

    std::stringstream ss;
    IOStreamWriter writer(ss);
    tracker.Serialize(writer);

    DenseDuplicateTracker restored(allocator.get());
    IOStreamReader reader(ss);
    restored.Deserialize(reader);

    REQUIRE(sorted_duplicates(restored.GetDuplicateIds(0)) == std::vector<InnerIdType>{1, 2});
    REQUIRE(sorted_duplicates(restored.GetDuplicateIds(1)) == std::vector<InnerIdType>{0, 2});
    REQUIRE(sorted_duplicates(restored.GetDuplicateIds(4)) == std::vector<InnerIdType>{5});
    REQUIRE(restored.GetGroupId(2) == 0);
    REQUIRE(restored.GetGroupId(5) == 4);
}

TEST_CASE("DenseDuplicateTracker resize initializes new ids", "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    DenseDuplicateTracker tracker(allocator.get());
    tracker.Resize(2);
    tracker.SetDuplicateId(0, 1);

    tracker.Resize(6);
    tracker.SetDuplicateId(4, 5);

    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(0)) == std::vector<InnerIdType>{1});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(4)) == std::vector<InnerIdType>{5});
}

TEST_CASE("DenseDuplicateTracker deserializes legacy format", "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    std::stringstream ss;
    IOStreamWriter writer(ss);
    size_t duplicate_count = 2;
    StreamWriter::WriteObj(writer, duplicate_count);

    InnerIdType head0 = 0;
    std::vector<InnerIdType> group0{1, 2};
    StreamWriter::WriteObj(writer, head0);
    StreamWriter::WriteVector(writer, group0);

    InnerIdType head1 = 4;
    std::vector<InnerIdType> group1{5};
    StreamWriter::WriteObj(writer, head1);
    StreamWriter::WriteVector(writer, group1);

    DenseDuplicateTracker tracker(allocator.get());
    IOStreamReader reader(ss);
    tracker.DeserializeFromLegacyFormat(reader, 6);

    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(0)) == std::vector<InnerIdType>{1, 2});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(2)) == std::vector<InnerIdType>{0, 1});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(4)) == std::vector<InnerIdType>{5});
    REQUIRE(tracker.GetGroupId(0) == 0);
    REQUIRE(tracker.GetGroupId(2) == 0);
    REQUIRE(tracker.GetGroupId(5) == 4);
}

TEST_CASE("DenseDuplicateTracker deserialize rejects out-of-range ids",
          "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("head_id out of range") {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        size_t duplicate_count = 1;
        size_t size = 4;
        StreamWriter::WriteObj(writer, duplicate_count);
        StreamWriter::WriteObj(writer, size);

        InnerIdType head_id = 10;  // >= size
        StreamWriter::WriteObj(writer, head_id);
        std::vector<InnerIdType> id_list{1};
        StreamWriter::WriteVector(writer, id_list);

        DenseDuplicateTracker tracker(allocator.get());
        IOStreamReader reader(ss);
        REQUIRE_THROWS_AS(tracker.Deserialize(reader), VsagException);
    }

    SECTION("dup_id out of range") {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        size_t duplicate_count = 1;
        size_t size = 4;
        StreamWriter::WriteObj(writer, duplicate_count);
        StreamWriter::WriteObj(writer, size);

        InnerIdType head_id = 0;
        StreamWriter::WriteObj(writer, head_id);
        std::vector<InnerIdType> id_list{99};  // >= size
        StreamWriter::WriteVector(writer, id_list);

        DenseDuplicateTracker tracker(allocator.get());
        IOStreamReader reader(ss);
        REQUIRE_THROWS_AS(tracker.Deserialize(reader), VsagException);
    }
}

TEST_CASE("DenseDuplicateTracker legacy deserialize rejects out-of-range ids",
          "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("id out of range") {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        size_t duplicate_count = 1;
        StreamWriter::WriteObj(writer, duplicate_count);

        InnerIdType id = 10;  // >= total_size (6)
        StreamWriter::WriteObj(writer, id);
        std::vector<InnerIdType> id_list{1};
        StreamWriter::WriteVector(writer, id_list);

        DenseDuplicateTracker tracker(allocator.get());
        IOStreamReader reader(ss);
        REQUIRE_THROWS_AS(tracker.DeserializeFromLegacyFormat(reader, 6), VsagException);
    }

    SECTION("duplicate_id out of range") {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        size_t duplicate_count = 1;
        StreamWriter::WriteObj(writer, duplicate_count);

        InnerIdType id = 0;
        StreamWriter::WriteObj(writer, id);
        std::vector<InnerIdType> id_list{99};  // >= total_size (6)
        StreamWriter::WriteVector(writer, id_list);

        DenseDuplicateTracker tracker(allocator.get());
        IOStreamReader reader(ss);
        REQUIRE_THROWS_AS(tracker.DeserializeFromLegacyFormat(reader, 6), VsagException);
    }
}

TEST_CASE("DenseDuplicateTracker deserialize rejects overlapping groups",
          "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("duplicate head_id across groups") {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        size_t duplicate_count = 2;
        size_t size = 6;
        StreamWriter::WriteObj(writer, duplicate_count);
        StreamWriter::WriteObj(writer, size);

        // group 1: head=0, members=[1]
        InnerIdType head0 = 0;
        StreamWriter::WriteObj(writer, head0);
        std::vector<InnerIdType> group0{1};
        StreamWriter::WriteVector(writer, group0);

        // group 2: head=0 again — should be rejected
        InnerIdType head1 = 0;
        StreamWriter::WriteObj(writer, head1);
        std::vector<InnerIdType> group1{2};
        StreamWriter::WriteVector(writer, group1);

        DenseDuplicateTracker tracker(allocator.get());
        IOStreamReader reader(ss);
        REQUIRE_THROWS_AS(tracker.Deserialize(reader), VsagException);
    }
}

TEST_CASE("DenseDuplicateTracker legacy deserialize rejects overlapping groups",
          "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("duplicate id across groups") {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        size_t duplicate_count = 2;
        StreamWriter::WriteObj(writer, duplicate_count);

        InnerIdType id0 = 0;
        StreamWriter::WriteObj(writer, id0);
        std::vector<InnerIdType> group0{1};
        StreamWriter::WriteVector(writer, group0);

        InnerIdType id1 = 0;  // already used
        StreamWriter::WriteObj(writer, id1);
        std::vector<InnerIdType> group1{2};
        StreamWriter::WriteVector(writer, group1);

        DenseDuplicateTracker tracker(allocator.get());
        IOStreamReader reader(ss);
        REQUIRE_THROWS_AS(tracker.DeserializeFromLegacyFormat(reader, 6), VsagException);
    }
}

TEST_CASE("DenseDuplicateTracker deserialize rejects duplicate member ids",
          "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("dup_id already used in another group") {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        size_t duplicate_count = 2;
        size_t size = 6;
        StreamWriter::WriteObj(writer, duplicate_count);
        StreamWriter::WriteObj(writer, size);

        // group 1: head=0, members=[1]
        InnerIdType head0 = 0;
        StreamWriter::WriteObj(writer, head0);
        std::vector<InnerIdType> group0{1};
        StreamWriter::WriteVector(writer, group0);

        // group 2: head=2, members=[1] — id 1 already used
        InnerIdType head1 = 2;
        StreamWriter::WriteObj(writer, head1);
        std::vector<InnerIdType> group1{1};
        StreamWriter::WriteVector(writer, group1);

        DenseDuplicateTracker tracker(allocator.get());
        IOStreamReader reader(ss);
        REQUIRE_THROWS_AS(tracker.Deserialize(reader), VsagException);
    }
}

TEST_CASE("DenseDuplicateTracker legacy deserialize rejects duplicate member ids",
          "[ut][DenseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("duplicate_id already used in another group") {
        std::stringstream ss;
        IOStreamWriter writer(ss);
        size_t duplicate_count = 2;
        StreamWriter::WriteObj(writer, duplicate_count);

        InnerIdType id0 = 0;
        StreamWriter::WriteObj(writer, id0);
        std::vector<InnerIdType> group0{1};
        StreamWriter::WriteVector(writer, group0);

        InnerIdType id1 = 2;
        StreamWriter::WriteObj(writer, id1);
        std::vector<InnerIdType> group1{1};  // already used
        StreamWriter::WriteVector(writer, group1);

        DenseDuplicateTracker tracker(allocator.get());
        IOStreamReader reader(ss);
        REQUIRE_THROWS_AS(tracker.DeserializeFromLegacyFormat(reader, 6), VsagException);
    }
}
