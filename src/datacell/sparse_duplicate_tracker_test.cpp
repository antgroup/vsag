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

#include "sparse_duplicate_tracker.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <sstream>

#include "impl/allocator/default_allocator.h"
#include "unittest.h"
using namespace vsag;

namespace {

auto
sorted_duplicates(std::vector<InnerIdType> ids) -> std::vector<InnerIdType> {
    std::sort(ids.begin(), ids.end());
    return ids;
}

}  // namespace

TEST_CASE("SparseDuplicateTracker tracks duplicate groups", "[ut][SparseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    SparseDuplicateTracker tracker(allocator.get());

    tracker.SetDuplicateId(0, 1);
    tracker.SetDuplicateId(0, 2);
    tracker.SetDuplicateId(0, 3);
    tracker.SetDuplicateId(4, 5);

    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(0)) == std::vector<InnerIdType>{1, 2, 3});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(1)) == std::vector<InnerIdType>{0, 2, 3});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(2)) == std::vector<InnerIdType>{0, 1, 3});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(3)) == std::vector<InnerIdType>{0, 1, 2});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(4)) == std::vector<InnerIdType>{5});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(5)) == std::vector<InnerIdType>{4});
    REQUIRE(tracker.GetGroupId(0) == 0);
    REQUIRE(tracker.GetGroupId(1) == 0);
    REQUIRE(tracker.GetGroupId(2) == 0);
    REQUIRE(tracker.GetGroupId(3) == 0);
    REQUIRE(tracker.GetGroupId(5) == 4);
    REQUIRE(tracker.GetGroupId(7) == 7);
}

TEST_CASE("SparseDuplicateTracker requires canonical group ids", "[ut][SparseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    SparseDuplicateTracker tracker(allocator.get());

    tracker.SetDuplicateId(0, 1);
    tracker.SetDuplicateId(0, 2);
    tracker.SetDuplicateId(0, 3);

    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(0)) == std::vector<InnerIdType>{1, 2, 3});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(2)) == std::vector<InnerIdType>{0, 1, 3});
    REQUIRE(tracker.GetGroupId(3) == 0);
}

TEST_CASE("SparseDuplicateTracker ignores duplicate reinsertion", "[ut][SparseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    SparseDuplicateTracker tracker(allocator.get());

    tracker.SetDuplicateId(0, 1);
    tracker.SetDuplicateId(0, 1);
    tracker.SetDuplicateId(2, 1);

    REQUIRE(tracker.GetDuplicateIds(0) == std::vector<InnerIdType>{1});
}

TEST_CASE("SparseDuplicateTracker serialize and deserialize", "[ut][SparseDuplicateTracker]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    SparseDuplicateTracker tracker(allocator.get());
    tracker.SetDuplicateId(0, 1);
    tracker.SetDuplicateId(0, 2);
    tracker.SetDuplicateId(4, 5);

    std::stringstream ss;
    IOStreamWriter writer(ss);
    tracker.Serialize(writer);

    SparseDuplicateTracker restored(allocator.get());
    IOStreamReader reader(ss);
    restored.Deserialize(reader);

    REQUIRE(sorted_duplicates(restored.GetDuplicateIds(0)) == std::vector<InnerIdType>{1, 2});
    REQUIRE(sorted_duplicates(restored.GetDuplicateIds(1)) == std::vector<InnerIdType>{0, 2});
    REQUIRE(sorted_duplicates(restored.GetDuplicateIds(4)) == std::vector<InnerIdType>{5});
    REQUIRE(restored.GetGroupId(2) == 0);
    REQUIRE(restored.GetGroupId(5) == 4);
}

TEST_CASE("SparseDuplicateTracker deserializes legacy format", "[ut][SparseDuplicateTracker]") {
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

    SparseDuplicateTracker tracker(allocator.get());
    IOStreamReader reader(ss);
    tracker.DeserializeFromLegacyFormat(reader, 6);

    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(0)) == std::vector<InnerIdType>{1, 2});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(2)) == std::vector<InnerIdType>{0, 1});
    REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(4)) == std::vector<InnerIdType>{5});
    REQUIRE(tracker.GetGroupId(0) == 0);
    REQUIRE(tracker.GetGroupId(2) == 0);
    REQUIRE(tracker.GetGroupId(5) == 4);
}
