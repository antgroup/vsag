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

#include "label_table.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <memory>
#include <sstream>

#include "datacell/dense_duplicate_tracker.h"
#include "impl/allocator/default_allocator.h"

using namespace vsag;

namespace {

auto
sorted_duplicates(std::vector<InnerIdType> ids) -> std::vector<InnerIdType> {
    std::sort(ids.begin(), ids.end());
    return ids;
}

}  // namespace

TEST_CASE("LabelTable Basic Operations", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get());

    SECTION("Insert and GetLabelById") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.Insert(2, 300);

        REQUIRE(label_table.GetLabelById(0) == 100);
        REQUIRE(label_table.GetLabelById(1) == 200);
        REQUIRE(label_table.GetLabelById(2) == 300);
    }

    SECTION("GetIdByLabel with reverse map") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.Insert(2, 300);

        REQUIRE(label_table.GetIdByLabel(100) == 0);
        REQUIRE(label_table.GetIdByLabel(200) == 1);
        REQUIRE(label_table.GetIdByLabel(300) == 2);
    }

    SECTION("CheckLabel") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        REQUIRE(label_table.CheckLabel(100) == true);
        REQUIRE(label_table.CheckLabel(200) == true);
        REQUIRE(label_table.CheckLabel(300) == false);
    }

    SECTION("MarkRemove and IsRemoved") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.Insert(2, 300);

        REQUIRE(label_table.CheckLabel(100) == true);
        label_table.MarkRemove(100);
        REQUIRE(label_table.IsRemoved(0) == true);
        REQUIRE(label_table.CheckLabel(100) == false);
    }

    SECTION("GetIdByLabel with removed label") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.MarkRemove(100);

        REQUIRE_THROWS_AS(label_table.GetIdByLabel(100), VsagException);
        REQUIRE(label_table.GetIdByLabel(100, true) == 0);  // return even removed
    }

    SECTION("TryGetIdByLabel") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        // Test successful lookup
        auto [success1, id1] = label_table.TryGetIdByLabel(100);
        REQUIRE(success1 == true);
        REQUIRE(id1 == 0);

        auto [success2, id2] = label_table.TryGetIdByLabel(200);
        REQUIRE(success2 == true);
        REQUIRE(id2 == 1);

        // Test non-existent label
        auto [success3, id3] = label_table.TryGetIdByLabel(999);
        REQUIRE(success3 == false);

        // Test removed label
        label_table.MarkRemove(100);
        auto [success4, id4] = label_table.TryGetIdByLabel(100);
        REQUIRE(success4 == false);

        // Test removed label with return_even_removed=true
        auto [success5, id5] = label_table.TryGetIdByLabel(100, true);
        REQUIRE(success5 == true);
        REQUIRE(id5 == 0);
    }

    SECTION("SetImmutable disables reverse map") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        REQUIRE(label_table.use_reverse_map_ == true);
        label_table.SetImmutable();
        REQUIRE(label_table.use_reverse_map_ == false);

        // Should still work but using linear search
        REQUIRE(label_table.GetIdByLabel(100) == 0);
        REQUIRE(label_table.GetIdByLabel(200) == 1);
    }
}

TEST_CASE("LabelTable Without Reverse Map", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get(), false);  // disable reverse map

    SECTION("Insert and GetIdByLabel without reverse map") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.Insert(2, 300);

        REQUIRE(label_table.GetIdByLabel(100) == 0);
        REQUIRE(label_table.GetIdByLabel(200) == 1);
        REQUIRE(label_table.GetIdByLabel(300) == 2);
    }

    SECTION("TryGetIdByLabel without reverse map") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        // Test successful lookup
        auto [success1, id1] = label_table.TryGetIdByLabel(100);
        REQUIRE(success1 == true);
        REQUIRE(id1 == 0);

        // Test non-existent label
        auto [success2, id2] = label_table.TryGetIdByLabel(999);
        REQUIRE(success2 == false);
    }
}

TEST_CASE("LabelTable Memory Management", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get());

    SECTION("GetTotalCount") {
        REQUIRE(label_table.GetTotalCount() == 0);

        label_table.Insert(0, 100);
        REQUIRE(label_table.GetTotalCount() == 1);

        label_table.Insert(1, 200);
        REQUIRE(label_table.GetTotalCount() == 2);
    }

    SECTION("Resize") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        label_table.Resize(10);
        REQUIRE(label_table.GetTotalCount() == 2);

        // Should be able to insert at new positions
        label_table.Insert(9, 900);
        REQUIRE(label_table.GetLabelById(9) == 900);
    }

    SECTION("GetMemoryUsage") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);

        auto usage = label_table.GetMemoryUsage();
        REQUIRE(usage > 0);
    }
}

TEST_CASE("LabelTable Filter Operations", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get());

    SECTION("GetDeletedIdsFilter with no deletions") {
        auto filter = label_table.GetDeletedIdsFilter();
        REQUIRE(filter == nullptr);
    }

    SECTION("GetDeletedIdsFilter with deletions") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 200);
        label_table.MarkRemove(100);

        auto filter = label_table.GetDeletedIdsFilter();
        REQUIRE(filter != nullptr);
    }
}

TEST_CASE("LabelTable Edge Cases", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get());

    SECTION("GetLabelById with invalid id") {
        label_table.Insert(0, 100);

        REQUIRE_THROWS_AS(label_table.GetLabelById(1), VsagException);
        REQUIRE_THROWS_AS(label_table.GetLabelById(1000), VsagException);
    }

    SECTION("GetIdByLabel with non-existent label") {
        REQUIRE_THROWS_AS(label_table.GetIdByLabel(999), VsagException);
    }

    SECTION("Insert at large id") {
        label_table.Insert(1000, 5000);
        REQUIRE(label_table.GetLabelById(1000) == 5000);
        REQUIRE(label_table.GetIdByLabel(5000) == 1000);
    }
}

TEST_CASE("LabelTable Serialize and Deserialize preserves total count", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("Serialize and Deserialize preserves total count") {
        auto label_table = std::make_shared<LabelTable>(allocator.get(), true, true);
        label_table->Resize(3);
        label_table->Insert(0, 100);
        label_table->Insert(1, 200);
        label_table->Insert(2, 300);

        auto count_before = label_table->GetTotalCount();

        std::stringstream ss;
        vsag::IOStreamWriter writer(ss);
        label_table->Serialize(writer);

        auto new_label_table = std::make_shared<LabelTable>(allocator.get(), true, true);
        vsag::IOStreamReader reader(ss);
        new_label_table->Deserialize(reader);

        REQUIRE(new_label_table->GetTotalCount() == count_before);
    }
}

TEST_CASE("LabelTable deserializes legacy duplicate payload", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    auto label_table = std::make_shared<LabelTable>(allocator.get(), true, true);
    auto tracker = std::make_shared<DenseDuplicateTracker>(allocator.get());
    label_table->SetDuplicateTracker(tracker);
    label_table->is_legacy_duplicate_format_ = true;

    std::stringstream ss;
    IOStreamWriter writer(ss);

    std::vector<LabelType> labels{100, 200, 300, 400, 500, 600};
    StreamWriter::WriteVector(writer, labels);

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

    IOStreamReader reader(ss);
    label_table->Deserialize(reader);

    REQUIRE(label_table->GetTotalCount() == static_cast<int64_t>(labels.size()));
    REQUIRE(label_table->GetIdByLabel(500) == 4);
    REQUIRE(label_table->is_legacy_duplicate_format_ == false);
    REQUIRE(sorted_duplicates(tracker->GetDuplicateIds(0)) == std::vector<InnerIdType>{1, 2});
    REQUIRE(sorted_duplicates(tracker->GetDuplicateIds(2)) == std::vector<InnerIdType>{0, 1});
    REQUIRE(sorted_duplicates(tracker->GetDuplicateIds(4)) == std::vector<InnerIdType>{5});
    REQUIRE(tracker->GetGroupId(0) == 0);
    REQUIRE(tracker->GetGroupId(2) == 0);
    REQUIRE(tracker->GetGroupId(5) == 4);
}

TEST_CASE("LabelTable Hole List Operations", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get());

    SECTION("PushHole and PopHole basic operations") {
        REQUIRE(label_table.HasHole() == false);
        REQUIRE(label_table.GetHoleCount() == 0);

        label_table.PushHole(5);
        REQUIRE(label_table.HasHole() == true);
        REQUIRE(label_table.GetHoleCount() == 1);

        auto [success, id] = label_table.PopHole();
        REQUIRE(success == true);
        REQUIRE(id == 5);
        REQUIRE(label_table.HasHole() == false);
    }

    SECTION("PopHole returns false when empty") {
        auto [success, id] = label_table.PopHole();
        REQUIRE(success == false);
        REQUIRE(id == 0);
    }

    SECTION("Multiple Push and Pop (LIFO order)") {
        label_table.PushHole(1);
        label_table.PushHole(2);
        label_table.PushHole(3);
        REQUIRE(label_table.GetHoleCount() == 3);

        auto [s1, id1] = label_table.PopHole();
        REQUIRE(s1 == true);
        REQUIRE(id1 == 3);

        auto [s2, id2] = label_table.PopHole();
        REQUIRE(s2 == true);
        REQUIRE(id2 == 2);

        auto [s3, id3] = label_table.PopHole();
        REQUIRE(s3 == true);
        REQUIRE(id3 == 1);
    }

    SECTION("RemoveHole removes specific id") {
        label_table.PushHole(1);
        label_table.PushHole(2);
        label_table.PushHole(3);
        REQUIRE(label_table.GetHoleCount() == 3);

        bool removed = label_table.RemoveHole(2);
        REQUIRE(removed == true);
        REQUIRE(label_table.GetHoleCount() == 2);

        // RemoveHole for non-existent id returns false
        removed = label_table.RemoveHole(99);
        REQUIRE(removed == false);
        REQUIRE(label_table.GetHoleCount() == 2);
    }
}

TEST_CASE("LabelTable Hole List Serialization", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("Serialize and Deserialize with holes") {
        auto label_table = std::make_shared<LabelTable>(allocator.get());
        label_table->Insert(0, 100);
        label_table->Insert(1, 200);
        label_table->Insert(2, 300);

        label_table->PushHole(0);
        label_table->PushHole(1);
        REQUIRE(label_table->GetHoleCount() == 2);

        std::stringstream ss;
        IOStreamWriter writer(ss);
        label_table->Serialize(writer);

        auto new_label_table = std::make_shared<LabelTable>(allocator.get());
        IOStreamReader reader(ss);
        new_label_table->Deserialize(reader);

        REQUIRE(new_label_table->GetHoleCount() == 0);
        auto [s1, id1] = new_label_table->PopHole();
        REQUIRE(s1 == false);
    }
}
