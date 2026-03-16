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

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <memory>

#include "impl/allocator/default_allocator.h"

using namespace vsag;

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

TEST_CASE("LabelTable Duplicate ID Operations", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("SetDuplicateId and GetDuplicateId with single duplicate") {
        // Setup: Insert two labels with same label value (simulating duplicate)
        LabelTable label_table(allocator.get(), true, true);  // enable compress_duplicate_data
        label_table.Resize(2);
        label_table.Insert(0, 100);
        label_table.Insert(1, 100);
        label_table.SetDuplicateId(0, 1);

        auto duplicates = label_table.GetDuplicateId(0);
        REQUIRE(duplicates.size() == 1);
        REQUIRE(duplicates.count(1) == 1);
    }

    SECTION("SetDuplicateId and GetDuplicateId with multiple duplicates") {
        // Setup: Insert multiple labels with same label value
        LabelTable label_table(allocator.get(), true, true);  // enable compress_duplicate_data
        label_table.Resize(4);
        label_table.Insert(0, 100);
        label_table.Insert(1, 100);
        label_table.Insert(2, 100);
        label_table.Insert(3, 100);

        label_table.SetDuplicateId(0, 1);
        label_table.SetDuplicateId(0, 2);
        label_table.SetDuplicateId(0, 3);

        auto duplicates = label_table.GetDuplicateId(0);
        REQUIRE(duplicates.size() == 3);
        REQUIRE(duplicates.count(1) == 1);
        REQUIRE(duplicates.count(2) == 1);
        REQUIRE(duplicates.count(3) == 1);
    }

    SECTION("GetDuplicateId returns empty set for ID without duplicates") {
        LabelTable label_table(allocator.get(), true, true);  // enable compress_duplicate_data
        label_table.Resize(1);
        label_table.Insert(0, 100);

        auto duplicates = label_table.GetDuplicateId(0);
        REQUIRE(duplicates.empty());
    }

    SECTION("Multiple independent duplicate groups") {
        LabelTable label_table(allocator.get(), true, true);  // enable compress_duplicate_data
        label_table.Resize(5);
        // Group 1: IDs 0, 1, 2 share label 100
        label_table.Insert(0, 100);
        label_table.Insert(1, 100);
        label_table.Insert(2, 100);

        // Group 2: IDs 3, 4 share label 200
        label_table.Insert(3, 200);
        label_table.Insert(4, 200);

        label_table.SetDuplicateId(0, 1);
        label_table.SetDuplicateId(0, 2);
        label_table.SetDuplicateId(3, 4);

        auto group1 = label_table.GetDuplicateId(0);
        REQUIRE(group1.size() == 2);
        REQUIRE(group1.count(1) == 1);
        REQUIRE(group1.count(2) == 1);

        auto group2 = label_table.GetDuplicateId(3);
        REQUIRE(group2.size() == 1);
        REQUIRE(group2.count(4) == 1);
    }
}

TEST_CASE("LabelTable Serialize and Deserialize with Duplicates", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("Serialize and Deserialize with duplicate IDs") {
        // Create and populate label table with duplicates
        auto label_table = std::make_shared<LabelTable>(allocator.get(), true, true);
        label_table->Resize(5);
        label_table->Insert(0, 100);
        label_table->Insert(1, 100);
        label_table->Insert(2, 100);
        label_table->Insert(3, 200);
        label_table->Insert(4, 200);

        label_table->SetDuplicateId(0, 1);
        label_table->SetDuplicateId(0, 2);
        label_table->SetDuplicateId(3, 4);

        // Serialize
        std::stringstream ss;
        vsag::IOStreamWriter writer(ss);
        label_table->Serialize(writer);

        // Deserialize into new label table
        auto new_label_table = std::make_shared<LabelTable>(allocator.get(), true, true);
        vsag::IOStreamReader reader(ss);
        new_label_table->Deserialize(reader);

        // Verify labels are preserved
        REQUIRE(new_label_table->GetLabelById(0) == 100);
        REQUIRE(new_label_table->GetLabelById(1) == 100);
        REQUIRE(new_label_table->GetLabelById(2) == 100);
        REQUIRE(new_label_table->GetLabelById(3) == 200);
        REQUIRE(new_label_table->GetLabelById(4) == 200);

        // Verify duplicates are preserved
        auto group1 = new_label_table->GetDuplicateId(0);
        REQUIRE(group1.size() == 2);
        REQUIRE(group1.count(1) == 1);
        REQUIRE(group1.count(2) == 1);

        auto group2 = new_label_table->GetDuplicateId(3);
        REQUIRE(group2.size() == 1);
        REQUIRE(group2.count(4) == 1);
    }

    SECTION("Serialize and Deserialize without duplicates") {
        auto label_table = std::make_shared<LabelTable>(allocator.get(), true, true);
        label_table->Resize(3);
        label_table->Insert(0, 100);
        label_table->Insert(1, 200);
        label_table->Insert(2, 300);

        std::stringstream ss;
        vsag::IOStreamWriter writer(ss);
        label_table->Serialize(writer);

        auto new_label_table = std::make_shared<LabelTable>(allocator.get(), true, true);
        vsag::IOStreamReader reader(ss);
        new_label_table->Deserialize(reader);

        REQUIRE(new_label_table->GetLabelById(0) == 100);
        REQUIRE(new_label_table->GetLabelById(1) == 200);
        REQUIRE(new_label_table->GetLabelById(2) == 300);

        REQUIRE(new_label_table->GetDuplicateId(0).empty());
        REQUIRE(new_label_table->GetDuplicateId(1).empty());
        REQUIRE(new_label_table->GetDuplicateId(2).empty());
    }

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

TEST_CASE("LabelTable Duplicate ID with Resize", "[ut][LabelTable]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    LabelTable label_table(allocator.get(), true, true);
    label_table.Resize(2);

    SECTION("Resize preserves duplicate information") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 100);
        label_table.SetDuplicateId(0, 1);

        label_table.Resize(100);

        auto duplicates = label_table.GetDuplicateId(0);
        REQUIRE(duplicates.size() == 1);
        REQUIRE(duplicates.count(1) == 1);

        // Verify we can still insert at new positions
        label_table.Insert(50, 500);
        REQUIRE(label_table.GetLabelById(50) == 500);
    }

    SECTION("Resize and add new duplicates") {
        label_table.Insert(0, 100);
        label_table.Insert(1, 100);
        label_table.SetDuplicateId(0, 1);

        label_table.Resize(10);

        // Add new entries and create another duplicate group
        label_table.Insert(5, 500);
        label_table.Insert(6, 500);
        label_table.SetDuplicateId(5, 6);

        auto group1 = label_table.GetDuplicateId(0);
        REQUIRE(group1.size() == 1);
        REQUIRE(group1.count(1) == 1);

        auto group2 = label_table.GetDuplicateId(5);
        REQUIRE(group2.size() == 1);
        REQUIRE(group2.count(6) == 1);
    }
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
