# Duplicate Tracker Ring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the ring-based duplicate tracker redesign with `GetGroupId(...)`, canonical `group_id` semantics, and sparse single-map storage while preserving `DeserializeFromLegacyFormat(...)` compatibility.

**Architecture:** Keep `DenseDuplicateTracker` and `SparseDuplicateTracker` as separate classes but align them on the same ring semantics. Add `GetGroupId(...)` to the duplicate tracker abstraction and graph wrapper, update sparse storage to a single `next` map, and preserve behavior through unit and wrapper tests before touching the production implementation.

**Tech Stack:** C++, Catch2, VSAG allocator-aware containers, GNU Make / CMake

---

### Task 1: Lock In The New Contract With Tests

**Files:**
- Modify: `src/datacell/dense_duplicate_tracker_test.cpp`
- Modify: `src/datacell/sparse_duplicate_tracker_test.cpp`
- Modify: `src/datacell/graph_datacell_test.cpp`
- Modify: `src/datacell/compressed_graph_datacell_test.cpp`

- [ ] **Step 1: Write failing tests for `GetGroupId(...)` and canonical `group_id` semantics**

```cpp
REQUIRE(tracker.GetGroupId(0) == 0);
REQUIRE(tracker.GetGroupId(1) == 0);
REQUIRE(tracker.GetGroupId(7) == 7);
```

```cpp
tracker.SetDuplicateId(0, 1);
tracker.SetDuplicateId(0, 2);
tracker.SetDuplicateId(0, 3);
REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(2)) == std::vector<InnerIdType>{0, 1, 3});
REQUIRE(tracker.GetGroupId(3) == 0);
```

```cpp
enabled_graph->SetDuplicateId(0, 1);
REQUIRE(enabled_graph->GetGroupId(0) == 0);
REQUIRE(enabled_graph->GetGroupId(1) == 0);
REQUIRE(disabled_graph->GetGroupId(5) == 5);
```

- [ ] **Step 2: Add a failing sparse legacy-compatibility test**

```cpp
SparseDuplicateTracker tracker(allocator.get());
IOStreamReader reader(ss);
tracker.DeserializeFromLegacyFormat(reader, 6);

REQUIRE(sorted_duplicates(tracker.GetDuplicateIds(2)) == std::vector<InnerIdType>{0, 1});
REQUIRE(tracker.GetGroupId(2) == 0);
```

- [ ] **Step 3: Run the targeted unit tests and verify they fail for the expected reason**

Run: `cmake --build build-release --target unittests && ./build-release/tests/unittests "[DenseDuplicateTracker]" "[SparseDuplicateTracker]" "[GraphDataCell]" "[CompressedGraphDatacell]"`

Expected: compile failure or test failure because `GetGroupId(...)` is not implemented yet and sparse tests still encode the old API contract.

### Task 2: Add The New Duplicate Tracker API Surface

**Files:**
- Modify: `src/datacell/duplicate_interface.h`
- Modify: `src/datacell/graph_interface.h`

- [ ] **Step 1: Extend the duplicate tracker interface**

```cpp
virtual InnerIdType
GetGroupId(InnerIdType id) const = 0;
```

- [ ] **Step 2: Update duplicate-id parameter naming to `group_id` in the public abstraction**

```cpp
virtual void
SetDuplicateId(InnerIdType group_id, InnerIdType duplicate_id) = 0;
```

- [ ] **Step 3: Add the graph wrapper helper**

```cpp
InnerIdType
GetGroupId(InnerIdType id) const {
    if (duplicate_tracker_) {
        return duplicate_tracker_->GetGroupId(id);
    }
    return id;
}
```

- [ ] **Step 4: Rebuild the targeted unit tests**

Run: `cmake --build build-release --target unittests`

Expected: build continues failing in the dense/sparse implementations because the new pure virtual method is not implemented yet.

### Task 3: Update `DenseDuplicateTracker` To The Final Contract

**Files:**
- Modify: `src/datacell/dense_duplicate_tracker.h`
- Modify: `src/datacell/dense_duplicate_tracker.cpp`
- Test: `src/datacell/dense_duplicate_tracker_test.cpp`

- [ ] **Step 1: Implement `GetGroupId(...)` with a single ring walk**

```cpp
auto current_id = duplicate_ids_[id];
InnerIdType group_id = id;
while (current_id != id) {
    group_id = std::min(group_id, current_id);
    current_id = duplicate_ids_[current_id];
}
return group_id;
```

- [ ] **Step 2: Simplify `SetDuplicateId(...)` to splice after canonical `group_id`**

```cpp
if (duplicate_ids_[group_id] == group_id) {
    duplicate_count_++;
}
duplicate_ids_[duplicate_id] = duplicate_ids_[group_id];
duplicate_ids_[group_id] = duplicate_id;
```

- [ ] **Step 3: Preserve legacy deserialize behavior while satisfying the new API**

```cpp
duplicate_ids_.resize(total_size);
for (size_t i = 0; i < total_size; ++i) {
    duplicate_ids_[i] = static_cast<InnerIdType>(i);
}
```

- [ ] **Step 4: Run dense duplicate tracker tests**

Run: `cmake --build build-release --target unittests && ./build-release/tests/unittests "[DenseDuplicateTracker]"`

Expected: PASS.

### Task 4: Replace Sparse Dual Maps With A Single Ring Map

**Files:**
- Modify: `src/datacell/sparse_duplicate_tracker.h`
- Modify: `src/datacell/sparse_duplicate_tracker.cpp`
- Test: `src/datacell/sparse_duplicate_tracker_test.cpp`

- [ ] **Step 1: Replace the dual maps with one sparse `next` map**

```cpp
UnorderedMap<InnerIdType, InnerIdType> next_ids_;
```

- [ ] **Step 2: Implement sparse ring operations**

```cpp
if (next_ids_.count(duplicate_id) > 0) {
    return;
}
if (next_ids_.count(group_id) == 0) {
    next_ids_[group_id] = duplicate_id;
    next_ids_[duplicate_id] = group_id;
    duplicate_count_++;
    return;
}
next_ids_[duplicate_id] = next_ids_[group_id];
next_ids_[group_id] = duplicate_id;
```

- [ ] **Step 3: Rebuild sparse state from legacy payloads instead of storing legacy maps directly**

```cpp
std::vector<InnerIdType> members;
members.push_back(group_id_from_stream);
members.insert(members.end(), dup_list.begin(), dup_list.end());
const auto canonical_group_id = *std::min_element(members.begin(), members.end());
```

- [ ] **Step 4: Keep sparse `Serialize()` / `Deserialize()` internally self-consistent**

```cpp
StreamWriter::WriteObj(writer, duplicate_count_);
StreamWriter::WriteObj(writer, duplicate_count_);
```

- [ ] **Step 5: Run sparse duplicate tracker tests**

Run: `cmake --build build-release --target unittests && ./build-release/tests/unittests "[SparseDuplicateTracker]"`

Expected: PASS.

### Task 5: Verify Graph Wrappers And Related Compatibility Tests

**Files:**
- Test: `src/datacell/graph_datacell_test.cpp`
- Test: `src/datacell/compressed_graph_datacell_test.cpp`
- Test: `src/impl/label_table_test.cpp`

- [ ] **Step 1: Run graph wrapper tests**

Run: `cmake --build build-release --target unittests && ./build-release/tests/unittests "[GraphDataCell]" "[CompressedGraphDatacell]"`

Expected: PASS.

- [ ] **Step 2: Run the legacy label-table compatibility test**

Run: `./build-release/tests/unittests "[LabelTable]"`

Expected: PASS, including the legacy duplicate payload path.

- [ ] **Step 3: Run the complete duplicate-tracker-focused unit test slice**

Run: `./build-release/tests/unittests "[DenseDuplicateTracker]" "[SparseDuplicateTracker]" "[GraphDataCell]" "[CompressedGraphDatacell]" "[LabelTable]"`

Expected: PASS.
