# Duplicate Tracker Ring Redesign

## Summary

Redesign duplicate tracking so `DenseDuplicateTracker` and `SparseDuplicateTracker` share the same
ring-based semantics while keeping storage specialized for dense and sparse datacells.

The main goal is reducing `SparseDuplicateTracker` memory usage by replacing the current
`original_to_duplicates_ + duplicate_to_original_` dual structure with a single sparse `next`
mapping. Compatibility is only required for `DeserializeFromLegacyFormat(...)`; compatibility of
the current `Serialize()` and `Deserialize()` formats is explicitly out of scope.

## Goals

- Cut `SparseDuplicateTracker` storage materially, targeting roughly half of the current overhead.
- Unify dense and sparse duplicate semantics around the same ring model.
- Rename duplicate-group semantics from `origin_id` to `group_id`.
- Add `GetGroupId(InnerIdType id)`.
- Preserve current query behavior: `GetDuplicateIds(id)` returns all other ids in the same group
  and excludes `id` itself.
- Preserve compatibility for legacy duplicate payloads through `DeserializeFromLegacyFormat(...)`.

## Non-Goals

- Preserving binary compatibility of the current `Serialize()` / `Deserialize()` payloads.
- Collapsing dense and sparse trackers into a single storage class.
- Refactoring unrelated searcher logic.
- Renaming every historical variable such as `InnerSearchParam::duplicate_id` in this change.

## Confirmed Constraints

- `group_id` is the canonical representative of a duplicate group.
- `group_id` is defined as the minimum id in the group.
- `SetDuplicateId(group_id, duplicate_id)` expects callers to provide canonical `group_id`.
- Hot paths in `HGraph` and `Pyramid` already satisfy that contract and should not pay an extra
  normalization lookup.
- `GetGroupId(id)` exists primarily for external callers and non-hot-path convenience.
- IDs are monotonic in the current system, so the invalid pattern `SetDuplicateId(10, 5)` does not
  occur in practice.
- `GetDuplicateIds(id)` must exclude `id` itself because searchers add the matched id first and
  then expand duplicates separately.

## Current Context

### Dense

`DenseDuplicateTracker` already behaves like a ring:

- storage is a dense vector indexed by `InnerIdType`
- each member stores the next member in the same duplicate group
- a non-duplicate member is represented by a self-loop
- `Resize()` establishes dense indexability

### Sparse

`SparseDuplicateTracker` currently stores both:

- a forward map from representative to duplicate vector
- a reverse map from duplicate to representative

This duplicates relationship data and adds per-group vector overhead. The behavior is correct, but
the storage shape is more expensive than needed.

## Proposed API

Keep the public mutation name minimal and change the semantics:

```cpp
void SetDuplicateId(InnerIdType group_id, InnerIdType duplicate_id);
Vector<InnerIdType> GetDuplicateIds(InnerIdType id) const;
InnerIdType GetGroupId(InnerIdType id) const;
```

Semantics:

- `SetDuplicateId(group_id, duplicate_id)` inserts `duplicate_id` into the group represented by the
  canonical minimum-id `group_id`.
- `GetDuplicateIds(id)` returns every other member of the same duplicate group and excludes `id`.
- `GetGroupId(id)` returns the canonical minimum-id representative if `id` belongs to a group,
  otherwise it returns `id`.

For minimal churn, the method name remains `SetDuplicateId(...)`. Only the parameter meaning is
updated from `origin_id` to `group_id`.

## Proposed Internal Model

### Shared Ring Semantics

Both trackers use the same logical model:

- every duplicate group is a circular list
- each member points to the next member in the group
- iterating from any member eventually visits the full group and returns to the starting point
- `GetDuplicateIds(id)` walks one full cycle starting from `next(id)` and stops when it returns to
  `id`
- `GetGroupId(id)` walks one full cycle and returns the minimum visited id

### Dense Storage

Dense storage stays close to the current implementation:

- `duplicate_ids_[id]` stores the next member in the ring
- singleton members keep `duplicate_ids_[id] == id`
- `Resize(new_size)` grows the vector and initializes new entries as self-loops

This keeps constant-time indexed access and preserves the dense tracker's fit for contiguous id
spaces.

### Sparse Storage

Sparse storage changes to a single mapping:

- `next_ids_[id] = next_member`
- only ids that belong to a duplicate group are present in the map
- singleton members have no entry

For a group `{3, 8, 10}`, sparse storage would contain:

```text
3 -> 8
8 -> 10
10 -> 3
```

This removes:

- the reverse `duplicate_to_original_` map
- the per-group `std::vector` in `original_to_duplicates_`

The result is one sparse entry per grouped member instead of one reverse entry plus one forward
vector slot plus container overhead.

## Core Operations

### SetDuplicateId(group_id, duplicate_id)

Insertion uses ring splicing in both implementations:

1. If `group_id` is currently a singleton, create a two-node ring.
2. If the group already exists, insert `duplicate_id` immediately after `group_id`.

Example for existing group `{3, 8, 10}` and new `duplicate_id = 12`:

```text
before: 3 -> 8 -> 10 -> 3
after:  3 -> 12 -> 8 -> 10 -> 3
```

This keeps insertion local and avoids rebuilding a full vector representation.

The implementation will trust the caller contract that `group_id` is already canonical. The hot
path should not perform an extra normalization lookup.

### GetDuplicateIds(id)

- If `id` is a singleton, return an empty vector.
- Otherwise, start from the successor of `id`, walk the ring once, and collect visited ids until
  reaching `id` again.
- `id` itself is never returned.

Example for group `{3, 8, 10}`:

- `GetDuplicateIds(3)` returns `{8, 10}`
- `GetDuplicateIds(8)` returns `{10, 3}`
- `GetDuplicateIds(10)` returns `{3, 8}`

The ordering is ring order, not sorted order. This matches the existing semantic requirement that
the current hit is excluded while the rest of the group is visible.

### GetGroupId(id)

- If `id` is a singleton, return `id`.
- Otherwise, walk the ring once and return the minimum id in the cycle.

This is `O(group_size)`, which is acceptable because:

- hot insertion paths already know canonical `group_id`
- external callers can trade a small scan for a simpler API
- duplicate groups are expected to remain small relative to full graph size

## Legacy Compatibility

`DeserializeFromLegacyFormat(...)` is the only required compatibility boundary.

Required behavior:

- accept the existing legacy duplicate payload layout unchanged
- reconstruct the new ring-based runtime structure from that payload
- preserve externally visible duplicate semantics after load

Implementation approach:

1. Read legacy payload data using the existing legacy reader contract.
2. Materialize each legacy duplicate group in memory.
3. Compute canonical `group_id` as the minimum id in that group.
4. Rebuild the new runtime structure by inserting all other members under that canonical
   `group_id`.

Because ids are monotonic in the system, the legacy representative is expected to already match the
minimum id in normal data. Canonicalization during rebuild keeps the new API semantics explicit and
robust without introducing a new compatibility format version.

## Serialize / Deserialize

Compatibility of the current `Serialize()` / `Deserialize()` format is out of scope.

That allows the new implementation to write and read a simpler format that matches the new runtime
structure. The only requirement for this path is internal consistency between the new writer and
reader.

To keep the patch reviewable, the format should favor straightforward reconstruction over clever
compression.

## Call Site Impact

### Duplicate Interface / Graph Interface

- add `GetGroupId(InnerIdType id) const`
- update parameter naming and comments from `origin_id` to `group_id`

### HGraph / Pyramid

- keep the current behavior of passing canonical ids directly into `SetDuplicateId(...)`
- no extra lookup should be added in the hot insertion path

### Searchers

- `BasicSearcher` and `ParallelSearcher` require no logic change
- they already push the current hit first and then expand duplicates through
  `GetDuplicateIds(id)`
- preserving exclusion of `id` is therefore sufficient for behavioral compatibility

### Naming Cleanup

`InnerSearchParam::duplicate_id` may eventually be renamed to `group_id`, but that should remain a
separate cleanup so this redesign stays scoped to duplicate tracker behavior and storage.

## Error Handling And Invariants

This redesign does not introduce a new public error surface.

The trackers continue to rely on the existing caller contract:

- `group_id` is canonical
- `duplicate_id` is not already linked into another group

Implementation checks should stay minimal and follow existing project patterns. If extra internal
validation is added, it should be limited to lightweight debug assertions rather than new runtime
exceptions in the hot path.

## Testing Plan

### Unit Tests

- update dense tracker tests to cover `GetGroupId()`
- update sparse tracker tests to cover `GetGroupId()`
- keep explicit assertions that `GetDuplicateIds(id)` excludes `id`
- add parity cases so dense and sparse trackers produce the same results for the same group inputs
- add sparse insertion tests that cover singleton-to-group promotion and repeated group extension

### Legacy Compatibility Tests

- preserve and update tests around `DeserializeFromLegacyFormat(...)`
- verify legacy payloads still load correctly into the new runtime structure
- verify loaded groups return the expected `GetDuplicateIds(id)` and `GetGroupId(id)` values

### Integration Coverage

- confirm duplicate expansion behavior in searcher-related tests remains unchanged
- confirm HGraph and Pyramid duplicate tests continue to pass without daily-only coverage changes

## Implementation Sequence

1. Extend `DuplicateInterface` and `GraphInterface` with `GetGroupId(...)`.
2. Rename parameter semantics and comments from `origin_id` to `group_id`.
3. Update `DenseDuplicateTracker` to expose the finalized API semantics, including `GetGroupId()`.
4. Replace `SparseDuplicateTracker` internals with the single sparse `next` mapping.
5. Rework `DeserializeFromLegacyFormat(...)` to rebuild the new structure from legacy payloads.
6. Simplify `Serialize()` / `Deserialize()` to match the new structure without preserving the old
   non-legacy payload layout.
7. Update unit and compatibility tests.
8. Run the relevant duplicate tracker, HGraph, and Pyramid test coverage.

## Decision Summary

- Keep dense and sparse trackers as separate classes.
- Unify them semantically around a ring model.
- Introduce `group_id` as the canonical minimum id.
- Add `GetGroupId(id)`.
- Preserve `DeserializeFromLegacyFormat(...)` compatibility only.
- Do not expand scope into unrelated naming cleanup or searcher refactors.
