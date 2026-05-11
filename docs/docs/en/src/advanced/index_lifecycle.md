# Index Lifecycle Management

After an index is built, VSAG provides several operations that mutate the index in place or
produce a new index derived from it. This page documents the full lifecycle surface:

- `Remove` — delete vectors by id.
- `UpdateVector` / `UpdateId` — modify an existing vector or rename its id.
- `Clone` — produce a deep copy of an existing index.
- `ExportModel` — extract the trained model as an empty index for reuse.

Each operation is optional and is exposed only when the underlying index advertises the matching
capability flag via `index->CheckFeature(...)`.

## Capability Flags

| Operation         | Capability Flag                          | HGraph | IVF | SINDI |
|-------------------|------------------------------------------|:------:|:---:|:-----:|
| `Remove`          | _(no dedicated flag — see below)_        |   Yes  |  —  |   —   |
| `UpdateVector`    | `SUPPORT_UPDATE_VECTOR_CONCURRENT`       |   Yes  |  —  |  Yes  |
| `UpdateId`        | `SUPPORT_UPDATE_ID_CONCURRENT`           |   Yes  |  —  |  Yes  |
| `Clone`           | `SUPPORT_CLONE`                          |   Yes  | Yes |   —   |
| `ExportModel`     | `SUPPORT_EXPORT_MODEL`                   |   Yes  | Yes |   —   |

For the flag-gated operations, check at runtime with `index->CheckFeature(vsag::SUPPORT_*)` before
calling; unsupported indexes return `UNSUPPORTED_INDEX_OPERATION`. `Remove` does not currently
have a dedicated capability flag — see the next section for how to determine whether your index
supports it and which mode it supports.

## Removing Vectors

`Remove` deletes vectors by id. HGraph supports two deletion modes with different requirements:

- `RemoveMode::MARK_REMOVE` (the default) only writes a tombstone via the label table and works
  regardless of `support_remove`. The id is filtered out of subsequent searches, but the underlying
  graph node and vector storage are kept.
- `RemoveMode::FORCE_REMOVE` physically rewrites the graph and reclaims the slot. This mode is
  only available when the index was built with `support_remove: true` in `index_param` (which
  causes the graph data cell to allocate the delete-tracking metadata). Calling `FORCE_REMOVE` on
  an index built without `support_remove: true` will fail.

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 16,
        "ef_construction": 100,
        "support_remove": true
    }
}
```

The JSON snippet above is only required if you intend to use `FORCE_REMOVE`. For `MARK_REMOVE`
alone you can omit the `support_remove` flag.

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 16,
        "ef_construction": 100,
        "support_remove": true
    }
}
```

```cpp
// Single-id and batch overloads are available.
index->Remove(id);
index->Remove(std::vector<int64_t>{id1, id2, id3});
```

### Remove Modes

The optional `RemoveMode` argument selects the deletion strategy:

| Mode                          | Behavior                                                          |
|-------------------------------|-------------------------------------------------------------------|
| `RemoveMode::MARK_REMOVE` (default) | Tombstones the id; fast, no shrink or graph repair. Subsequent searches skip the id. Does not require `support_remove: true`. |
| `RemoveMode::FORCE_REMOVE`    | Physically removes the vector and repairs the graph. Heavier. Requires the index to be built with `support_remove: true`. |

`Remove` returns the number of ids that were successfully removed. Ids that did not exist are
silently skipped and not counted.

A runnable example is available at `examples/cpp/303_feature_remove.cpp`.

## Updating Vectors and Ids

### `UpdateVector`

`UpdateVector(id, new_base, force_update = false)` replaces the vector data of an existing id in
place. The default `force_update = false` mode performs a connectivity check: if the new vector
is far from the original (which would degrade graph quality), the update is **rejected** and the
caller is expected to fall back to `Remove` + `Add`.

```cpp
std::vector<float> new_vec(dim);  // populate with the replacement vector
auto upd = vsag::Dataset::Make();
upd->NumElements(1)->Dim(dim)->Ids(&id)->Float32Vectors(new_vec.data())->Owner(false);

auto status = index->UpdateVector(id, upd, /*force_update=*/false);
if (status.has_value() && *status) {
    // updated in place
} else if (status.has_value() && not *status) {
    // rejected: new vector is too far from the old one — fall back to remove + add
    index->Remove(id);
    index->Add(upd);
}
```

Setting `force_update = true` skips the check and always applies the update; use with caution as
it may degrade recall.

### `UpdateId`

`UpdateId(old_id, new_id)` renames an existing id without touching the underlying vector.
Returns `true` on success, `false` if `old_id` was not found or `new_id` already exists.

```cpp
index->UpdateId(123, 456);
```

A runnable example combining `UpdateVector`, `Remove`, and `Add` is available at
`examples/cpp/305_feature_update.cpp`.

## Cloning an Index

`Clone()` produces a deep copy of the entire index — vectors, graph, quantizer state, and
metadata — as an independent `IndexPtr`. The clone can be searched, mutated, or serialized
independently of the source.

```cpp
auto cloned = index->Clone().value();

// Both indexes return identical search results immediately after cloning.
auto r1 = index->KnnSearch(query, k, params).value();
auto r2 = cloned->KnnSearch(query, k, params).value();
```

`Clone` optionally accepts a custom `Allocator` so that the cloned index uses a different memory
region than the source — useful for handing an index off to a thread or component that owns its
own allocator. See [Memory Management](memory.md) for allocator details.

A runnable example is available at `examples/cpp/309_feature_clone.cpp`.

## Exporting the Trained Model

`ExportModel()` returns an empty index that retains all trained state (quantization codebooks,
centroids, hyperparameters) of the source but contains no vectors. It is the canonical way to
share a pre-trained model across shards, processes, or hosts without re-running training.

```cpp
auto exported = index->ExportModel();
if (not exported.has_value()) {
    // index does not support ExportModel — handle the error
    return;
}
auto model = *exported;

// Populate the empty model with a new (potentially different) vector set.
for (int64_t i = 0; i < num_vectors; ++i) {
    auto one = vsag::Dataset::Make();
    one->NumElements(1)->Dim(dim)->Ids(ids + i)
       ->Float32Vectors(vectors + i * dim)->Owner(false);
    model->Add(one);
}
```

The returned index behaves identically to one freshly created via `Factory::CreateIndex(...)` and
trained on the source data — only the per-vector storage is empty. This pattern is particularly
useful for IVF-style indexes where training (k-means on centroids) is the dominant cost.

A runnable example is available at `examples/cpp/310_feature_export_model.cpp`.

## Notes and Limitations

- `Remove`, `UpdateVector`, and `UpdateId` are concurrent-safe on HGraph when the matching
  `*_CONCURRENT` capability flag is set. The flag set also gates safe combinations with
  concurrent search and add (e.g. `SUPPORT_ADD_SEARCH_DELETE_CONCURRENT`).
- `MARK_REMOVE` does not free memory; use `FORCE_REMOVE` or rebuild periodically if you need to
  reclaim space.
- `Clone` cost scales linearly with index size. For large indexes prefer serialization +
  deserialization with a dedicated reader if you only need a snapshot on disk.
- `ExportModel` preserves training but **not** any inserted vectors. The exported model can be
  freely serialized and shipped before any vectors are added.
