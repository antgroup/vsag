# Index Introspection

VSAG indexes expose three families of introspection APIs that let callers discover what an index
can do, compute distances against existing vectors, and read back structured information about
the built index without re-running a search:

- **`CheckFeature(IndexFeature)`** — runtime capability discovery.
- **`CalDistanceById(...)`** — distance from a query to specific stored ids.
- **`GetIndexDetailInfos()` / `GetDetailDataByName(...)`** — structured per-index detail data.

These APIs are read-only and safe to call concurrently with search.

## Capability Discovery — `CheckFeature`

`index->CheckFeature(vsag::SUPPORT_*)` returns `true` when the underlying index implementation
advertises the given feature. Use it whenever a code path takes an `IndexPtr` of unknown concrete
type (e.g. user-supplied configuration, polymorphic store):

```cpp
if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
    uint64_t est = index->EstimateMemory(100'000);
}

if (not index->CheckFeature(vsag::SUPPORT_DELETE_BY_ID)) {
    // Skip / fall back to remove + re-add via a different index.
}
```

Feature flags cover almost every optional surface in the library: build / add /
serialize variants, concurrent combinations, metric types, attribute and extra-info filters,
`Clone`, `ExportModel`, `Tune`, and more. See `include/vsag/index_features.h` for the full
enumeration.

A runnable example is available at `examples/cpp/307_feature_check_features.cpp`.

## Distances to Existing Ids — `CalDistanceById`

`CalDistanceById` computes the distance between a query and one or more vectors that are
**already stored in the index**, without running a search. This is useful for re-ranking, A/B
evaluation, ground-truth checks, or computing pairwise distances to a known shortlist.

Two overloads are provided:

```cpp
// Dense vector indexes (HGraph, BruteForce, IVF)
auto r = index->CalDistanceById(query_ptr, ids, count, /*calculate_precise_distance=*/true);

// Sparse vector indexes (SINDI) — wrap the query in a Dataset
auto query_ds = vsag::Dataset::Make();
query_ds->NumElements(1)->SparseVectors(/* ... */);
auto r = index->CalDistanceById(query_ds, ids, count, /*calculate_precise_distance=*/true);
```

The result `Dataset` holds `count` distances in `GetDistances()`. A value of `-1.0F` means the
corresponding id was invalid (not present in the index).

### `calculate_precise_distance`

The trailing `bool` argument trades precision for latency:

| Value | Behavior                                                                                                  |
|-------|-----------------------------------------------------------------------------------------------------------|
| `true` (default) | Use the full-precision vector representation. May incur disk I/O on hybrid memory-disk indexes. |
| `false`          | Use the quantized / approximate representation cached for search. Faster, no I/O.               |

A runnable example is available at `examples/cpp/306_feature_calculate_distance_by_id.cpp`.

## Detail Data — `GetIndexDetailInfos` / `GetDetailDataByName`

`GetIndexDetailInfos()` returns a list of `IndexDetailInfo` records that describe every named
piece of structured data the index can expose. Each record carries a `name`, a `description`, and
a `type` enum that selects the right typed accessor on `DetailData`.

Support is index-dependent — there is no dedicated `SUPPORT_*` flag for these two APIs. The
`Index` base class throws `std::runtime_error("Index doesn't support ...")` by default
(`GetIndexDetailInfos` and `GetDetailDataByName` in `include/vsag/index.h:658,674`);
HGraph / IVF / BruteForce / Pyramid / SINDI / WARP implement them through
`InnerIndexInterface`. Always handle the `tl::expected` error path when calling these APIs.

```cpp
auto infos = index->GetIndexDetailInfos().value();
for (const auto& info : infos) {
    std::cout << info.name << " : " << info.description << '\n';
}
```

Once you know which entries are available, call `GetDetailDataByName(name, info)` to retrieve the
typed payload:

```cpp
vsag::IndexDetailInfo info;
auto detail = index->GetDetailDataByName(vsag::INDEX_DETAIL_NAME_NUM_ELEMENTS, info).value();
int64_t n = detail->GetDataScalarInt64();

detail = index->GetDetailDataByName(vsag::INDEX_DETAIL_NAME_LABEL_TABLE, info).value();
auto table = detail->GetData2DArrayInt64();   // [row][col] int64 matrix

detail = index->GetDetailDataByName(vsag::INDEX_DETAIL_DATA_TYPE, info).value();
std::string dt = detail->GetDataScalarString();
```

### Data Types

`info.type` selects which accessor on `DetailData` is valid:

| `IndexDetailDataType`     | Accessor                              |
|---------------------------|---------------------------------------|
| `TYPE_SCALAR_INT64`       | `GetDataScalarInt64()`                |
| `TYPE_SCALAR_DOUBLE`      | `GetDataScalarDouble()`               |
| `TYPE_SCALAR_BOOL`        | `GetDataScalarBool()`                 |
| `TYPE_SCALAR_STRING`      | `GetDataScalarString()`               |
| `TYPE_1DArray_INT64`      | `GetData1DArrayInt64()`               |
| `TYPE_2DArray_INT64`      | `GetData2DArrayInt64()`               |

Standard detail names exposed as constants in `include/vsag/index_detail_info.h`:

| Constant                            | Typical type           | Meaning                                  |
|-------------------------------------|------------------------|------------------------------------------|
| `INDEX_DETAIL_NAME_NUM_ELEMENTS`    | `TYPE_SCALAR_INT64`    | Number of vectors currently in the index. |
| `INDEX_DETAIL_NAME_LABEL_TABLE`     | `TYPE_2DArray_INT64`   | Per-vector label table (e.g. internal-to-user id mapping). |
| `INDEX_DETAIL_DATA_TYPE`            | `TYPE_SCALAR_STRING`   | Underlying vector data type (e.g. `"float32"`). |

Individual indexes may expose additional names; iterate `GetIndexDetailInfos()` to discover them
at runtime. A runnable example is available at `examples/cpp/317_feature_get_detail_data.cpp`.

## Notes and Limitations

- `CheckFeature` is constant-time. Prefer it over `try` / `catch` around an unsupported call.
- `CalDistanceById` requires the underlying index to retain enough information to recompute the
  distance. For purely quantized indexes (no raw vectors retained), `calculate_precise_distance =
  true` may return the quantized distance instead.
- `GetIndexDetailInfos` and `GetDetailDataByName` are read-only snapshots. The values returned
  reflect the index state at the moment of the call; concurrent mutations may invalidate them.
