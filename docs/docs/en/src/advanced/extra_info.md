# Extra Info

`extra_info` is a fixed-size, opaque per-vector byte payload stored alongside vectors inside
the index. It lets you keep small pieces of non-vector metadata (e.g. timestamps, category ids,
permission tags, application-specific fields) right next to the vectors, so you can:

- Retrieve metadata by vector id without a separate KV store.
- Update a vector's metadata in place without re-inserting the vector.
- Filter candidates **during** graph traversal using your own metadata, instead of post-filtering
  results.

The library treats the payload as raw bytes — you fully own its layout, serialization, and
interpretation.

## Index Support

| Index      | Store on Build/Add | `GetExtraInfoByIds` | `UpdateExtraInfo` | In-graph filter (`use_extra_info_filter`) | Returned in search results |
|------------|:------------------:|:-------------------:|:-----------------:|:-----------------------------------------:|:--------------------------:|
| **HGraph** |         Yes        |         Yes         |         Yes       |                    Yes                    |             Yes            |
| IVF        |         Yes        |          —          |         —         |                     —                     |              —             |
| SINDI      |         Yes        |          —          |         —         |                     —                     |              —             |

Only HGraph advertises the related capability flags; for the richest experience use HGraph.
You can always check at runtime with `index->CheckFeature(...)`.

## Enabling Extra Info

Add the top-level integer field `extra_info_size` to the build parameters. The value is the size
in bytes of the payload reserved per vector. Once an index is built, the size is fixed and is
serialized together with the index.

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "extra_info_size": 12,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 26,
        "ef_construction": 100
    }
}
```

If `extra_info_size` is omitted or set to `0`, the feature is disabled.

## Providing Extra Info on Build / Add

Use the `Dataset` builder API to attach the payload. The buffer must be contiguous, with vector
`i`'s payload at byte offset `i * extra_info_size`.

```cpp
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)
    ->Dim(dim)
    ->Ids(ids.data())
    ->Float32Vectors(vectors.data())
    ->ExtraInfos(extra_infos.data())   // num_vectors * extra_info_size bytes
    ->ExtraInfoSize(extra_info_size)   // must match the index's extra_info_size
    ->Owner(false);

index->Build(base);   // or index->Add(base)
```

`ExtraInfoSize` must equal the index's `extra_info_size`; otherwise the call is rejected.

## Retrieving Extra Info

### From Search Results (HGraph)

When `extra_info_size > 0`, HGraph automatically populates the result `Dataset` with the matching
extra_info bytes for every returned id:

```cpp
auto result = index->KnnSearch(query, k, search_params).value();
const char* infos = result->GetExtraInfos();          // length = result->GetDim() * extra_info_size
```

The result `Dataset` carries the `ExtraInfos` buffer but **does not** set `ExtraInfoSize` on it,
so `result->GetExtraInfoSize()` will return `0`. Use the `extra_info_size` you configured at
build time to compute offsets and lengths.

### By Ids (`GetExtraInfoByIds`)

Allocate a `count * extra_info_size` byte buffer and call:

```cpp
if (index->CheckFeature(vsag::SUPPORT_GET_EXTRA_INFO_BY_ID)) {
    std::vector<char> out(count * extra_info_size);
    index->GetExtraInfoByIds(ids, count, out.data());
}
```

If the feature is not enabled, the call returns `UNSUPPORTED_INDEX_OPERATION`.

## Updating Extra Info In Place

Update a single vector's payload without touching the vector itself:

```cpp
if (index->CheckFeature(vsag::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT)) {
    auto upd = vsag::Dataset::Make();
    upd->NumElements(1)
       ->Ids(&id)
       ->ExtraInfos(buffer.data())
       ->ExtraInfoSize(extra_info_size)
       ->Owner(false);
    index->UpdateExtraInfo(upd);
}
```

The dataset must contain exactly one element and the size must match.

## In-Graph Filtering with Extra Info (HGraph)

Post-filtering can be wasteful when the filter prunes many candidates. HGraph can call your
filter on each candidate's extra_info bytes during graph traversal, so disqualified candidates
never enter the result set.

1. Override the byte-buffer overload of `vsag::Filter`:

   ```cpp
   class CategoryFilter : public vsag::Filter {
   public:
       CategoryFilter(uint32_t lo, uint32_t hi) : lo_(lo), hi_(hi) {}
       bool CheckValid(int64_t /*id*/) const override { return true; }   // unused on this path
       bool CheckValid(const char* data) const override {
           uint32_t category_id;
           std::memcpy(&category_id, data, sizeof(category_id));
           return category_id >= lo_ && category_id <= hi_;
       }
       float ValidRatio() const override { return 0.5F; }
   private:
       uint32_t lo_, hi_;
   };
   ```

2. Enable `use_extra_info_filter` inside the `hgraph` block of the search parameters and pass the
   filter to `KnnSearch`:

   ```cpp
   std::string search_params = R"({
       "hgraph": {
           "ef_search": 100,
           "use_extra_info_filter": true
       }
   })";
   auto filter = std::make_shared<CategoryFilter>(3, 7);
   auto result = index->KnnSearch(query, k, search_params, filter).value();
   ```

When `use_extra_info_filter` is true, HGraph dispatches to `CheckValid(const char*)` instead of
`CheckValid(int64_t)`. You can guard with
`index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER)`.

## Capability Flags

| Flag                                          | Meaning                                              |
|-----------------------------------------------|------------------------------------------------------|
| `vsag::SUPPORT_GET_EXTRA_INFO_BY_ID`          | `GetExtraInfoByIds` is available.                    |
| `vsag::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT`  | `UpdateExtraInfo` is available and thread-safe.      |
| `vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER`     | `use_extra_info_filter` is available in search.      |

## Notes and Limitations

- The payload is opaque bytes; you are responsible for serialization/deserialization. The library
  only `memcpy`s by offset.
- `extra_info_size` is fixed at build time and persisted in the serialized index.
- Storage cost is `extra_info_size * num_elements` bytes, accounted into `EstimateMemory`.
- Keep the payload compact — it is loaded into memory and walked during in-graph filtering.
- The feature is currently C++ only; there is no Python binding for `extra_info`.

## Example

A complete, runnable example is available at
`examples/cpp/320_feature_extra_info.cpp`. It demonstrates building an HGraph index with
`extra_info`, retrieval by id, in-graph filtering, and in-place updates.
