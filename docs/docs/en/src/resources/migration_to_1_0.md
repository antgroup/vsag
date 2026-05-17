# Migrating to VSAG 1.0

This page collects everything users coming from **VSAG 0.18.x** (and
earlier) need to know to upgrade smoothly to **VSAG 1.0**. Read it before
you recompile or redeploy.

> Tracked in
> [issue #2069](https://github.com/antgroup/vsag/issues/2069) /
> [PR #2070](https://github.com/antgroup/vsag/pull/2070). Corrections
> and "what we hit during the upgrade" feedback are welcome — please
> open an issue.

> 1.0 follows [Semantic Versioning](https://semver.org/). The compatibility
> rules going forward will be spelled out in a dedicated *API Stability*
> page, planned as a follow-up PR tracked in
> [#2069](https://github.com/antgroup/vsag/issues/2069); this page focuses
> on the one-time 0.18 → 1.0 migration.

## At a glance

| Area | Status in 1.0 | Action |
|------|---------------|--------|
| `hnsw` index | Deprecated, still works | Move new deployments to [HGraph](../indexes/hgraph.md) |
| `diskann` index | Deprecated, still works | Move new deployments to [IVF](../indexes/ivf.md) or the [hybrid memory-disk index](../advanced/hybrid_index.md) |
| `Index::KnnSearch(query, k, SearchParam&)` | Deprecated overload | Switch to `Index::SearchWithRequest(SearchRequest)` |
| `SearchParam::allocator` | Deprecated field | Use `SearchRequest::search_allocator_` |
| `Index::CalDistanceById` (batch) | Kept (typo'd name) | Continue to use; a correctly-spelled `CalcDistancesById` is planned (see [#2068](https://github.com/antgroup/vsag/issues/2068)) |
| Serialized indexes from 0.18.x | Readable by 1.0 | Re-serialize after upgrade to pick up any layout improvements |
| Public C ABI | Stable | No action |

The rest of this page expands each row with concrete code samples.

## Deprecated indexes

### `hnsw` → `hgraph`

`hnsw` is the original graph-based index inherited from hnswlib. In 1.0 it
is retained for backward compatibility but **deprecated**; new deployments
should use [HGraph](../indexes/hgraph.md), which is a superset:

- Same hierarchical-graph topology, with the same `max_degree` /
  `ef_construction` / `ef_search` knobs.
- A unified `index_param` build schema with richer quantization options
  (`fp32`, `fp16`, `bf16`, `sq8`, `sq8_uniform`, `sq4_uniform`, `pq`,
  `pqfs`, `rabitq`).
- Optional re-ranking (`use_reorder` + `precise_quantization_type`),
  duplicate handling, `Remove()`, and ELP-based runtime tuning.

Build-time mapping:

```diff
- auto index = vsag::Factory::CreateIndex("hnsw", R"({
-     "dim": 768,
-     "dtype": "float32",
-     "metric_type": "ip",
-     "hnsw": {
-         "max_degree": 32,
-         "ef_construction": 400
-     }
- })").value();
+ auto index = vsag::Factory::CreateIndex("hgraph", R"({
+     "dim": 768,
+     "dtype": "float32",
+     "metric_type": "ip",
+     "index_param": {
+         "base_quantization_type": "fp32",
+         "max_degree": 32,
+         "ef_construction": 400
+     }
+ })").value();
```

Search-time mapping:

```diff
- auto result = index->KnnSearch(query, k, R"({"hnsw": {"ef_search": 100}})").value();
+ auto result = index->KnnSearch(query, k, R"({"hgraph": {"ef_search": 100}})").value();
```

Two things to remember:

1. The build sub-object key changes from `"hnsw"` to `"index_param"`, and
   `base_quantization_type` becomes a required field.
2. The search sub-object key also changes from `"hnsw"` to `"hgraph"`.

### `diskann` → `ivf` or hybrid memory-disk

`diskann` provided memory-disk hybrid retrieval with PQ-in-memory and
full vectors on disk. In 1.0 it is **deprecated**; choose one of:

- [IVF](../indexes/ivf.md) — for partition-based search at scale; the
  natural in-memory replacement when your dataset fits in RAM.
- [Hybrid memory-disk index](../advanced/hybrid_index.md) — when you
  genuinely need part of the index on NVMe (large corpora under tight
  memory budgets).

Pick IVF first; only fall back to the disk-resident hybrid configuration
if you have measured that memory is the binding constraint.

### `hnsw` and `diskann` examples are no longer the primary references

The on-website pages [Creating an Index](../guide/create_index.md),
[Index Parameters](index_parameters.md), and
[Serialization](../advanced/serialization.md) will be updated to use
`hgraph` as the default example in follow-up PRs tracked in
[#2069](https://github.com/antgroup/vsag/issues/2069). The legacy
examples remain in `examples/cpp/101_index_hnsw.cpp` and
`examples/cpp/102_index_diskann.cpp` for reference.

## Deprecated search API: `SearchParam` → `SearchRequest`

VSAG accumulated several `Index::KnnSearch` overloads over time. The 1.0
public API converges on a single entry point that carries **all** search
options through one struct:

```cpp
[[nodiscard]] tl::expected<DatasetPtr, Error>
SearchWithRequest(const SearchRequest& request) const;
```

`SearchRequest` (declared in [`include/vsag/search_request.h`](https://github.com/antgroup/vsag/blob/main/include/vsag/search_request.h))
supports KNN and range search, attribute filtering, callback filtering,
bitset filtering, iterator search, per-search allocators, and "expected
labels" reasoning — all from one struct. The older
`Index::KnnSearch(query, k, SearchParam&)` overload is **deprecated** and
will be removed in a future major release.

### Field mapping

| `SearchParam` (old) | `SearchRequest` (new) | Notes |
|---------------------|-----------------------|-------|
| `parameters` (`const std::string&`) | `params_str_` (`std::string`) | The JSON parameter string (e.g. `{"hgraph": {"ef_search": 200}}`). |
| `filter` | `filter_` + `enable_filter_ = true` | The callback `Filter` object. Must explicitly enable. |
| `allocator` | `search_allocator_` | Per-search arena allocator. See [Per-Search Allocator](../advanced/search_allocator.md). |
| `iter_ctx` | `p_iter_ctx_` + `enable_iterator_search_ = true` | Note the `**` shape — `SearchRequest` takes `IteratorContext**`. |
| `is_iter_filter` | folded into `enable_iterator_search_` | Iterator search is now opt-in via a single boolean. |
| `is_last_search` | `is_last_search_` | Same semantics. |

`SearchRequest` additionally exposes capabilities that `SearchParam` never
had:

- `mode_` (`SearchMode::KNN_SEARCH` / `SearchMode::RANGE_SEARCH`),
  `topk_`, `radius_`, `limited_size_` — one struct for both KNN and
  range search.
- `enable_attribute_filter_` + `attribute_filter_str_` — SQL-like
  attribute filtering; see [Attribute Filter](../advanced/attribute_filter.md).
- `enable_bitset_filter_` + `bitset_filter_` — bitset-based filtering.
- `expected_labels_` — for recall-debugging / reasoning analysis.

### Code migration

Before:

```cpp
vsag::SearchParam param(
    /*iter_filter_flag=*/false,
    R"({"hgraph": {"ef_search": 200}})",
    /*filter=*/my_filter,
    /*allocator=*/my_arena);
auto result = index->KnnSearch(query, /*k=*/10, param).value();
```

After:

```cpp
vsag::SearchRequest req;
req.query_              = query;
req.mode_               = vsag::SearchMode::KNN_SEARCH;
req.topk_               = 10;
req.params_str_         = R"({"hgraph": {"ef_search": 200}})";
req.enable_filter_      = static_cast<bool>(my_filter);
req.filter_             = my_filter;
req.search_allocator_   = my_arena;
auto result = index->SearchWithRequest(req).value();
```

Range search collapses into the same call by switching `mode_`:

```cpp
req.mode_         = vsag::SearchMode::RANGE_SEARCH;
req.radius_       = 0.42F;
req.limited_size_ = 1000;   // -1 means no cap
auto result = index->SearchWithRequest(req).value();
```

> **Tip.** `SearchRequest` is a plain struct with default values, so
> wrapping it in a small helper / builder is straightforward and tends
> to read more clearly than the multi-argument `SearchParam`
> constructor.

## `CalDistanceById` typo and the `CalcDistancesById` path

VSAG exposes two flavors of distance-by-ID APIs on `Index`:

- **Single** ID, correctly spelled: `CalcDistanceById(...)`.
- **Batch** IDs, *misspelled* historically: `CalDistanceById(...)`
  (missing the `c` in `Calc`).

The naming inconsistency is documented in
[Calculate Distance by ID](../advanced/calc_distance_by_id.md) and tracked
in [#2068](https://github.com/antgroup/vsag/issues/2068).

**What 1.0 does:**

- Both names continue to work; the batch method is **not** renamed in
  1.0.
- The batch method will be renamed to `CalcDistancesById` in a future
  release, with the old name kept as a deprecated alias for at least
  one minor cycle.

**What you should do today:**

- Keep using `CalDistanceById` for batch calls.
- Centralize the call behind a thin wrapper in your codebase. When the
  rename ships, you only need to update the wrapper:

  ```cpp
  // wrappers/vsag_calc_distance.h
  inline auto CalcDistances(const vsag::IndexPtr& index,
                            const float* query,
                            const int64_t* ids,
                            int64_t count,
                            bool precise = true) {
      // Today: forwards to the typo'd name.
      return index->CalDistanceById(query, ids, count, precise);
  }
  ```

## Serialization compatibility

VSAG 1.0 can **read** indexes serialized by 0.18.x via any of the three
serialization interfaces (`BinarySet` / `ReaderSet`, file streams, custom
`WriteFuncType`); the on-disk layout and metadata format are compatible
on the forward path.

Recommendations:

- After upgrading, **re-serialize once** so newly-produced artefacts use
  any layout improvements that ship with 1.0.
- The reverse direction (1.0 → 0.18.x) is **not** supported. Pin a single
  reader version per production cluster during the upgrade window.
- `Deserialize` still requires an empty target index whose build
  configuration (`dim`, `dtype`, `metric_type`, …) matches the original;
  see [Serialization](../advanced/serialization.md).
- DiskANN's on-disk shards remain managed independently; if you are
  migrating away from `diskann`, treat the disk files as throwaway data
  and rebuild on the new index type.

Going forward, the compatibility contract between minor versions will be
codified in a dedicated *API Stability* page, planned as a follow-up PR
tracked in [#2069](https://github.com/antgroup/vsag/issues/2069).

## Default-value and behavioral changes

Things to double-check after pulling 1.0:

- **MKL is off by default.** `VSAG_ENABLE_INTEL_MKL` (CMake:
  `ENABLE_INTEL_MKL`) defaults to `OFF`. On Intel CPUs where MKL was
  expected, set `VSAG_ENABLE_INTEL_MKL=ON` at build time. The
  [reference performance](performance.md) numbers are gathered with MKL
  off.
- **HGraph defaults.** `max_degree` defaults to `64`, `ef_construction`
  to `400`, `graph_type` to `"nsw"`. The build sub-object key is
  `index_param`; `base_quantization_type` is required.
- **`support_remove` / `support_duplicate` are opt-in.** If you relied
  on `Remove()` or on duplicate detection from an experimental branch,
  enable them explicitly under `index_param`.
- **`store_raw_vector`** is opt-in and only needed when you require the
  raw vector after build (e.g. for `cosine` re-ranking when the base
  representation is quantized).

If a behavioral change surfaces that is not covered here, please file an
issue and link this page.

## Build-system and packaging notes

- **Toolchain pins remain unchanged.** `clang-format` / `clang-tidy`
  must be **version 15 exactly**; GCC ≥ 9.4, Clang ≥ 13.0, CMake ≥ 3.18.
- **ABI variants are unchanged.** Choose the redistributable tarball
  matching your downstream toolchain:
  - `make dist-pre-cxx11-abi` — GCC `_GLIBCXX_USE_CXX11_ABI=0`.
  - `make dist-cxx11-abi` — GCC `_GLIBCXX_USE_CXX11_ABI=1`.
  - `make dist-libcxx` — Clang's libc++.
- **Python wheels.** `pip install pyvsag` continues to work; build from
  source via `make pyvsag PY_VERSION=3.10` or `make pyvsag-all`.
- **Node.js / TypeScript.** `npm install vsag`.

## Upgrade checklist

A short, ordered list to drive an upgrade from 0.18.x to 1.0:

1. **Read this page** end-to-end and skim the
   [release notes](release_notes.md).
2. **Inventory deprecated usage** in your codebase:
   - `vsag::Factory::CreateIndex("hnsw", ...)` and `("diskann", ...)`.
   - `Index::KnnSearch(query, k, SearchParam&)` and any code that
     constructs `vsag::SearchParam` directly.
   - Direct calls to `CalDistanceById` (the batch overload); add a
     wrapper now to soften the future rename.
3. **Plan replacements** using the tables in this page; aim for HGraph
   and `SearchRequest` first.
4. **Test in staging.** Build an HGraph (and/or IVF) index with the same
   `dim` / `metric_type` as your existing one; compare recall and
   latency via [`eval_performance`](eval.md).
5. **Validate serialization round-trip.** Load 0.18.x artefacts with the
   1.0 binary, then re-serialize and reload.
6. **Roll out gradually.** Keep one cluster on 0.18.x as a fall-back
   until the new cluster has been stable for at least one release of
   1.0.x.
7. **Update CI/CD pinning.** `pip install pyvsag==1.0.*`,
   `npm install vsag@^1.0.0`, and pin the C++ tarball to the matching
   ABI variant.

When the upgrade is complete, please consider filing an issue or
contributing a short "what we hit" note so this page can keep improving.

## See also

- [Release Notes](release_notes.md)
- *API Stability* (planned, see [#2069](https://github.com/antgroup/vsag/issues/2069))
- [HGraph](../indexes/hgraph.md)
- [IVF](../indexes/ivf.md)
- [Per-Search Allocator](../advanced/search_allocator.md)
- [Serialization](../advanced/serialization.md)
- Serialization-format compatibility statement.
- Default-value and behavioral changes.
- Build-system / packaging notes.
- Step-by-step upgrade checklist.
