# Migrating to VSAG 1.0

> **Status:** living document — sections are being added on PR
> [#2070](https://github.com/antgroup/vsag/pull/2070) as part of the 1.0
> documentation readiness effort tracked in
> [issue #2069](https://github.com/antgroup/vsag/issues/2069).

This page collects everything users coming from **VSAG 0.18.x** (and
earlier) need to know to upgrade smoothly to **VSAG 1.0**. Read it before
you recompile or redeploy.

> 1.0 follows [Semantic Versioning](https://semver.org/). The compatibility
> rules going forward are spelled out in
> [API Stability](api_stability.md); this page focuses on the one-time
> 0.18 → 1.0 migration.

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
[Serialization](../advanced/serialization.md) use `hgraph` as the
default example in 1.0. The legacy examples remain in
`examples/cpp/101_index_hnsw.cpp` and `examples/cpp/102_index_diskann.cpp`
for reference.

## Outline of the remaining sections (to be expanded)

- Deprecated search API and the `SearchRequest` transition.
- `CalDistanceById` typo and `CalcDistancesById` migration path.
- Serialization-format compatibility statement.
- Default-value and behavioral changes.
- Build-system / packaging notes.
- Step-by-step upgrade checklist.
