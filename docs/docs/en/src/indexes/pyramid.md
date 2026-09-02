# Pyramid

![Pyramid: a tree of per-node proximity sub-graphs keyed by a path string; the search walks down the tree along the query's path prefix and runs ef_search inside the leaf sub-graph](../figures/indexes/pyramid-overview.svg)

Pyramid is VSAG's **hierarchical, path-partitioned** graph index. Every vector is
tagged with a path string such as `"a/d/f"`, and Pyramid builds a graph per node
in that path tree. At query time you supply a path prefix, and Pyramid restricts
the search to the corresponding sub-tree.

This is ideal for multi-tenant deployments, tag-partitioned catalogs, or any
scenario where one logical index serves many groups that must not cross-contaminate
results.

- Source: `src/algorithm/pyramid.{h,cpp}`, `src/algorithm/pyramid_zparameters.{h,cpp}`
- Example (single hierarchy): [`examples/cpp/107_index_pyramid.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/107_index_pyramid.cpp)
- Example (multi-hierarchy): [`examples/cpp/112_index_pyramid_multi_hierarchy.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/112_index_pyramid_multi_hierarchy.cpp)

## How it works

1. **Path tree.** Each vector carries a `path` in addition to its id. Paths use
   `/` as separator (e.g. `"tenant_a/lang_en/topic_news"`). Pyramid builds one
   sub-index for every path prefix seen during build.
2. **Per-level sub-graphs.** By default every level gets its own proximity graph.
   Use `no_build_levels` to skip levels that are too small or too coarse to
   benefit from graph indexing — those levels still exist as passthrough
   containers, but search degrades to a scan.
3. **Graph construction.** Each sub-graph is built with the same machinery as
   HGraph: `nsw` insertion or `odescent` with `graph_iter_turn`,
   `neighbor_sample_rate`, and `alpha` for pruning. Base vectors are stored in
   `base_quantization_type`; optional reordering keeps a high-precision copy.
4. **Search.** Query vectors also carry a path. The search walks down the tree
   to the most specific sub-graph matching the query path and runs a graph search
   there with `ef_search` (and `subindex_ef_search` for intermediate levels).

## Quick start

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "alpha": 1.2,
        "graph_type": "odescent",
        "graph_iter_turn": 15,
        "neighbor_sample_rate": 0.2,
        "no_build_levels": [0, 1],
        "use_reorder": true,
        "build_thread_count": 16
    }
})";
auto index = vsag::Factory::CreateIndex("pyramid", params).value();

// Build with per-vector paths.
auto base = vsag::Dataset::Make();
base->NumElements(n)
    ->Dim(128)
    ->Ids(ids)
    ->Paths(paths)          // std::string* of length n, e.g. "a/d/f"
    ->Float32Vectors(data)
    ->Owner(false);
index->Build(base);

// Search restricted to a path prefix.
std::string query_path = "a/d";
auto query = vsag::Dataset::Make();
query->NumElements(1)
    ->Dim(128)
    ->Float32Vectors(q)
    ->Paths(&query_path)
    ->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10,
    R"({"pyramid": {"ef_search": 100}})").value();
```

## Input data type

The public `Build`, `Add`, and search paths currently accept FP32 vectors supplied with `Dataset::Float32Vectors`; set `dtype` to `"float32"`. `base_quantization_type` selects internal encoding and storage and does not by itself enable FP16, BF16, or INT8 input.

## Build parameters

Build-time parameters live under `index_param`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_quantization_type` | string | — | Coarse storage quantizer (`fp32`, `fp16`, `bf16`, `sq8`, `sq4`, `sq8_uniform`, `sq4_uniform`, `pq`, `pqfs`, `rabitq`, `tq`). See the [Quantization chapter](../quantization/README.md) for per-quantizer details. |
| `tq_chain` | string | — | Transform chain used when `base_quantization_type` is `tq`, for example `"mrle, rabitq"`. |
| `mrle_dim` | int | `0` | Prefix dimension retained by MRLE; `0` keeps the input dimension. |
| `max_degree` | int | `64` | Maximum out-degree per node within a sub-graph. |
| `graph_type` | string | `"nsw"` | `nsw` or `odescent`. |
| `ef_construction` | int | `400` | Candidate list size for `nsw` builds. |
| `alpha` | float | `1.2` | Pruning factor during graph construction. |
| `graph_iter_turn` | int | — | ODescent iterations (effective with `graph_type: "odescent"`). |
| `neighbor_sample_rate` | float | — | ODescent neighbor sampling rate. |
| `no_build_levels` | int[] | `[]` | Tree levels that skip graph construction (0-indexed from the root). |
| `use_reorder` | bool | `false` | Keep a high-precision copy for rescoring. |
| `precise_quantization_type` | string | `"fp32"` | Quantizer for reordering. Use `"rabitq"` with `rabitq_bits_per_dim_precise` to enable RaBitQ x+y split reorder from base storage. |
| `rabitq_bits_per_dim_base` | int | `1` | RaBitQ stored-code bits. In x+y split mode, this is `x`, the filter bits used during graph traversal; allowed range is `[1, 8]`. |
| `rabitq_bits_per_dim_precise` | int | unset | RaBitQ split `y` bits. When set with `base_quantization_type: "rabitq"` and `precise_quantization_type: "rabitq"`, Pyramid uses split storage; `rabitq_bits_per_dim_base` remains `x`, and `x + y <= 8`. |
| `fast_encode_rabitq` | bool | `true` | Use the fast multi-bit RaBitQ encoder for RaBitQ base or precise storage; set to `false` for the exact encoder. |
| `fast_encode_rabitq_rounds` | int | `6` | Fast RaBitQ refinement rounds in `[1, 32]`. |
| `base_io_type` / `precise_io_type` | string | `"block_memory_io"` | Base and reorder storage backends; `uring_io` is available in builds with liburing. |
| `base_file_path` / `precise_file_path` | string | — | Required for disk-backed storage such as `buffer_io`, `async_io`, `uring_io`, or `mmap_io`. |
| `store_raw_vector` | bool | `false` | Preserve an FP32 copy for `GetRawVectorByIds` and precise distance-by-id calculations. |
| `store_paths` | bool | `false` | Top-level switch that preserves the original paths supplied to `Build` and `Add` so `GetDataByIdsWithFlag` can return them when `DATA_FLAG_PATH` is selected. It applies to every configured hierarchy and cannot be overridden per hierarchy. |
| `index_min_size` | int | `0` | Minimum sub-index size; smaller groups fall back to scan. |
| `support_duplicate` | bool | `false` | Allow duplicate ids. |
| `build_thread_count` | int | `1` | Threads used for parallel build. |
| `hierarchies` | array | `[]` | Named hierarchy definitions. Each element is either a string (inherits all top-level params) or an object with `name` and optional overrides (`max_degree`, `ef_construction`, `alpha`, `no_build_levels`, `index_min_size`). When present, multi-hierarchy mode is activated and each hierarchy maintains its own independent path tree. |

### RaBitQ split configuration

Set all five parameters together to enable RaBitQ x+y split storage and reordering:

```json
{
    "use_reorder": true,
    "base_quantization_type": "rabitq",
    "precise_quantization_type": "rabitq",
    "rabitq_bits_per_dim_base": 3,
    "rabitq_bits_per_dim_precise": 5
}
```

Pyramid uses split-code code-to-code distances for incremental flat-to-graph promotion, so it does
not retain an internal FP32 copy by default. Set `store_raw_vector` to `true` at build time when
raw-vector access and complete analyzer metrics are required.

### MRLE with split RaBitQ

Pyramid can truncate embeddings trained with Matryoshka Representation Learning before encoding
them as split RaBitQ codes:

```json
{
    "base_quantization_type": "tq",
    "tq_chain": "mrle, rabitq",
    "mrle_dim": 1280,
    "precise_quantization_type": "rabitq",
    "rabitq_bits_per_dim_base": 3,
    "rabitq_bits_per_dim_precise": 5,
    "use_reorder": true
}
```

The 3-bit filter planes are used for graph traversal and the 5-bit supplement planes for
reordering; both encode the same truncated vector. Pyramid uses split code-to-code distances for
graph promotion and does not retain the original FP32 vectors. Metrics that require decodable
vectors are marked unavailable unless `store_raw_vector` was enabled at build time. Truncation can
reduce recall unless the embedding model was trained for prefix dimensions.

## Build cache

`ExportCache` captures per-hierarchy, per-node NSW graph seeds and `ImportCache` makes them available to a later `Build`. Cache data uses the index cache payload format, not the streaming index serialization format. Set `persist_source_id: true` before footer-serializing an index whose cache will be reused, and provide a unique `Dataset::SourceID` for every vector in both builds. Cache warm builds apply only to `graph_type: "nsw"`; ODescent, duplicate-ID mode, missing source IDs, and duplicate source IDs automatically fall back to a normal cold build. `ef_construction` is not an eligibility condition for the cache path. Fully restored cached graph rows are retained, while cache misses are constructed from the current vectors.

## Search parameters

Search-time parameters live under the `pyramid` sub-object:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ef_search` | int | `100` | Candidate list size for the leaf-level graph search. |
| `hops_limit` | int | unlimited | Hard cap on hops for root-graph KNN search; ignored when it is not greater than `ef_search`. |
| `subindex_ef_search` | int | `50` | Candidate list size used when traversing intermediate sub-graphs on the path. |
| `hierarchies` | string[] | `[]` | Select which hierarchy to search. Empty means use the default (unnamed) hierarchy. |
| `hierarchy_op` | string | `"single"` | How to combine results across hierarchies: `single` (search one hierarchy), `union`, or `intersection`. **Note:** `union` and `intersection` are not yet implemented — setting them will cause `KnnSearch`/`RangeSearch` to return an error. |
| `rabitq_error_rate` | float | `1.9` | Positive lower-bound error multiplier for this search. The default `1.9` is relatively large; increasing it improves accuracy but slows down search. |

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"pyramid": {"ef_search": 200, "subindex_ef_search": 80}})").value();
```

## Multi-Hierarchy Support

A single Pyramid index can maintain **multiple independent path trees**, each
identified by a name (e.g. `"site"`, `"category"`). Vectors share IDs and data
across all hierarchies — only the paths differ. Each hierarchy can optionally
override graph construction parameters.

This is useful when the same set of vectors needs to be partitioned along
different dimensions simultaneously. For example, an e-commerce platform might
partition products by **site** (`site-a/lang-en`) and by **category**
(`electronics/phones`) at the same time, and search can target either hierarchy
independently.

### Build configuration

Add a `hierarchies` array inside `index_param`. Each element is either:
- A **string** (inherits all top-level params): `"site"`
- An **object** with `name` and optional per-hierarchy overrides:
  `{"name": "category", "max_degree": 64, "no_build_levels": [0]}`

Overridable per-hierarchy parameters: `max_degree`, `ef_construction`, `alpha`,
`no_build_levels`, `index_min_size`.

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "alpha": 1.2,
        "graph_type": "odescent",
        "graph_iter_turn": 15,
        "neighbor_sample_rate": 0.2,
        "no_build_levels": [0, 1],
        "use_reorder": true,
        "build_thread_count": 16,
        "hierarchies": [
            "site",
            {"name": "category", "max_degree": 64, "no_build_levels": [0]}
        ]
    }
}
```

### Dataset API for named hierarchies

Use the overloaded `Paths(hierarchy_name, paths)` method to assign paths per
hierarchy. The same `Ids()` and `Float32Vectors()` are shared across all
hierarchies:

```cpp
auto base = vsag::Dataset::Make();
base->NumElements(n)
    ->Dim(128)
    ->Ids(ids)
    ->Float32Vectors(data)
    ->Paths("site", site_paths)         // std::string* of length n
    ->Paths("category", category_paths) // independent paths for 2nd hierarchy
    ->Owner(false);
index->Build(base);
```

### Retrieving paths by ID

Set the top-level build parameter `store_paths` to `true` to retain the original paths for ID-based
retrieval. Select `DATA_FLAG_PATH` in `GetDataByIdsWithFlag`; the returned paths follow the requested
ID order. Use `GetPaths()` for the default unnamed hierarchy and `GetPaths(hierarchy_name)` for a
named hierarchy:

```cpp
int64_t requested_ids[] = {product_id_b, product_id_a};
auto data = index->GetDataByIdsWithFlag(
    requested_ids, 2, DATA_FLAG_ID | DATA_FLAG_PATH).value();

const std::string* site_paths = data->GetPaths("site");
const std::string* category_paths = data->GetPaths("category");
```

`GetDataByIds` and `GetDataByIdsWithFlag` calls without `DATA_FLAG_PATH` do not attach path arrays.
Selecting `DATA_FLAG_PATH` while `store_paths` is `false` returns an invalid-argument error. When
path storage is enabled, a hierarchy is included only if every requested ID has a recorded path in
that hierarchy. If even one requested ID was built or added without that hierarchy's path, its
getter returns `nullptr`; other hierarchies whose requested paths are complete are still returned.
In single-hierarchy mode, the same completeness rule applies to `GetPaths()`.

### Searching a specific hierarchy

Specify which hierarchy to search via `"hierarchies"` in the search parameters.
The query dataset must also set its path on the matching hierarchy name:

```cpp
auto query = vsag::Dataset::Make();
query->NumElements(1)
    ->Dim(128)
    ->Float32Vectors(q)
    ->Paths("site", &query_path)   // target the "site" hierarchy
    ->Owner(false);

auto result = index->KnnSearch(
    query, /*topk=*/10,
    R"({"pyramid": {"ef_search": 100, "hierarchies": ["site"]}})").value();
```

### Incremental insertion (Add)

`Add()` works the same as `Build()` — provide named paths and the index inserts
into all matching hierarchies:

```cpp
auto new_data = vsag::Dataset::Make();
new_data->NumElements(count)
    ->Dim(128)
    ->Ids(new_ids)
    ->Float32Vectors(new_vectors)
    ->Paths("site", new_site_paths)
    ->Paths("category", new_cat_paths);
index->Add(new_data);
```

### RangeSearch

RangeSearch also supports hierarchy selection via the same search parameters:

```cpp
auto result = index->RangeSearch(
    query, /*radius=*/20.0f,
    R"({"pyramid": {"ef_search": 100, "hierarchies": ["category"]}})").value();
```

### Serialize & Deserialize

Multi-hierarchy indexes serialize and deserialize transparently. The serialized format includes
all hierarchy names and their graph structures. With `store_paths: true`, both regular and
streaming serialization also persist the retained original paths, making them available through
`GetDataByIdsWithFlag` after deserialization. With the default `false`, the graph hierarchy is
persisted but the original per-ID paths are not:

```cpp
// Serialize
auto binary_set = index->Serialize().value();

// Deserialize into a new index (must use the same build params)
auto new_index = vsag::Factory::CreateIndex("pyramid", build_params).value();
new_index->Deserialize(binary_set);
```

## When to use Pyramid

- Multi-tenant services where each tenant must see results only from its own
  partition, and you would otherwise maintain one index per tenant.
- Content catalogs with hierarchical tags (language / region / category) where
  queries always scope to a known prefix.
- Workloads with many small partitions: `no_build_levels` and `index_min_size`
  let you skip graph construction for partitions too small to benefit.

If you don't need path-based scoping, [HGraph](hgraph.md) is simpler and generally
faster.

Use [Index Analysis](../resources/analyze_index.md) to inspect Pyramid tree structure,
per-subindex quality, sampled base recall, and duplicate ratios reported by `GetStats()`.
`AnalyzeIndexBySearch` also reports path-scoped query recall, distance, latency, and, when reorder
is enabled, quantization metrics. Its query dataset must carry the same default or named-hierarchy
paths required by `KnnSearch`; when paths are required or supplied for a batched dataset, provide
one path per query. The `analyze_index` tool cannot currently load hierarchy paths from its dense
query file, so use the C++ API for path-scoped dynamic analysis.

## Mark remove

Pyramid supports `RemoveMode::MARK_REMOVE`. Calling `Remove(ids)` (the default
mode) tombstones the given ids: they are excluded from subsequent search results,
`GetNumElements()` drops by the number removed, and `GetNumberRemoved()` reports
the running total. Removing an id that is absent or already removed is a no-op.
`RemoveMode::FORCE_REMOVE` is not supported and returns an error.

Mark-removed vectors still occupy memory until the index is rebuilt; the space is
not physically reclaimed.

## See also

- [Creating an Index](../guide/create_index.md)
- [Index Parameters](../resources/index_parameters.md)
- [Graph Enhancement](../advanced/enhance_graph.md)
- [HGraph](hgraph.md)
