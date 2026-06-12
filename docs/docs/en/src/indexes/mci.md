# MCI

MCI is a dense-vector index in VSAG that combines a k-nearest-neighbor graph with a
maximal-clique candidate structure. Compared with a pure graph walk, MCI spends more work at
build time to organise neighbors into clique-like candidate groups, then uses those groups to
reduce the number of vectors scored at query time.

MCI also provides an optional **HGraph hybrid overlay** for filtered search. In that mode, MCI
remains the serialized primary index, while a separate HGraph index can be loaded through
`hgraph_index_path` and used only when the filter is broad enough.

- Source: `src/algorithm/mci.{h,cpp}`
- Example: [`examples/cpp/322_feature_mci_hybrid_filter.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/322_feature_mci_hybrid_filter.cpp)
- Incremental add guide: [Incremental Add for MCI](../advanced/mci_incremental_add.md)

## How it works

1. **Build or import a KNN graph.** MCI starts from a candidate graph whose degree is capped by
   `mcs`. If `knng_path` is empty, MCI derives the graph internally with ODescent. If
   `knng_path` is set, it reads a fixed-width binary KNNG file instead.
2. **Enumerate clique candidates.** The graph is reorganised into maximal-clique style groups,
   bounded by `clique_max`, so each node can jump to a compact candidate set during search.
3. **Score within candidate sets.** At query time MCI seeds the search with `seed_count`, scans
   clique candidates, and then optionally reorders the final heap if `use_reorder` is enabled.
4. **Route broad filters to HGraph when configured.** If `use_hgraph_hybrid` is enabled and the
   filter's `ValidRatio()` is greater than or equal to `hgraph_valid_ratio_threshold`, MCI can
   forward the request to the external HGraph index instead of using the clique path.

## Quick start

### Build a plain MCI index

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "base_codes_type": "flatten",
        "max_degree": 32,
        "mcs": 200,
        "clique_max": 50
    }
})";

auto index = vsag::Factory::CreateIndex("mci", params).value();

// Populate the base set (replace with your own data source).
int64_t n = 10000;
std::vector<int64_t> ids(n);
std::vector<float> data(n * 128);
// ... fill ids and data ...

auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids.data())->Float32Vectors(data.data())->Owner(false);
index->Build(base);

// Build a single query vector.
std::vector<float> q(128);
// ... fill query vector q ...

auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q.data())->Owner(false);
auto result = index->KnnSearch(
    query, 10, R"({"mci": {"ef_search": 80, "seed_count": 32}})").value();
```

### Enable the HGraph hybrid overlay

```cpp
std::string hybrid_params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "base_codes_type": "flatten",
        "max_degree": 32,
        "mcs": 200,
        "clique_max": 50,
        "use_hgraph_hybrid": true,
        "hgraph_valid_ratio_threshold": 0.2,
        "hgraph_index_path": "/path/to/hgraph.index",
        "hgraph_ef_search": 100,
        "hgraph_index_param": {
            "base_quantization_type": "fp32",
            "graph_type": "odescent",
            "max_degree": 32,
            "alpha": 1.2,
            "graph_iter_turn": 20,
            "neighbor_sample_rate": 0.2
        }
    }
})";

auto hybrid = vsag::Factory::CreateIndex("mci", hybrid_params).value();
std::ifstream input("/path/to/mci.index", std::ios::binary);
hybrid->Deserialize(input);
```

Hybrid is **not** a separate on-disk index type. The on-disk primary index is still the MCI
serialization; `hgraph_index_path` points to the external HGraph companion index loaded by the
overlay.

## Build parameters

MCI uses the generic `index_param` object for build-time parameters.

| Parameter | Type | Typical value | Description |
|-----------|------|---------------|-------------|
| `base_quantization_type` | string | `fp32`, `sq8`, `rabitq` | Quantization used for the base storage |
| `base_codes_type` | string | `flatten` | Base code layout used by the flat data cell |
| `max_degree` | int | `16`-`48` | Maximum out-degree of the clique/search graph |
| `mcs` | int | `64`-`256` | Candidate budget used when building or importing the KNN graph |
| `clique_max` | int | `16`-`64` | Upper bound on the size of a clique candidate group |
| `alpha` | float | `1.2` | ODescent expansion factor when MCI builds its own KNN graph |
| `join_ratio_threshold` | float | `0.6` | Minimum overlap required before an incrementally added vector joins an existing clique |
| `added_mct` | int | `3` | Maximum number of eligible existing cliques an incrementally added vector joins |
| `knng_path` | string | empty | Optional fixed-width binary KNNG file; if unset, MCI builds the graph internally |
| `clique_path` | string | empty | Optional precomputed clique index file |
| `use_reorder` | bool | `false` | Keep a higher-precision copy and rerank final candidates |

### KNNG file format

When `knng_path` is provided, MCI expects a binary file with these properties:

- no header
- one fixed-width row per base vector
- each row stores neighbor ids as `uint32_t` / `InnerIdType`
- all rows have the same degree

The example [`examples/cpp/322_feature_mci_hybrid_filter.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/322_feature_mci_hybrid_filter.cpp)
shows one way to derive such a file from an HGraph index.

## Search parameters

Search-time parameters live under the `mci` object.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ef_search` | int | Number of retained candidates during MCI search |
| `seed_count` | int | Number of seed ids collected before clique expansion |
| `hops_limit` | int | Optional safety cap for graph expansion hops |
| `rabitq_one_bit_search` | bool | Enable RabitQ lower-bound search mode when the underlying codes support it |

```cpp
auto result = index->KnnSearch(
    query, 10, R"({"mci": {"ef_search": 120, "seed_count": 64}})").value();
```

## HGraph hybrid overlay

The hybrid overlay is meant for **filtered KNN** rather than plain unfiltered search.

### Routing rule

MCI routes a filtered request to HGraph only when all of the following are true:

- `use_hgraph_hybrid` is `true`
- the HGraph companion index is loaded and has the same size as the MCI index
- the request uses a `Filter` object rather than a bitset-only blacklist
- `filter->ValidRatio()` is greater than or equal to `hgraph_valid_ratio_threshold`

Otherwise the request stays on the normal MCI path.

### Hybrid-specific build parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `use_hgraph_hybrid` | bool | Enable HGraph-assisted filtered-search routing |
| `hgraph_valid_ratio_threshold` | float | Minimum valid ratio required before routing to HGraph |
| `hgraph_index_path` | string | Path to the serialized external HGraph index |
| `hgraph_ef_search` | int | Default HGraph `ef_search` when the request doesn't supply an `hgraph` search object |
| `hgraph_index_param` | object | Build parameters used to instantiate the companion HGraph index before loading it |

When a query runs, the result statistics include `mci_hybrid_route`,
`mci_hybrid_valid_ratio`, and `mci_hybrid_threshold`, which are useful when checking whether a
filter actually took the HGraph route.

## When to use MCI

- Dense-vector workloads where you want a compact candidate structure instead of a pure graph walk.
- Pipelines that already have an offline KNN graph and want to reuse it through `knng_path`.
- Filtered-search scenarios where narrow predicates stay on MCI, but broader predicates can reuse
  an existing HGraph index through the hybrid overlay.

If your workload is mostly unfiltered and graph-first, compare against [HGraph](hgraph.md). If
your main need is vector + structured attributes rather than id-based filter objects, also see
[Attribute Filter (Hybrid Search)](../advanced/attribute_filter.md).
