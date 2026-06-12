# Index Parameters

This page summarises the commonly used parameters for every VSAG index type. For the full
enumeration, consult the source:

- Build parameter keys: `src/constants.cpp`
- Public constants: `include/vsag/constants.h`
- Per-index examples: `examples/cpp/101_index_hnsw.cpp` and friends.

## Common Fields

Every index requires these top-level fields at build time:

| Field | Values | Description |
|-------|--------|-------------|
| `dim` | positive integer | Vector dimensionality; cannot change after build |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` | Vector data type; determines internal representation |
| `metric_type` | `l2` / `ip` / `cosine` | Distance metric |

## HNSW

HNSW uses the `hnsw` sub-object for build parameters. It does not accept HGraph-only keys
such as `base_quantization_type`.

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "hnsw": {
        "max_degree": 32,
        "ef_construction": 400,
        "use_conjugate_graph": false
    }
}
```

| Field | Typical | Description |
|-------|---------|-------------|
| `max_degree` | 16–48 | Maximum out-degree per node |
| `ef_construction` | 200–500 | Candidate set size during build; larger = higher recall, slower build |
| `use_conjugate_graph` | bool | Build the [conjugate graph](../advanced/enhance_graph.md) |

At search time:

```json
{"hnsw": {"ef_search": 100, "use_conjugate_graph_search": false}}
```

## HGraph

HGraph places its build parameters under the generic `index_param` key (see
`examples/cpp/103_index_hgraph.cpp`); the `hgraph` key is reserved for search-time parameters.

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 400
    }
}
```

| Field | Typical | Description |
|-------|---------|-------------|
| `max_degree` | 16–48 | Maximum out-degree per node |
| `ef_construction` | 200–500 | Candidate set size during build; larger = higher recall, slower build |
| `base_quantization_type` | `fp32` / `fp16` / `bf16` / `sq8` / `sq4` / `pq` | Quantization of the base storage — see the [Quantization chapter](../quantization/README.md) for all supported values |

At search time:

```json
{"hgraph": {"ef_search": 100}}
```

The `hgraph` search-param object also accepts `brute_force_threshold` (a float
in `[0.0, 1.0]`, default `0.0`). When set above zero and the request carries a
filter whose `ValidRatio()` is at most this threshold, HGraph skips the graph
traversal and runs an exact scan over the surviving ids. See the
[HGraph index page](../indexes/hgraph.md#brute-force-fallback-under-highly-selective-filters-brute_force_threshold)
for details.

## MCI

MCI also uses the generic `index_param` key for build-time parameters, and the `mci` key for
search-time parameters.

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
        "base_quantization_type": "sq8",
        "base_codes_type": "flatten",
        "max_degree": 32,
        "mcs": 200,
        "clique_max": 50,
        "knng_path": "",
        "use_hgraph_hybrid": false,
        "hgraph_valid_ratio_threshold": 1.0,
        "hgraph_index_path": ""
    }
}
```

| Field | Typical | Description |
|-------|---------|-------------|
| `max_degree` | 16-48 | Maximum out-degree of the candidate graph |
| `mcs` | 64-256 | Candidate budget used when building or importing the KNN graph |
| `clique_max` | 16-64 | Maximum clique candidate size |
| `alpha` | 1.2 | ODescent expansion factor when building the graph internally |
| `knng_path` | path or empty | Optional fixed-width binary KNN graph file |
| `clique_path` | path or empty | Optional precomputed clique index |
| `use_hgraph_hybrid` | bool | Enable filtered-search routing to an external HGraph index |
| `hgraph_valid_ratio_threshold` | 0.0-1.0 | Minimum valid ratio required before routing to HGraph |
| `hgraph_index_path` | path | Serialized HGraph companion index |
| `hgraph_ef_search` | 32-200 | Default HGraph `ef_search` for hybrid-routed queries |

At search time:

```json
{"mci": {"ef_search": 80, "seed_count": 32}}
```

## DiskANN

```json
{
    "diskann": {
        "max_degree": 32,
        "ef_construction": 400,
        "pq_sample_rate": 0.1,
        "pq_dims": 32,
        "use_async_io": true
    }
}
```

## IVF

```json
{
    "ivf": {
        "nlist": 4096,
        "base_quantization_type": "sq8",
        "nprobe": 32
    }
}
```

## Brute Force

```json
{"brute_force": {}}
```

No extra parameters.

## Pyramid

Pyramid supports organising multiple subgraphs by tag:

```json
{
    "pyramid": {
        "tag_dim": 1,
        "max_degree": 24,
        "ef_construction": 300
    }
}
```

## SINDI (sparse vectors)

```json
{
    "sindi": {
        "top_k": 32,
        "doc_prune_ratio": 0.1
    }
}
```

## Runtime Parameters

Beyond build-time parameters, `Index::Tune` and `SearchParam` tweak runtime settings such as
`ef_search` and `nprobe`. See [Optimizer](../advanced/optimizer.md) and the
`examples/cpp/3xx_feature_*.cpp` examples.
