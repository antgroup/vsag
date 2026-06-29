# Index Parameters

This page summarises the commonly used parameters for every VSAG index type. For the full
enumeration, consult the source:

- Build parameter keys: `src/constants.cpp`
- Public constants: `include/vsag/constants.h`
- Per-index examples: the `examples/cpp/*_index_*.cpp` files (e.g. `103_index_hgraph.cpp`).

## Common Fields

Every index requires these top-level fields at build time:

| Field | Values | Description |
|-------|--------|-------------|
| `dim` | positive integer | Vector dimensionality; cannot change after build |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` | Vector data type; determines internal representation |
| `metric_type` | `l2` / `ip` / `cosine` | Distance metric |

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
