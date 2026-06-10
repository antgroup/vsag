# Creating an Index

All VSAG indexes are built through `vsag::Factory::CreateIndex(name, build_params_json)`. The
`name` selects the implementation; `build_params_json` configures dimension, metric, and
index-specific options.

## Supported Index Types

| Name | Description | Page | Example |
|------|-------------|------|---------|
| `hgraph` | Improved graph index with richer quantization options | [HGraph](../indexes/hgraph.md) | `examples/cpp/103_index_hgraph.cpp` |
| `ivf` | Inverted file with quantization | [IVF](../indexes/ivf.md) | `examples/cpp/106_index_ivf.cpp` |
| `sindi` | Sparse-vector index (e.g. BM25, SPLADE) | [SINDI](../indexes/sindi.md) | `examples/cpp/109_index_sindi.cpp` |
| `pyramid` | Multi-tenant / tag-partitioned graph index | [Pyramid](../indexes/pyramid.md) | `examples/cpp/107_index_pyramid.cpp` |
| `brute_force` | Exact exhaustive search; useful as baseline | — | `examples/cpp/105_index_brute_force.cpp` |
| `hnsw` | Classic HNSW graph index (**deprecated** — prefer `hgraph`) | — | `examples/cpp/101_index_hnsw.cpp` |
| `diskann` | Memory-disk hybrid (**deprecated** — prefer `ivf`) | — | `examples/cpp/102_index_diskann.cpp` |

## Common Top-Level Fields

| Field | Values | Notes |
|-------|--------|-------|
| `dim` | positive integer | Fixed after build |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` | Public API currently uses `float32` |
| `metric_type` | `l2` / `ip` / `cosine` | Must match at query time |

## Examples

### HGraph

HGraph uses `index_param` as the build-time sub-object (`hgraph` is reserved for search-time
parameters like `ef_search`). See `examples/cpp/103_index_hgraph.cpp`.

```cpp
std::string params = R"(
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
)";
auto index = vsag::Factory::CreateIndex("hgraph", params).value();
```

### HGraph with SQ8 quantization

Switch `base_quantization_type` to `sq8` to store base vectors as 8-bit scalar-quantized codes
(roughly a 4× reduction versus `fp32`) with minimal recall impact; other quantization types
(`fp16`, `bf16`, `pq`, …) are selected the same way.

```cpp
std::string params = R"(
{
    "dim": 768,
    "dtype": "float32",
    "metric_type": "ip",
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "ef_construction": 400
    }
}
)";
auto index = vsag::Factory::CreateIndex("hgraph", params).value();
```

See [Index Parameters](../resources/index_parameters.md) for the full reference.
