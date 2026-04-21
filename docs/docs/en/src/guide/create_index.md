# Creating an Index

All VSAG indexes are built through `vsag::Factory::CreateIndex(name, build_params_json)`. The
`name` selects the implementation; `build_params_json` configures dimension, metric, and
index-specific options.

## Supported Index Types

| Name | Description | Example |
|------|-------------|---------|
| `hnsw` | Classic HNSW graph index | `examples/cpp/101_index_hnsw.cpp` |
| `hgraph` | Improved graph index with richer quantization options | `examples/cpp/103_index_hgraph.cpp` |
| `diskann` | Memory-disk hybrid, suitable for billion-scale datasets | `examples/cpp/102_index_diskann.cpp` |
| `ivf` | Inverted file with quantization | `examples/cpp/106_index_ivf.cpp` |
| `pyramid` | Multi-tenant / tag-partitioned graph index | `examples/cpp/107_index_pyramid.cpp` |
| `sindi` | Sparse-vector index (e.g. BM25, SPLADE) | `examples/cpp/109_index_sindi.cpp` |
| `brute_force` | Exact exhaustive search; useful as baseline | `examples/cpp/105_index_brute_force.cpp` |

## Common Top-Level Fields

| Field | Values | Notes |
|-------|--------|-------|
| `dim` | positive integer | Fixed after build |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` | Public API currently uses `float32` |
| `metric_type` | `l2` / `ip` / `cosine` | Must match at query time |

## Examples

### HNSW

```cpp
std::string params = R"(
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "hnsw": {
        "max_degree": 32,
        "ef_construction": 400
    }
}
)";
auto index = vsag::Factory::CreateIndex("hnsw", params).value();
```

### HGraph with FP16 quantization

HGraph uses `index_param` as the build-time sub-object (`hgraph` is reserved for search-time
parameters like `ef_search`). See `examples/cpp/103_index_hgraph.cpp`.

```cpp
std::string params = R"(
{
    "dim": 768,
    "dtype": "float32",
    "metric_type": "ip",
    "index_param": {
        "base_quantization_type": "fp16",
        "max_degree": 32,
        "ef_construction": 400
    }
}
)";
auto index = vsag::Factory::CreateIndex("hgraph", params).value();
```

See [Index Parameters](../resources/index_parameters.md) for the full reference.
