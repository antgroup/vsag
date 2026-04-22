# Indexes

VSAG ships a family of index implementations that share a single builder-style API,
one serialization format, and one set of operations (`Build`, `Add`, `KnnSearch`,
`RangeSearch`, `Remove`, `Serialize` / `Deserialize`, ...). They differ in the data
structures and trade-offs they use under the hood.

The pages in this section cover the actively developed indexes:

| Index | Page | Best for |
|-------|------|----------|
| `hgraph` | [HGraph](hgraph.md) | General-purpose, high-recall graph with rich quantization options |
| `ivf` | [IVF](ivf.md) | Partition-based search, high-throughput batch queries, large corpora |
| `sindi` | [SINDI](sindi.md) | Sparse vectors (BM25 / learned sparse) on inner-product |
| `pyramid` | [Pyramid](pyramid.md) | Multi-tenant or tag-partitioned corpora with hierarchical paths |

`brute_force` is also available as an exact-search baseline (see
[Creating an Index](../guide/create_index.md) and `examples/cpp/105_index_brute_force.cpp`).

`hnsw` and `diskann` are retained for backward compatibility but are **deprecated**; new
deployments should prefer `hgraph` (graph-based) or `ivf` (partition-based) instead.

## Parameter conventions

All indexes share the same top-level build fields:

| Field | Values | Notes |
|-------|--------|-------|
| `dim` | positive integer | Vector dimensionality; fixed after build |
| `dtype` | `float32` / `float16` / `bfloat16` / `int8` / `sparse` | `sparse` is SINDI only |
| `metric_type` | `l2` / `ip` / `cosine` | Must match at query time (SINDI is `ip` only) |

Index-specific build parameters live under the `index_param` sub-object; search-time
parameters live under a sub-object named after the index (e.g. `hgraph`, `ivf`,
`sindi`, `pyramid`). Concrete schemas are documented on each page and enumerated in
[Index Parameters](../resources/index_parameters.md).
