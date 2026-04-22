# HGraph

HGraph is VSAG's flagship **graph-based** index. It builds a hierarchical proximity graph
similar in spirit to HNSW, but with a richer set of quantization options, a unified
build-parameter schema (`index_param`), and first-class support for reordering,
incremental updates, deletion, and ELP-based runtime tuning.

For most dense-vector workloads (text / image / multimodal embeddings, 64–4096 dims,
from a few thousand up to hundreds of millions of points), HGraph is the recommended
default.

- Source: `src/algorithm/hgraph.{h,cpp}`
- Example: [`examples/cpp/103_index_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/103_index_hgraph.cpp)

## How it works

1. **Graph construction.** Vectors are organised in a layered proximity graph; upper
   layers act as navigation aids, the bottom layer connects every data point to its
   nearest neighbours within a `max_degree` budget. The construction algorithm can be
   either NSW-style insertion (`graph_type: "nsw"`, the default) or ODescent
   (`graph_type: "odescent"`).
2. **Quantization.** The base storage is compressed with a configurable quantizer
   (`base_quantization_type` — `fp32`, `fp16`, `bf16`, `sq8`, `sq8_uniform`, `sq4_uniform`,
   `pq`, `pqfs`, `rabitq`). Optionally, a second high-precision copy is kept
   (`use_reorder: true` with `precise_quantization_type`) and used to re-rank the
   candidates returned by the coarse search.
3. **Search.** Greedy beam search traverses the graph top-down, expanding the current
   frontier up to `ef_search` candidates. When reordering is enabled, the final list is
   re-scored against the precise representation.

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
        "ef_construction": 400
    }
})";
auto index = vsag::Factory::CreateIndex("hgraph", params).value();

// Build.
auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Build(base);

// Search.
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10, R"({"hgraph": {"ef_search": 100}})").value();
```

## Build parameters

Build-time parameters live under `index_param`. The table below highlights the keys
most users need; the exhaustive list is in [Index Parameters](../resources/index_parameters.md)
and `docs/hgraph.md` in the repository.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_quantization_type` | string | — (required) | `fp32`, `fp16`, `bf16`, `sq8`, `sq8_uniform`, `sq4_uniform`, `pq`, `pqfs`, `rabitq` |
| `max_degree` | int | `64` | Maximum out-degree per graph node |
| `ef_construction` | int | `400` | Candidate list size during build (higher = better recall, slower build) |
| `graph_type` | string | `"nsw"` | Graph algorithm: `nsw` or `odescent` |
| `use_reorder` | bool | `false` | Keep a high-precision copy and re-rank after the coarse search |
| `precise_quantization_type` | string | `"fp32"` | Quantizer used for reordering (takes effect only with `use_reorder: true`) |
| `base_pq_dim` | int | `128` | Number of PQ subspaces (required with `pq` / `pqfs`) |
| `build_thread_count` | int | `100` | Threads used to parallelise build |
| `support_duplicate` | bool | `false` | Enable duplicate-ID detection on insert |
| `support_remove` | bool | `false` | Enable `Remove()` on the built index |
| `store_raw_vector` | bool | `false` | Keep the raw vector in addition to the quantized copy (useful for `cosine`) |
| `use_elp_optimizer` | bool | `false` | Auto-tune search parameters after build |
| `base_io_type` / `precise_io_type` | string | `"block_memory_io"` | Storage backend (`memory_io`, `block_memory_io`, `buffer_io`, `async_io`, `mmap_io`) |
| `base_file_path` / `precise_file_path` | string | `"./default_file_path"` | File path when using a disk-backed IO type |
| `hgraph_init_capacity` | int | `100` | Initial capacity hint (doesn't cap the final size) |

## Search parameters

Search-time parameters live under the `hgraph` sub-object:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ef_search` | int | Size of the search frontier. Larger = higher recall, slower query. |

```cpp
auto result = index->KnnSearch(
    query, topk, R"({"hgraph": {"ef_search": 200}})").value();
```

## When to use HGraph

- Dense float vectors with dimensions roughly between 64 and 4096.
- Latency-sensitive queries where high recall matters.
- Mixed workloads with incremental insertion (optionally deletion via `support_remove`).
- Memory-constrained deployments that benefit from `sq8` / `sq4_uniform` / `pq` — often
  in combination with `use_reorder` to recover recall.

If your workload is partition-heavy (coarse-grained buckets scanned per query) or
strongly I/O-bound on a SSD, compare against [IVF](ivf.md) before committing to HGraph.

## See also

- [Creating an Index](../guide/create_index.md)
- [Graph Enhancement](../advanced/enhance_graph.md)
- [Optimizer (Tune)](../advanced/optimizer.md)
- [Serialization](../advanced/serialization.md)
