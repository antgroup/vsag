# VSAG C++ Examples

Small, self-contained C++ programs that show how to build, query, and
manage VSAG indexes from native code. Each file is independent: pick
one, read it top-to-bottom, then adapt it to your data.

## Prerequisites

- A C++17 (or newer) compiler (`clang-format` / `clang-tidy` 15 are
  required only for development, not for running these examples).
- CMake 3.18+.
- The build prerequisites listed in
  [`docs/agents/build-and-test.md`](../../docs/agents/build-and-test.md).

## Build

Examples are gated behind `ENABLE_EXAMPLES=ON`. The shortest path is
`make dev`, which enables tests, Python bindings, tools, mockimpl, and
the examples in a single configure:

```bash
make dev
# binaries land in build/examples/cpp/
./build/examples/cpp/101_index_hnsw
```

To build only one example without the full developer set:

```bash
cmake -B build -DENABLE_EXAMPLES=ON
cmake --build build --target 101_index_hnsw
./build/examples/cpp/101_index_hnsw
```

## File naming convention

Examples are grouped by a three-digit prefix so related ones sit
together when the directory is listed:

| Prefix | Topic |
| --- | --- |
| `1xx` | Index types — one example per supported algorithm. |
| `2xx` | Resource management — allocator, logger, thread pool injection. |
| `3xx` | Per-search and per-index features — filter, range search, remove, update, introspection, tuning. |
| `4xx` | Persistence — serializing and streaming an index in/out of storage. |
| `5xx` | Quantization — quantizer-level recipes. |

## Examples

### Index types

| File | Index | Notes |
| --- | --- | --- |
| [`101_index_hnsw.cpp`](101_index_hnsw.cpp) | HNSW | The shortest "build + KNN" round-trip; start here. |
| [`102_index_diskann.cpp`](102_index_diskann.cpp) | DiskANN | Disk-resident graph; links against `vsag_static`. |
| [`103_index_hgraph.cpp`](103_index_hgraph.cpp) | HGraph | VSAG's headline in-memory graph index. |
| [`104_index_fresh_hnsw.cpp`](104_index_fresh_hnsw.cpp) | Fresh HNSW | HNSW variant that supports incremental updates. |
| [`105_index_brute_force.cpp`](105_index_brute_force.cpp) | BruteForce | Exact reference for recall calibration. |
| [`106_index_ivf.cpp`](106_index_ivf.cpp) | IVF | Inverted file, tuned for large-`k` / batch queries. |
| [`107_index_pyramid.cpp`](107_index_pyramid.cpp) | Pyramid | Multi-tenant index. |
| [`108_index_gno_imi.cpp`](108_index_gno_imi.cpp) | GNO-IMI | IVF variant with multi-index partitioning. |
| [`109_index_sindi.cpp`](109_index_sindi.cpp) | SINDI | Sparse vector index (text / inverted retrieval). |
| [`110_index_warp.cpp`](110_index_warp.cpp) | Warp | Hybrid index. |
| [`111_index_lazy_hgraph.cpp`](111_index_lazy_hgraph.cpp) | LazyHGraph | Starts as exact BruteForce for small data, then converts to HGraph after a threshold. |
| [`316_index_int8_hgraph.cpp`](316_index_int8_hgraph.cpp) | HGraph + INT8 | HGraph with INT8-quantized vectors. |
| [`321_index_fp16_hgraph.cpp`](321_index_fp16_hgraph.cpp) | HGraph + FP16 | HGraph with FP16-quantized vectors. |

> Note: `316_*` and `321_*` carry a `3xx` numeric prefix for historical
> reasons but are topically index-type demos, so they are grouped here
> rather than under "Per-index / per-search features" below.

### Resource management (`2xx`)

| File | What it shows |
| --- | --- |
| [`201_custom_allocator.cpp`](201_custom_allocator.cpp) | Plug in a custom `vsag::Allocator` for per-index memory isolation. |
| [`202_custom_logger.cpp`](202_custom_logger.cpp) | Redirect VSAG's logs to your own sink. |
| [`203_custom_thread_pool.cpp`](203_custom_thread_pool.cpp) | Inject a custom `vsag::ThreadPool` to control parallelism. |

### Per-index / per-search features (`3xx`)

| File | What it shows |
| --- | --- |
| [`301_feature_filter.cpp`](301_feature_filter.cpp) | Filtered KNN search via `vsag::Filter`. |
| [`302_feature_range_search.cpp`](302_feature_range_search.cpp) | Range search (`RangeSearch`). |
| [`303_feature_remove.cpp`](303_feature_remove.cpp) | Soft-remove ids from an index. |
| [`304_feature_enhance_graph.cpp`](304_feature_enhance_graph.cpp) | Continuous graph enhancement (Conjugate Graph). |
| [`305_feature_update.cpp`](305_feature_update.cpp) | In-place id/vector update. |
| [`306_feature_calculate_distance_by_id.cpp`](306_feature_calculate_distance_by_id.cpp) | `CalcDistanceById` API. |
| [`307_feature_check_features.cpp`](307_feature_check_features.cpp) | Query an index's capability flags (`CheckFeature`). |
| [`308_feature_estimate_memory.cpp`](308_feature_estimate_memory.cpp) | Pre-build memory estimation. |
| [`309_feature_clone.cpp`](309_feature_clone.cpp) | Clone an index without re-building. |
| [`310_feature_export_model.cpp`](310_feature_export_model.cpp) | Export the trained model component. |
| [`311_feature_train.cpp`](311_feature_train.cpp) | Train a model separately from build. |
| [`312_feature_odescent.cpp`](312_feature_odescent.cpp) | OD-Escent graph construction. |
| [`313_feature_search_allocator.cpp`](313_feature_search_allocator.cpp) | Per-search scratch allocator. |
| [`314_feature_hgraph_search_allocator.cpp`](314_feature_hgraph_search_allocator.cpp) | Per-search scratch allocator on HGraph. |
| [`315_feature_hgraph_merge.cpp`](315_feature_hgraph_merge.cpp) | Merge multiple HGraph indexes (FGIM). |
| [`317_feature_get_detail_data.cpp`](317_feature_get_detail_data.cpp) | Introspect raw index detail data. |
| [`318_feature_tune.cpp`](318_feature_tune.cpp) | Online `Tune()` optimizer. |
| [`319_feature_get_memory_usage.cpp`](319_feature_get_memory_usage.cpp) | Live memory-usage reporting. |
| [`320_feature_extra_info.cpp`](320_feature_extra_info.cpp) | Attach per-vector extra info / payload. |
| [`322_feature_hgraph_brute_force_threshold.cpp`](322_feature_hgraph_brute_force_threshold.cpp) | HGraph search-time `brute_force_threshold`: automatically switch to an exact scan under highly selective filters. |
| [`324_feature_lazy_hgraph_extra_info.cpp`](324_feature_lazy_hgraph_extra_info.cpp) | LazyHGraph `extra_info` filtering across flat and graph phases. |

### Persistence (`4xx`)

| File | What it shows |
| --- | --- |
| [`401_persistent_kv.cpp`](401_persistent_kv.cpp) | Serialize / deserialize against a key-value backend. |
| [`402_persistent_streaming.cpp`](402_persistent_streaming.cpp) | Streaming serialization for large indexes. |
| [`403_persistent_streaming_load.cpp`](403_persistent_streaming_load.cpp) | Static `Index::Load` from streaming serialization across index types. |
| [`404_persistent_streaming_load_hybrid.cpp`](404_persistent_streaming_load_hybrid.cpp) | Static `Index::Load` for a hybrid HGraph index with memory and disk-backed RaBitQ split codes. |

### Quantization (`5xx`)

| File | What it shows |
| --- | --- |
| [`501_quantization_transform.cpp`](501_quantization_transform.cpp) | Transform Quantizer (TQ) recipe. |

## Where to go next

- For the equivalent recipes in other languages, see
  [`examples/python/`](../python/) and [`examples/typescript/`](../typescript/).
- For deeper conceptual material, see the user guide at
  [https://vsag.io/docs](https://vsag.io/docs) or its in-repo source
  under [`docs/docs/en/src/`](../../docs/docs/en/src/).
