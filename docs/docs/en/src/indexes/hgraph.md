# HGraph

![HGraph: hierarchical proximity graph with top-down greedy search and optional reorder](../figures/indexes/hgraph-overview.svg)

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
   (`base_quantization_type` — `fp32`, `fp16`, `bf16`, `sq8`, `sq4`, `sq8_uniform`, `sq4_uniform`,
   `pq`, `pqfs`, `rabitq`, `tq`). Optionally, a second high-precision copy is kept
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
| `base_quantization_type` | string | — (required) | `fp32`, `fp16`, `bf16`, `sq8`, `sq4`, `sq8_uniform`, `sq4_uniform`, `pq`, `pqfs`, `rabitq`, `tq` — see the [Quantization chapter](../quantization/README.md) for per-quantizer details |
| `max_degree` | int | `64` | Maximum out-degree per graph node |
| `ef_construction` | int | `400` | Candidate list size during build (higher = better recall, slower build) |
| `graph_type` | string | `"nsw"` | Graph algorithm: `nsw` or `odescent` |
| `use_reorder` | bool | `false` | Keep a high-precision copy and re-rank after the coarse search |
| `precise_quantization_type` | string | `"fp32"` | Quantizer used for reordering (takes effect only with `use_reorder: true`) |
| `base_pq_dim` | int | `1` | Number of PQ subspaces. When using `pq` / `pqfs`, set this explicitly instead of relying on the default. |
| `build_thread_count` | int | `100` | Threads used to parallelise build |
| `support_duplicate` | bool | `false` | Enable duplicate-ID detection on insert |
| `duplicate_distance_threshold` | float | `0.0` | Duplicate-detection distance threshold. When greater than `0`, deduplicate by the nearest candidate distance; when `0`, fall back to the current code `memcmp` check |
| `support_remove` | bool | `false` | Enable `Remove()` on the built index |
| `store_raw_vector` | bool | `false` | Keep the raw vector in addition to the quantized copy (useful for `cosine`) |
| `use_elp_optimizer` | bool | `false` | Auto-tune search parameters after build |
| `base_io_type` / `precise_io_type` | string | `"block_memory_io"` | Storage backend (`memory_io`, `block_memory_io`, `buffer_io`, `async_io`, `mmap_io`) |
| `base_file_path` / `precise_file_path` | string | — | File path; required when the corresponding `*_io_type` is disk-backed (`buffer_io`, `async_io`, `mmap_io`) |
| `hgraph_init_capacity` | int | `100` | Initial capacity hint (doesn't cap the final size) |

## Supported input data types

The `dtype` field in the top-level build config selects how `Dataset` interprets the raw vector
bytes. HGraph supports four input types; the `dtype` value, the corresponding `Dataset` setter,
and the example demonstrating each combination are summarised below.

| `dtype`     | Element type | `Dataset` setter         | Example                                                                                                |
|-------------|--------------|--------------------------|--------------------------------------------------------------------------------------------------------|
| `float32`   | `float`      | `Float32Vectors`         | [`103_index_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/103_index_hgraph.cpp) |
| `int8`      | `int8_t`     | `Int8Vectors`            | [`316_index_int8_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/316_index_int8_hgraph.cpp) |
| `float16`   | `uint16_t` (IEEE 754 binary16, bit-pattern packed) | `Float16Vectors` | [`321_index_fp16_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/321_index_fp16_hgraph.cpp) |
| `bfloat16`  | `uint16_t` (Brain Float, bit-pattern packed) | `Float16Vectors` (shared with FP16) | adapt `321_index_fp16_hgraph.cpp` per the notes below                                                  |

The `dim` value is the logical vector dimensionality (number of elements), not the byte length, so
the same `dim` is reused across all four data types.

### `int8` input

Quantized `int8` vectors are passed directly via `Int8Vectors`:

```cpp
std::vector<int8_t> data(num_vectors * dim);  // populate with int8 elements
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids)
    ->Int8Vectors(data.data())->Owner(false);
```

Build config (note `dtype: "int8"`):

```json
{
    "dtype": "int8",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pq",
        "max_degree": 26,
        "ef_construction": 100,
        "alpha": 1.2
    }
}
```

Queries use the same `Int8Vectors` setter and the same `dtype`. A runnable example is
[`316_index_int8_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/316_index_int8_hgraph.cpp).

### `float16` / `bfloat16` input

FP16 and BF16 vectors are both passed through `Float16Vectors`, which takes a `const uint16_t*`
that points at the 16-bit storage of each element. Conversion from `float` is up to the caller;
inside the VSAG source tree there are convenience helpers (`vsag::generic::FloatToFP16` in
[`src/simd/fp16_simd.h`](https://github.com/antgroup/vsag/blob/main/src/simd/fp16_simd.h)
and `vsag::generic::FloatToBF16` in
[`src/simd/bf16_simd.h`](https://github.com/antgroup/vsag/blob/main/src/simd/bf16_simd.h)),
but these are **internal headers** that are not installed under `include/vsag/`. Application code
linking against an installed VSAG library should provide its own conversion (for example, copy
the small helper, use `_cvtss_sh` / F16C intrinsics, or any FP16 library of choice). The snippet
below uses the in-tree helper for brevity:

```cpp
// The fp16/bf16 helpers below live in src/simd/ and are not part of the public
// installed headers. Replace with your own float -> uint16_t conversion when
// linking against an installed VSAG.
#include "simd/fp16_simd.h"  // FloatToFP16 (for BF16, use simd/bf16_simd.h / FloatToBF16)

std::vector<uint16_t> data(num_vectors * dim);
for (size_t i = 0; i < data.size(); ++i) {
    data[i] = vsag::generic::FloatToFP16(some_float_source());
}
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids)
    ->Float16Vectors(data.data())->Owner(false);
```

Build config:

```json
{
    "dtype": "float16",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pq",
        "max_degree": 26,
        "ef_construction": 100,
        "alpha": 1.2
    }
}
```

To switch the example to BF16, change `dtype` to `"bfloat16"` and replace `FloatToFP16` with
`FloatToBF16`; the `Float16Vectors` setter and the rest of the build/search flow stay the same.
A runnable FP16 example is
[`321_index_fp16_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/321_index_fp16_hgraph.cpp).

> **Note.** The header comment at the top of `321_index_fp16_hgraph.cpp` currently mentions a
> `BFloat16Vectors()` setter, but no such setter exists — `Float16Vectors` is the single entry
> point for both FP16 and BF16. Use it for both `dtype: "float16"` and `dtype: "bfloat16"`.

### Choosing an input type

- Pick `float32` when accuracy matters most and memory budget allows; this is the default.
- Pick `float16` / `bfloat16` to halve the input storage. FP16 has a smaller exponent range; BF16
  has fewer mantissa bits but the same exponent range as FP32, which is often preferable for
  embedding-style vectors.
- Pick `int8` when your data is already integer-quantised (e.g. produced by an upstream quantiser
  or by a model with int8 outputs). With `int8` input you typically still combine a coarse
  quantizer such as `pq` / `sq8` for the in-index storage.

The chosen `dtype` only constrains the **input** representation. The on-disk / in-memory storage is
still controlled by `base_quantization_type` (and optionally `precise_quantization_type` when
`use_reorder: true`), so e.g. `dtype: "float16"` + `base_quantization_type: "sq8"` is valid.

## Search parameters

Search-time parameters live under the `hgraph` sub-object:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ef_search` | int | Size of the search frontier. Larger = higher recall, slower query. |
| `enable_reorder` | bool | `true` by default. Set to `false` to skip the final reorder stage for this request even when the index was built with reorder enabled. This also disables the RaBitQ one-bit reorder path. |

```cpp
auto result = index->KnnSearch(
    query, topk, R"({"hgraph": {"ef_search": 200}})").value();
```

```cpp
auto fast_result = index->KnnSearch(
    query, topk,
    R"({"hgraph": {"ef_search": 200, "enable_reorder": false}})").value();
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
