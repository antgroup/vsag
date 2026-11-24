# WARP

WARP is VSAG's **multi-vector exact-search** index. Each document is stored as a *set* of
token-level vectors instead of a single embedding, which is the representation used by
late-interaction models such as ColBERT; relevance is scored with **MaxSim** (sum over query
tokens of the best per-token similarity).

WARP has no separate algorithm implementation: it is [BruteForce](brute_force.md) running in
multi-vector mode. `Factory::CreateIndex("warp", ...)` builds `IndexImpl<BruteForce>` and sets a
hidden internal marker (`_warp_mode`) that switches the base datacell to the multi-vector one.
Because the scan is exhaustive, results are exact for the configured quantizer.

- Source: `src/algorithm/bruteforce/bruteforce.{h,cpp}`, `src/datacell/multi_vector_datacell.{h,inl}`, `src/quantization/multi_vector_computer.cpp`
- Example: [`examples/cpp/110_index_warp.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/110_index_warp.cpp)

## How it works

1. **Build.** Documents are appended to a flat multi-vector store. Each record holds a
   `uint32` token count followed by that many encoded token vectors. No graph, no clustering;
   quantizers that need training (`fp16`, `sq8_uniform`, `int8`) run one training pass over all
   tokens when the first batch is inserted, while `fp32`/`bf16` skip training.
2. **Add.** New documents are appended; there is no rebalancing. Duplicate labels are reported
   back in `failed_ids`.
3. **Search.** Every stored document is decoded and scored against the query multi-vector:
   for each query token the closest document token is selected and the per-token distances are
   summed. The top-k heap is then filled exactly, optionally split across threads by the
   `parallelism` search parameter.

MaxSim needs a *distance*, so the per-token term is `1 - <q, d>` for `ip` and `||q - d||^2` for
`l2`. It follows that larger dot products are closer under `ip`, while the returned values are
distances (smaller is better) for both metrics.

## Quick start

```cpp
#include <vsag/vsag.h>

std::string build_params = R"({
    "dtype": "float32",
    "metric_type": "ip",
    "dim": 128
})";

auto index = vsag::Factory::CreateIndex("warp", build_params).value();

// Every document is a set of token vectors with the same `dim`.
std::vector<vsag::MultiVector> multi_vectors(num_docs);
for (int64_t i = 0; i < num_docs; ++i) {
    multi_vectors[i].len_ = token_count_of_doc_i;          // number of tokens
    multi_vectors[i].vectors_ = token_vectors_of_doc_i;    // len_ * dim floats
}

auto base = vsag::Dataset::Make();
base->NumElements(num_docs)
    ->Dim(dim)
    ->Ids(ids)
    ->MultiVectors(multi_vectors.data())
    ->MultiVectorDim(dim)
    ->Owner(true);
index->Build(base);

auto query = vsag::Dataset::Make();
query->NumElements(1)
    ->Dim(dim)
    ->MultiVectors(&query_multi_vector)   // one vsag::MultiVector
    ->MultiVectorDim(dim)
    ->Owner(false);

auto result = index->KnnSearch(query, topk, R"({"parallelism": 4})").value();
```

The search JSON is free-form, but only the `brute_force` and `hgraph` sub-object names are
recognised; because WARP reuses the BruteForce search parser, `{"warp": {...}}` is treated as a
root object and any keys nested inside `warp` are ignored. Put search keys at the root
(`{"parallelism": 4}`) to have them take effect.

## Build parameters

The top-level fields are `dim`, `dtype`, and `metric_type`:

| Field | Values | Notes |
|-------|--------|-------|
| `dim` | positive integer | Token dimensionality; required |
| `dtype` | `float32` | WARP rejects `int8` (`WARP not support int8 datatype`); other values are not exercised |
| `metric_type` | `ip` / `l2` | `cosine` is rejected at query time by `MultiVectorComputer` |

`index_param` accepts exactly the following keys; any other key raises
`invalid config param: <key>`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_quantization_type` | string | `"fp32"` | Quantizer for the stored token vectors: `fp32`, `fp16`, `bf16`, `sq8_uniform`, `int8`. `fp16`, `sq8_uniform`, and `int8` require an explicit training pass. |
| `base_io_type` | string | `"block_memory_io"` | IO backend for the token store: `block_memory_io`, `memory_io`, `buffer_io`, `async_io`, `uring_io`, `mmap_io`; `reader_io` is used when loading a serialized index. |
| `base_file_path` | string | `"./default_file_path"` | File path for disk-backed `base_io_type` values. Two indexes sharing the same path and writing concurrently will corrupt the store, so give each build a distinct path. |
| `resize_increase_count_bit` | int | `10` | `log2` of the capacity increment used when the store grows; must be in `[1, 31]`. |

WARP is deliberately narrower than [BruteForce](brute_force.md). The following BruteForce build
keys are **not** accepted by WARP and raise `invalid config param`: `precise_quantization_type`,
`precise_io_type`, `precise_file_path`, `base_pq_dim`, `thread_count`, `store_raw_vector`,
`use_attribute_filter`, `use_residual`. WARP builds a single multi-vector base store with no
precise codes and no extra-info store.

## Search parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallelism` | int | `1` | Splits the single-query scan across this many threads. Values `<= 0` are treated as `1`. |
| `use_extra_info_filter` | bool | `false` | Applies an extra-info predicate to candidate ids during the scan. Only meaningful when the index was created with a top-level `extra_info_size`. |
| `threshold` | float | — (no filter) | Root-level key, read by `ParseSearchThreshold`. Results whose distance is above the threshold are dropped from the returned top-k. Must be a finite number. |

`timeout_ms`, `factor`, and `enable_reorder` are accepted by the shared search parser but are
**not used** by WARP: the scan is not time-bounded, `factor` does not change the candidate set
of an exact search, and there is no reorder stage.

## Operational notes

- **Removal.** Only `MARK_REMOVE` is supported; `FORCE_REMOVE` returns an error
  (`multi-vector mode only supports MARK_REMOVE`). Marking an unknown or already-removed id is a
  no-op.
- **No attribute filtering.** `use_attribute_filter` is always false for WARP, so
  `SearchWithRequest` with an attribute expression fails with
  `attribute filter is not available`. Use id-based filters instead (see
  [Filtered Search](../advanced/filtered_search.md)).
- **No vector update.** `UpdateVector` requires a dense FP32 vector and is rejected for
  multi-vector indexes; rebuild or delete-and-add instead.
- **Memory estimate.** `EstimateMemory` assumes an average of 10 tokens per document and uses
  `dim * sizeof(float)` per token, so it is only an approximation for real multi-vector corpora.

## When to use WARP

Use WARP when documents are token sets and MaxSim quality matters more than latency: it gives
exact late-interaction scores on modest corpora. For graph-accelerated multi-vector retrieval
with a two-stage coarse/rerank pipeline, use [SIMQ](simq.md) instead.

## See also

- [SIMQ](simq.md) — approximate multi-vector retrieval with MaxSim reranking
- [BruteForce](brute_force.md) — the single-vector exact baseline WARP is built on
- [Filtered Search](../advanced/filtered_search.md) — id and function filters
- [Index Parameters](../resources/index_parameters.md) — cross-index parameter overview