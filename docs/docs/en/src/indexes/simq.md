# SIMQ

SIMQ (Single Index for Multi-vector Query) is VSAG's index for **multi-vector**
retrieval — the kind of data where each document is a *set* of token-level vectors
rather than a single embedding. This pattern arises in late-interaction models such
as ColBERT, where a document is represented by one vector per token and relevance
is computed via **MaxSim** (sum of maximum per-query-token similarities).


- Source: `src/algorithm/simq/`

## How it works

1. **Graph-driven token partitioning.** Tokens are encoded in a temporary pool. Sampled real tokens initialize an HGraph; remaining tokens route through the graph, and oversized partitions split dynamically. This is not iterative k-means.
2. **Representative graph routing.** Each partition selects a real member token as its representative. A quantized HGraph routes each query token to candidate partitions, whose document lists form a coarse-ranked candidate set.
3. **MaxSim reranking.** The top `rerank_k` candidate documents are read from memory or disk and decoded. MaxSim is evaluated over these stored representations. With lossy quantization this is not exact scoring against the original FP32 input.

Each document enters the coarse candidate set once per query, including when its accumulated coarse score is zero or contributions from query tokens cancel out.

The two-stage pipeline trades candidate recall, quantization error, memory, and latency.

## Quick start

```cpp
#include <vsag/vsag.h>

std::string build_params = R"({
    "dtype": "float32",
    "metric_type": "ip",
    "dim": 256,
    "index_param": {
        "base_io_type": "async_io",
        "base_file_path": "/path/to/simq_base_codes.bin",
        "init_cluster_ratio": 0.1,
        "max_cluster_size": 160,
        "split_start_idx": 80,
        "random_seed": 42,
        "coarse_k": 50,
        "rerank_k": 1000
    }
})";
auto index = vsag::Factory::CreateIndex("simq", build_params).value();

// Build a dataset of MultiVector.
// Each document has a variable number of token vectors, each of dimension `dim`.
std::vector<vsag::MultiVector> base_mvs(num_docs);
std::vector<int64_t> ids(num_docs);
for (int64_t i = 0; i < num_docs; ++i) {
    base_mvs[i].len_ = doc_token_counts[i];             // number of tokens in doc i
    base_mvs[i].vectors_ = doc_token_vectors[i];        // flat array: len_ * dim floats
    ids[i] = i;
}
auto base = vsag::Dataset::Make();
base->NumElements(num_docs)
    ->Dim(dim)
    ->Ids(ids.data())
    ->MultiVectors(base_mvs.data())
    ->MultiVectorDim(dim)
    ->Owner(false);
index->Build(base);

// Search with a multi-vector query.
vsag::MultiVector query_mv;
query_mv.len_ = query_token_count;
query_mv.vectors_ = query_token_vectors;
auto query = vsag::Dataset::Make();
query->NumElements(1)
    ->Dim(dim)
    ->MultiVectors(&query_mv)
    ->MultiVectorDim(dim)
    ->Owner(false);

std::string search_params = R"({
    "simq": {
        "coarse_k": 600,
        "rerank_k": 5000
    }
})";
auto result = index->KnnSearch(query, /*topk=*/100, search_params).value();

// Read results.
const int64_t* result_ids = result->GetIds();
const float* result_dists = result->GetDistances();
int64_t result_count = result->GetDim();
for (int64_t i = 0; i < result_count; ++i) {
    int64_t id = result_ids[i];
    float dist = result_dists[i];
}
```

## Build parameters

SIMQ-specific build parameters live under `index_param`; common fields stay at the top level.
Input is FP32 `MultiVector` data regardless of the internal storage quantization.

### Top-level common fields

| Parameter | Type | Default | Constraints and effect |
|-----------|------|---------|------------------------|
| `dim` | int | required | Positive dimension of every document/query token. |
| `dtype` | string | required | Must be `"float32"`; this is the input scalar type, not the storage quantizer. |
| `metric_type` | string | required | Must be `"ip"`. |
| `repr` | string | `"dense"` (common-parser default) | Optional common layout field; use `"multi_vector"` to describe this data explicitly. SIMQ's Build/Add/search APIs still require `MultiVectors`, not single dense vectors. |
| `extra_info_size` | int | `0` | Shared parser accepts an integer without a non-negative check; positive values initialize inherited extra-info storage, but SIMQ Build/Add and search do not populate or return it. Leave at `0` for this workflow. |
| `use_old_serial_format` | bool | `false` | Common compatibility flag; SIMQ does not consume it to select its serialization format. |

### Fields under `index_param`

These are all 15 public SIMQ build fields. Other keys under `index_param` are rejected.

| Parameter | Type | Default | Constraints and effect |
|-----------|------|---------|------------------------|
| `base_io_type` | string | `"async_io"` | Document-token code backend: `async_io`, `uring_io`, `buffer_io`, `mmap_io`, `memory_io`, `block_memory_io`, or `reader_io`; see IO caveats below. |
| `base_file_path` | string | `"./default_file_path"` | Provide a usable file path for disk-backed storage (`async_io`, `uring_io`, `buffer_io`, `mmap_io`); the default is a placeholder. |
| `base_enable_read_cache` | bool | `false` | Enable the optional LRU page cache for `async_io`, `uring_io`, `buffer_io`, and `reader_io`. Ignored by `memory_io`, `block_memory_io`, and `mmap_io` (`NoCache`). |
| `base_cache_total_size` | unsigned int | `268435456` (256 MiB) | Non-negative byte budget for that cache. Capacity is rounded down to 128 KiB pages; less than one page (including `0`) disables caching even when enabled. Applies to the document-code IO instance, not the representative graph. |
| `quantization_type` | string | `"fp32"` | Factory-supported document/graph storage types: `fp32`, `fp16`, `bf16`, `sq8_uniform`, `int8`. See the `int8` caveat below; this does not change FP32 input requirements. |
| `build_thread_count` | unsigned int | `1` | Shared parser accepts non-negative integers without a SIMQ-specific positive check; use `>= 1`. Controls construction/representative HGraphs and clustering work, not a per-query parallelism override. Clustering uses at least one worker. |
| `init_cluster_ratio` | float | `0.2` | In `(0, 1]`; fraction of tokens selected as initial seeds. Smaller values tend to produce fewer, larger partitions. |
| `max_cluster_size` | int | `64` | Must be `> 1`; token-count threshold for splitting a partition, not a document-count limit. |
| `split_start_idx` | int | `32` | Must be in `(1, max_cluster_size)`; start of the new partition in the distance-sorted overflowing partition. |
| `random_seed` | int | `42` | Seed for initial token shuffle and bounded document-model training sample; no additional SIMQ range check. |
| `coarse_k` | int | `8` | Must be `> 0`; default number of nearest representative clusters per query token, capped by available clusters at search time. |
| `rerank_k` | int | `100` | Must be `> 0`; default maximum candidate documents for MaxSim reranking, capped by document count at search time. |
| `split_delay_seconds` | float | `0.0` | Must be `>= 0`; delay after first overflow before incremental `Add` splits become eligible. `0` is immediate. Pending splits are checked during `Add`, not by a background timer; this does not delay initial Build clustering. |
| `support_duplicate` | bool | `false` | Let identical representative vectors share a representative-HGraph node. Does not deduplicate document IDs or suppress duplicate tokens in the construction-internal HGraph. |
| `deduplicate_storage` | bool | `false` | Requires `support_duplicate: true`; share duplicate vector-code storage slots in the representative HGraph, not document-token storage. |

**IO caveats.** Native `uring_io` requires liburing; without it, the backend uses a buffered fallback.
`reader_io` is recognized by the factory but is an external-reader, read-only backend requiring a
non-null Reader binding. Selecting it in JSON alone does not provide writable Build/Add storage.
Do not treat every recognized IO type as interchangeable for construction.

**Quantization caveat.** `int8` is factory-supported, but its current encoder copies raw input bytes
rather than performing a numeric FP32-to-int8 conversion. It must not be advertised as ordinary
scalar compression of FP32 token values. Prefer `fp32`, `fp16`, `bf16`, or `sq8_uniform` for the
FP32 workflow, and validate storage accuracy on your data.

> **Choosing cluster parameters.** `init_cluster_ratio` and `max_cluster_size`
> together control the number and size of clusters. A smaller
> `init_cluster_ratio` with a larger `max_cluster_size` yields fewer clusters and
> faster coarse search at the cost of recall. Start with
> `init_cluster_ratio = 0.1`–`0.2` and `max_cluster_size = 2 ×`
> `split_start_idx`, then tune with the search parameters.

## Search parameters

The following fields under the `simq` sub-object apply to both `KnnSearch` and `RangeSearch`:

| Parameter | Type | Default | Constraints and effect |
|-----------|------|---------|------------------------|
| `coarse_k` | int | `-1` → build value | Positive values override the build value; omitted or `<= 0` falls back to it. More clusters per query token widen the candidate pool, usually at higher latency. |
| `rerank_k` | int | `-1` → build value | Positive values override the build value; omitted or `<= 0` falls back to it. More candidates increase MaxSim IO/compute; this is independent of requested `topk`/range output size. |
| `timeout_ms` | int | omitted (no deadline) | Soft elapsed-time budget in milliseconds. Presence enables timing; the parser has no positivity check, so `<= 0` is an immediately expired budget, not “disabled”. Can return partial or empty results. |

The timeout starts before coarse search but does not interrupt representative-graph searches or
in-flight IO/MaxSim work. It is checked after coarse search and between rerank work units
(up to 32 documents per KNN batch; one document for range search), and again before returning.
A coarse-phase timeout clears the coarse candidates. This is not a hard wall-clock latency bound;
check the returned statistics' `is_timeout` flag.

The inherited shared parser also accepts these fields inside `simq`, but SIMQ consumes none of
them in either search path:

| Parameter | Type | Parsed default | SIMQ effect |
|-----------|------|----------------|-------------|
| `parallelism` | int | `1` | Ignored; the shared parser normalizes non-positive values to `1`. |
| `factor` | float | `0.0` | Ignored; no SIMQ range check. Tune `rerank_k` instead. |
| `enable_reorder` | bool | `true` | Ignored, including `false`; SIMQ always performs MaxSim reranking of retained candidates. |

| Root-level parameter | Type | Default | Constraints and effect |
|----------------------|------|---------|------------------------|
| `threshold` | float | omitted | KNN only: finite inclusive upper bound on final distance; see below. |

For **KNN only**, optional root-level `threshold` (float, absent by default) must be a finite JSON
number. It is an inclusive upper bound on the **final returned distance**, not a coarse-search
cutoff or minimum similarity. Only finite distances `<= threshold` pass; negative thresholds are
allowed and fewer than `topk` results may be returned. SIMQ's distance sums the per-query-token
minimum `1 - inner_product` over stored document tokens, rather than using a single-vector IP
distance. See [KNN threshold filtering](../guide/knn_search.md).
`RangeSearch` uses its `radius` argument instead and does not consume root-level `threshold`;
`limited_size` limits range output, not the candidate pool controlled by `rerank_k`.

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"threshold": 4.0, "simq": {"coarse_k": 600, "rerank_k": 5000, "timeout_ms": 100}})").value();
```

## Search statistics

`KnnSearch` and `RangeSearch` attach a JSON statistics object to the returned dataset. Read the
complete object or request individual values:

```cpp
std::cout << result->GetStatistics() << '\n';
auto values = result->GetStatistics(
    {"simq_coarse_candidate_count", "simq_rerank_candidate_count"});
```

| Key | Meaning |
|-----|---------|
| `dist_cmp` | Exact MaxSim distance comparisons during reranking |
| `simq_coarse_dist_cmp` | Distance comparisons performed by representative-graph searches |
| `simq_coarse_probe_count` | Representative clusters probed across all query tokens |
| `simq_coarse_candidate_count` | Unique document candidates produced by coarse search before applying `rerank_k` |
| `simq_rerank_candidate_count` | Candidates retained after applying `rerank_k` |
| `simq_filtered_candidate_count` | Rerank candidates rejected by the supplied filter |
| `simq_result_count` | Results returned after reranking and range/top-k limits |
| `simq_limited_size_applied` | Whether `RangeSearch` truncated matches to `limited_size` |

These counters make it possible to distinguish a narrow coarse candidate pool from filtering,
reranking, and output-limit effects when tuning recall and latency.

## When to use SIMQ

- **Late-interaction retrieval** with ColBERT or similar models where each
  document is a bag of token-level vectors and relevance is computed via MaxSim.
- **Multi-vector relevance** where a single embedding per document loses too much
  information and fine-grained token-level matching is needed.
- **Large-scale multi-vector corpora** where brute-force MaxSim is too slow and a
  two-stage coarse-then-rerank pipeline provides the right recall/latency
  tradeoff.

SIMQ only accepts `float32` multi-vector data with inner-product similarity. It
does **not** accept single dense vectors or sparse vectors (use HGraph or SINDI
for those).

## Practical guidance

- **Scaling `coarse_k` and `rerank_k`.** Increasing `coarse_k` widens the
  cluster-level candidate net; increasing `rerank_k` admits more documents to
  MaxSim scoring over stored representations. In practice, `rerank_k` has a larger impact on recall but also
  on latency because each additional candidate requires storage access and full
  MaxSim computation.
- **IO type selection.** Use `async_io` for large corpora that do not fit in
  memory. Use `memory_io` or `block_memory_io` when the multi-vector data fits in
  RAM for the lowest reranking latency.
- **Cluster sizing.** Set `max_cluster_size` to roughly twice `split_start_idx`.
  The split point determines how the token vectors are partitioned when a cluster
  overflows; centering it keeps the two halves balanced.

## MultiVector field reference

| Field | Type | Description |
|-------|------|-------------|
| `len_` | `uint32_t` | Number of token vectors in this document or query. |
| `vectors_` | `float*` | Contiguous array of `len_ * dim` floats |

## See also

- [Creating an Index](../guide/create_index.md)
- [Index Parameters](../resources/index_parameters.md)
- [k-Nearest Neighbor Search](../guide/knn_search.md)

## Quantized construction and incremental ingestion

The Dataset input remains FP32 `MultiVector`, including when `quantization_type` is `fp16`, `bf16`, or `sq8_uniform`. SIMQ encodes tokens before partition construction and uses the selected quantizer's native code-to-code distance for partition maintenance. Both construction and representative HGraphs use quantized base storage without an FP32 reorder copy. Temporary FP32 decoding remains necessary for graph insertion/query APIs and mean accumulation; this is not zero-decode construction.

Construction avoids a full FP32 flattened token copy. It uses a temporary encoded token pool, token-address metadata, and at most 4096 FP32 tokens per training/staging buffer. The encoded pool duplicates document codes until Build returns. Caller-owned FP32 input and O(tokens) metadata still contribute to peak memory. Build passes a deterministic sample of at most 4096 tokens to the document quantizer’s Train method (sampling with replacement for larger inputs); whether training learns data-dependent state is quantizer-specific. Subsequent Add calls reuse the document model without retraining. HGraphs own separate models and never exchange encoded bytes across models.

Use `Build(first_batch)` followed by `Add(next_batch)` for incremental ingestion. The first batch should represent subsequent data. A quantized or incrementally built index need not produce the same partitions or recall as an all-at-once FP32 build; benchmark your workload. FP32/BF16/FP16/SQ8 input and storage types must not be confused. SIMQ continues to accept only FP32 input and inner-product metric.

### Serialization compatibility

New readers support legacy, unversioned SIMQ indexes when constructed with matching dimension and document quantization. Legacy representative graphs remain FP32 to preserve existing search behavior. Load reconstructs document-model representative codes and refreshes token distances, requiring an O(tokens) migration pass with bounded vector scratch; subsequent incremental partitions are not promised to match older implementations. Resaving preserves the actual graph quantization and upgraded representative metadata. New writes use SIMQ format version 1; unknown versions and mismatched dimensions/quantizers are rejected. Reading new files with older binaries is not guaranteed.
