# SINDI

![SINDI: per-term inverted lists grouped by window; only the lists matching the query's non-zero terms are walked and accumulated into an n_candidate-sized heap](../figures/indexes/sindi-overview.svg)

SINDI (**S**parse **IN**verted **D**ense **I**ndex) is VSAG's index for **sparse
vectors** — the kind produced by BM25, SPLADE, and other learned-sparse encoders.
Unlike the dense indexes (HGraph, IVF), the SINDI family operates directly on term/value
pairs and is the only VSAG index family that accepts `dtype: "sparse"`.

- Source: `src/algorithm/sindi/`
- SINDI V2 source: `src/algorithm/sindi_v2/`
- Example: [`examples/cpp/109_index_sindi.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/109_index_sindi.cpp)

## How it works

1. **Window-based inverted lists.** Documents are grouped into fixed-size windows
   (`window_size`). Within each window, an inverted list per term maps a term id
   to the `(doc_id, value)` pairs that mention it.
2. **Optional pruning and quantization.** During construction, `doc_prune_ratio`
   drops low-weight terms per document. `use_quantization: true` stores SQ8 values,
   while `use_quantization: "fp16"` stores half-precision values.
3. **Scoring.** At query time, SINDI iterates the non-zero terms of the query,
   walks the corresponding inverted lists in each window, aggregates contributions
   into a max-heap of size `n_candidate`, and returns the top-k. When `use_reorder`
   is enabled, the candidates are re-scored against a forward store. The default
   forward store keeps fp32 values, while `rerank_type: "dmq8"` uses a compressed
   DMQ store to reduce rerank memory.

Distance is returned as `1 - inner_product` so results sort ascending as in the
dense indexes.

## `sindi` and `sindi_v2`

Both entries use the same windowed inverted-list search model, build parameters,
and distance definition. Their main difference is the serialized posting layout:

| Factory entry | `immutable` | In-memory DataCell | Serialized layout |
| --- | ---: | --- | --- |
| `sindi` | `false` | mutable | mutable window-first |
| `sindi` | `true` | immutable | immutable sparse window-first |
| `sindi_v2` | `false` | mutable | term-first |
| `sindi_v2` | `true` | immutable | term-first |

- Use `sindi` when the complete index stays in memory and is saved/restored by window.
- The SINDI V2 term-first layout can load postings per query term through
  `reader_io`, `mmap_io`, `buffer_io`, or `async_io`. `term_io` changes where
  deserialized postings reside, not the file format.
- Immutable construction keeps only one mutable staging window at a time. It
  compacts and appends that window as `ImmutableSINDIWindow`, releases the
  staging memory, and then builds the next window.
- `SetImmutable()` only changes the index lifecycle to read-only; it does not
  convert a mutable DataCell into an immutable one.

Each SINDI V2 term payload stores only non-empty windows:

```text
non_empty_window_count
ordered {window_id, posting_count}[]
local_doc_ids[]
alignment padding
encoded_values[]
```

At query time the sparse metadata is expanded once into dense `window_offsets`
with `window_count + 1` entries. Window `i` uses the posting range
`[window_offsets[i], window_offsets[i + 1])`. With `remap_term_ids`, the term
dictionary contains `mapper.Size()` compact entries; without remapping, it ends
at `max_active_term`.

## Quick start

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 1024,
    "index_param": {
        "term_id_limit": 30000,
        "window_size": 50000,
        "doc_prune_ratio": 0.0,
        "use_quantization": false,
        "use_reorder": false,
        "remap_term_ids": false
    }
})";
auto index = vsag::Factory::CreateIndex("sindi", params).value();

// Build a dataset of SparseVector.
auto base = vsag::Dataset::Make();
base->NumElements(n)
    ->SparseVectors(sparse_vectors)  // vsag::SparseVector*
    ->Ids(ids)
    ->Owner(false);
index->Build(base);

// Search.
auto query = vsag::Dataset::Make();
query->NumElements(1)->SparseVectors(&query_vec)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10,
    R"({"sindi": {"n_candidate": 100}})").value();
```

## Build parameters

Build-time parameters live under `index_param`. `dtype` **must** be `"sparse"`
and `metric_type` **must** be `"ip"`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | — (required) | Maximum number of non-zero elements per sparse vector. *Not* the vocabulary size. |
| `term_id_limit` | int | `1000000` | Upper bound on term id values (≥ max term id + 1). The maximum is 50 000 000 for both `sindi` and `sindi_v2`. |
| `window_size` | int | `50000` | Documents per window (range: 10 000 – 60 000). |
| `doc_prune_ratio` | float | `0.0` | Fraction of lowest-weight terms dropped per doc at build time (0.0 – 0.9). |
| `use_quantization` | bool or string | `false` | `false` stores FP32 values, `true` stores SQ8 values, and `"fp16"` stores FP16 values. |
| `use_reorder` | bool | `false` | Keep a high-precision forward store and rescore candidates after coarse SINDI scoring (~2× memory with FP32). |
| `rerank_type` | string | `"fp32"` | Forward-store type used when `use_reorder` is enabled. `fp32` keeps exact values; `dmq8` stores compressed 8-bit DMQ codes. |
| `dmq_shared_codebook_threshold` | int | `1024` | With `rerank_type: "dmq8"`, terms occurring at most this many times share one codebook; more frequent terms keep independent codebooks. Set to `0` to disable sharing. |
| `remap_term_ids` | bool | `false` | Remap term IDs before indexing; useful when term IDs are sparse or have large gaps. |
| `avg_doc_term_length` | int | `100` | Hint for memory estimation only. |
| `immutable` | bool | `false` | Build or load the compact read-only runtime. `Build()` compacts completed windows as it proceeds to reduce peak memory; incremental `Add()` is rejected. |

`sindi_v2` additionally accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `term_io` | object | `{"type": "reader_io"}` | Backend used when deserializing term-first postings. Supports `reader_io`, in-memory IO, and file-backed `mmap_io`, `buffer_io`, or `async_io`. File-backed IO requires `file_path`. |
| `rerank_io` | object | `{"type": "block_memory_io"}` | Backend for rerank vectors when `use_reorder` is enabled. A missing file-backed `file_path` is derived as `<term_io.file_path>.rerank`. |
| `rerank_layout` | non-negative int | `0` | Number of leading terms used by the top-terms-signature rerank layout. `0` disables the layout; a positive value requires `use_reorder: true`. |

For SINDI V2, `rerank_type: "dmq8"` currently requires the default in-memory
`block_memory_io` rerank backend and `rerank_layout: 0`.

Search parameters must use a sub-object matching the factory entry: `{"sindi": {...}}`
for `sindi`, or `{"sindi_v2": {...}}` for `sindi_v2`.

> **`dim` vs `term_id_limit`.** For the sparse vector `{0:0.1, 2:0.5, 177:0.8}`,
> `dim` is `3` (three non-zero entries) while `term_id_limit` must be ≥ `178`
> (largest term id + 1). Sizing `term_id_limit` to your vocabulary is the most
> common first-time mistake.

### Sparse value formats

`use_quantization` retains its legacy boolean behavior and also accepts one string value:

| Value | Stored value format | Trade-off |
| --- | --- | --- |
| `false` | FP32 | Highest value precision and largest posting payload |
| `true` | SQ8 | Smallest posting values; quantization range is learned during the initial build |
| `"fp16"` | FP16 | Half the FP32 value bytes without SQ8 range calibration |

Indexes built with `false` or `true` retain the legacy serialization representation. Older VSAG
versions cannot parse a SINDI index that uses the new `"fp16"` value format; upgrade readers
before deploying FP16 artifacts.

### Immutable low-memory build

Set `immutable: true` when the completed SINDI index is read-only:

```json
{
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 1024,
    "index_param": {
        "term_id_limit": 30000,
        "window_size": 50000,
        "use_quantization": "fp16",
        "remap_term_ids": true,
        "immutable": true
    }
}
```

During `Build()`, each completed mutable window is compacted into an immutable payload and the
temporary window is released before construction continues. In the Sparse4M FP16 measurement
reported with this feature, peak build memory fell from 27.03 GB to 6.08 GB (77.51% lower), while
build time increased from about 330 seconds to 599 seconds. Treat these figures as workload-specific
evidence, not a capacity guarantee.

The immutable runtime supports KNN and range search plus legacy `Serialize`/`Deserialize`.
It rejects incremental `Add`, `GetSparseVectorByInnerId`, `CalcDistanceById`, and
`CalDistanceById`. Streaming serialization is not supported for immutable SINDI; use the matching
legacy serialization APIs or keep the index mutable when the streaming format is required.
The serialized index must be loaded into a SINDI created with the same `immutable` setting.

## Search parameters

Search-time parameters live under the sub-object matching the factory entry:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_candidate` | int | `0` | Candidate heap size. When `0`, defaults to `SPARSE_AMPLIFICATION_FACTOR · topk` (500×). If set, must satisfy `1 ≤ n_candidate ≤ SPARSE_AMPLIFICATION_FACTOR · topk`. |
| `filter_callback_limit` | uint64 | `0` | Maximum user `Filter::CheckValid` callback invocations for one filtered search request. A value of `0` disables the limit. Internal checks such as the deleted-ID filter do not count. Reaching a positive limit stops candidate and window scanning after processing the final callback result and returns the already filtered candidates, so the result may be partial. The limit applies to KNN and range search through both the regular APIs and `SearchWithRequest`. |
| `query_prune_ratio` | float | `0.0` | Fraction of lowest-weight query terms skipped (`[0.0, 1.0)`). |
| `term_prune_ratio` | float | `0.0` | Fraction of the lowest-value postings skipped from each term list (`[0.0, 1.0)`). |
| `term_retain_threshold` | uint64 | `0` | Maximum postings for one term across all windows. A value of `0` disables this limit; positive values allow each non-empty window posting list to scan at most `max(1, floor(threshold / window_count))` postings. |

After combining the ratio and threshold limits, SINDI scans at least one posting from every
non-empty term list.

Within each window, postings for a term are sorted by their stored value in descending
order, then by internal document id in ascending order. If both term-prune limits are
enabled, the scanned prefix is the smaller of
`floor(list_size · (1 - ratio))` and `floor(threshold / window_count)`.
SINDI chooses the heap-insertion strategy automatically from the build-time
`doc_prune_ratio` and search-time `query_prune_ratio`. With the current `0.1`
threshold, SINDI uses the distance-array insertion path when both ratios are
`<= 0.1`; if either ratio is greater than `0.1`, it uses term-list heap
insertion. The legacy
`use_term_lists_heap_insert` search parameter is ignored by `sindi`; configure pruning
ratios instead. `sindi_v2` uses the explicit boolean setting.

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"sindi": {"n_candidate": 200, "filter_callback_limit": 10000, "query_prune_ratio": 0.1}})").value();
```

## When to use SINDI

- Sparse retrieval with BM25, SPLADE, uniCOIL, or similar learned-sparse encoders.
- Hybrid dense+sparse pipelines where SINDI handles the sparse leg in parallel with
  HGraph / IVF for dense embeddings.
- Memory-constrained deployments of sparse corpora (`use_quantization: true` selects SQ8,
  while `"fp16"` halves FP32 value bytes; `use_reorder:
  true` trades forward-store memory for recall, and `rerank_type: "dmq8"` reduces
  that forward-store overhead).
- Read-only snapshots that need lower peak build memory (`immutable: true`), accepting slower
  construction and no incremental writes.

SINDI does **not** accept dense vectors and supports only inner-product similarity.
Range search and id-based filtering are supported; see the example for usage.
When `rerank_type` is `dmq8`, codebooks are fixed by the initial build, so incremental
`Add` after the model is established and `UpdateVector` are not supported.

## Practical guidance

- For Chinese corpora, we recommend encoding sparse vectors with BGE-M3. For
    English corpora, SPLADE is the more common default.
- BGE-M3 can emit both sparse and dense vectors. Today SINDI handles the sparse
    leg, and VSAG plans to support fused sparse+dense scoring in a future release.
- Sparse vectors are not a complete replacement for BM25 full-text retrieval. In
    practice, three-way recall with BM25 + sparse + dense usually outperforms any
    two-way combination.
- At the index level, SINDI can also serve BM25-style scoring: use inverse
    document frequency as the query-side term weight, and use term-frequency-based
    weights as the document-side term value.

## Common configurations

1. Flat brute-force sparse index. Keep all non-zero terms in the inverted index
     (`doc_prune_ratio: 0.0`), disable the flat reranker (`use_reorder: false`),
     and disable quantization (`use_quantization: false`). This is the simplest
     high-recall baseline.
2. Pruned high-accuracy index. Prune most low-weight terms during build
     (`doc_prune_ratio: 0.4`), keep the flat copy for reranking
     (`use_reorder: true`), and enable quantization to shrink inverted-list memory
     (`use_quantization: true`). This is a common balance between memory and
     recall.
3. Pruned high-accuracy index with compressed reranking. Use the same pruning and
     inverted-list quantization as above, but set `rerank_type: "dmq8"` together
     with `use_reorder: true` to reduce forward-store memory.
4. Very large sparse vocabularies. When term IDs are sparse within the `uint32`
     range, such as hash-based tokenizers, external vocabulary IDs, or vocabularies
     with large gaps, enable `remap_term_ids: true`. This avoids managing many
     empty posting lists and helps stay below the `term_id_limit` ceiling.
5. Read-only low-memory build. Add `immutable: true`, normally together with
     `use_quantization: "fp16"` and `remap_term_ids: true`, when peak build memory matters more
     than build speed and the finished index will not receive incremental writes.

## Mark remove

SINDI supports `RemoveMode::MARK_REMOVE`. Calling `Remove(ids)` (the default mode)
tombstones the given ids so they no longer appear in search results;
`GetNumElements()` drops accordingly and `GetNumberRemoved()` reports the running
total. Removing an id that is absent or already removed is a no-op.
`RemoveMode::FORCE_REMOVE` is not supported and returns an error.

Mark-removed documents still occupy memory until the index is rebuilt; the space
is not physically reclaimed.

## See also

- [Creating an Index](../guide/create_index.md)
- [Index Parameters](../resources/index_parameters.md)
- [k-Nearest Neighbor Search](../guide/knn_search.md)
