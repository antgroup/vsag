# SINDI

SINDI (**S**parse **IN**verted **D**ense **I**ndex) is VSAG's index for **sparse
vectors** — the kind produced by BM25, SPLADE, and other learned-sparse encoders.
Unlike the dense indexes (HGraph, IVF), SINDI operates directly on term/value
pairs and is the only VSAG index that accepts `dtype: "sparse"`.

- Source: `src/algorithm/sindi/`
- Example: [`examples/cpp/109_index_sindi.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/109_index_sindi.cpp)

## How it works

1. **Window-based inverted lists.** Documents are grouped into fixed-size windows
   (`window_size`). Within each window, an inverted list per term maps a term id
   to the `(doc_id, value)` pairs that mention it.
2. **Optional pruning and quantization.** During construction, `doc_prune_ratio`
   drops low-weight terms per document, and `use_quantization` compresses the term
   values to shrink memory further.
3. **Scoring.** At query time, SINDI iterates the non-zero terms of the query,
   walks the corresponding inverted lists in each window, aggregates contributions
   into a max-heap of size `n_candidate`, and returns the top-k. When `use_reorder`
   is enabled, the candidates are re-scored against a high-precision flat copy.

Distance is returned as `1 - inner_product` so results sort ascending as in the
dense indexes.

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
        "use_reorder": false
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
| `term_id_limit` | int | `1000000` | Upper bound on term id values (≥ max term id + 1). |
| `window_size` | int | `50000` | Documents per window (range: 10 000 – 60 000). |
| `doc_prune_ratio` | float | `0.0` | Fraction of lowest-weight terms dropped per doc at build time (0.0 – 0.9). |
| `use_quantization` | bool | `false` | Quantize stored term values to cut memory. |
| `use_reorder` | bool | `false` | Keep a high-precision flat copy and rescore results (~2× memory). |
| `avg_doc_term_length` | int | `100` | Hint for memory estimation only. |

> **`dim` vs `term_id_limit`.** For the sparse vector `{0:0.1, 2:0.5, 177:0.8}`,
> `dim` is `3` (three non-zero entries) while `term_id_limit` must be ≥ `178`
> (largest term id + 1). Sizing `term_id_limit` to your vocabulary is the most
> common first-time mistake.

## Search parameters

Search-time parameters live under the `sindi` sub-object:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_candidate` | int | `0` | Candidate heap size. When `0`, defaults to `SPARSE_AMPLIFICATION_FACTOR · topk` (500×). If set, must satisfy `1 ≤ n_candidate ≤ SPARSE_AMPLIFICATION_FACTOR · topk`. |
| `query_prune_ratio` | float | `0.0` | Fraction of lowest-weight query terms skipped (0.0 – 0.9). |
| `term_prune_ratio` | float | `0.0` | Fraction of term-list entries skipped (0.0 – 0.9). |
| `use_term_lists_heap_insert` | bool | `true` | Term-list-ordered heap insertion; usually faster. |

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"sindi": {"n_candidate": 200, "query_prune_ratio": 0.1}})").value();
```

## When to use SINDI

- Sparse retrieval with BM25, SPLADE, uniCOIL, or similar learned-sparse encoders.
- Hybrid dense+sparse pipelines where SINDI handles the sparse leg in parallel with
  HGraph / IVF for dense embeddings.
- Memory-constrained deployments of sparse corpora (`use_quantization: true`
  roughly halves memory with a small recall loss; `use_reorder: true` trades
  memory for recall).

SINDI does **not** accept dense vectors and supports only inner-product similarity.
Range search and id-based filtering are supported; see the example for usage.

## See also

- [Creating an Index](../guide/create_index.md)
- [Index Parameters](../resources/index_parameters.md)
- [k-Nearest Neighbor Search](../guide/knn_search.md)
