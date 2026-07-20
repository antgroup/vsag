# SIMQ

SIMQ is VSAG's index for **multi-vector** retrieval — the kind of data where each
document is a *set* of token-level vectors rather than a single embedding. This
pattern arises in late-interaction models such as ColBERT, where a document is
represented by one vector per token and relevance is computed via **MaxSim**
(sum of maximum per-query-token similarities).


- Source: `src/algorithm/simq/`

## How it works

1. **Dynamic clustering of token vectors.** At build time, all token vectors
   across every document are extracted into a flat pool and clustered using an
   HGraph-based dynamic clustering algorithm. The initial cluster centers are
   sampled at a ratio controlled by `init_cluster_ratio`; clusters that grow
   beyond `max_cluster_size` are split incrementally.
2. **Representative graph for coarse search.** A representative HGraph is built
   over the cluster centroids. At query time, each query token searches this
   graph to find its nearest clusters (controlled by `coarse_k`). The cluster
   scores are accumulated across all query tokens to produce a candidate set.
3. **Exact MaxSim reranking.** The top `rerank_k` candidates are re-scored by
   reading back the original token vectors from disk (or memory) and computing
   the exact MaxSim similarity between query tokens and document tokens.

The combination of cluster-level coarse search and exact reranking gives SIMQ a
tunable recall/latency tradeoff for multi-vector workloads.

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

SIMQ-specific build parameters live under `index_param`. The common fields
`dim`, `dtype`, and `metric_type` are top-level. `dtype` **must** be
`"float32"` and `metric_type` **must** be `"ip"`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | — (required) | Dimension of each token vector. |
| `base_io_type` | string | `"async_io"` | Storage backend for reranking multi-vector data. |
| `base_file_path` | string | `"./default_file_path"` | File path for disk-backed IO types. |
| `init_cluster_ratio` | float | `0.2` | Fraction of tokens sampled as initial cluster centers. |
| `max_cluster_size` | int | `64` | Maximum token vectors per cluster before split. |
| `split_start_idx` | int | `32` | Split position within an overflowing cluster. |
| `random_seed` | int | `42` | Random seed for clustering shuffle. |
| `coarse_k` | int | `8` | Default nearest clusters per query token at build time. |
| `rerank_k` | int | `100` | Default max rerank candidates at build time. |

- **`dim`** — shared across all documents and queries.
- **`base_io_type`** — supported values: `async_io`, `memory_io`,
  `block_memory_io`, `buffer_io`, `mmap_io`, `reader_io`.
- **`base_file_path`** — the default is a placeholder; provide a real path
  when using a disk-backed type (`async_io`, `buffer_io`, `mmap_io`).
- **`init_cluster_ratio`** — range `(0, 1]`. Smaller values yield fewer,
  larger clusters; larger values produce more, finer-grained clusters.
- **`max_cluster_size`** — must be > 1.
- **`split_start_idx`** — typically half of `max_cluster_size`.
  Must be in `(1, max_cluster_size)`.
- **`coarse_k`**, **`rerank_k`** — must be > 0.

> **Choosing cluster parameters.** `init_cluster_ratio` and `max_cluster_size`
> together control the number and size of clusters. A smaller
> `init_cluster_ratio` with a larger `max_cluster_size` yields fewer clusters and
> faster coarse search at the cost of recall. Start with
> `init_cluster_ratio = 0.1`–`0.2` and `max_cluster_size = 2 ×`
> `split_start_idx`, then tune with the search parameters.

## Search parameters

Search-time parameters live under the `simq` sub-object:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coarse_k` | int | *(index default)* | Nearest clusters per query token. |
| `rerank_k` | int | *(index default)* | Max rerank candidates. |

- **`coarse_k`** — overrides the build-time value. Larger values increase
  the candidate pool and improve recall at the cost of latency.
- **`rerank_k`** — overrides the build-time value. Larger values improve
  recall at the cost of more disk reads and compute.
- When omitted, the build-time defaults are used. Both values must be > 0
  when explicitly set.

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"simq": {"coarse_k": 600, "rerank_k": 5000}})").value();
```

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
  exact scoring. In practice, `rerank_k` has a larger impact on recall but also
  on latency because each additional candidate requires a disk read and full
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
