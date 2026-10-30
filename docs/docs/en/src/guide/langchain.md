# LangChain Integration

`pyvsag.langchain.VSAGVectorStore` implements LangChain's `VectorStore` interface
on top of `pyvsag.Index`. Dense embeddings and synchronous operations only.

## Installation

```bash
pip install "pyvsag[langchain]"
```

The extra installs `langchain-core>=0.3.77,<2` and `numpy`; `pip install pyvsag`
does not install LangChain. Importing `pyvsag.langchain` without the extra raises
`ImportError` with this install command.

The LangChain integration requires Python 3.9 or later. On Python 3.6–3.8,
installing the extra does not install `langchain-core`, and the integration is
unavailable. The core `pyvsag` package retains its existing Python version support.

| Python      | `langchain-core` 0.3.x | `langchain-core` 1.x |
|-------------|------------------------|----------------------|
| 3.9         | Yes                    | No                   |
| 3.10 – 3.14 | Yes                    | Yes                  |

## Quick Start

```python
from langchain_core.embeddings import DeterministicFakeEmbedding
from pyvsag.langchain import VSAGVectorStore

store = VSAGVectorStore.from_texts(
    ["HGraph is a graph index", "SQ8 compresses dense vectors"],
    DeterministicFakeEmbedding(size=64),  # wiring demo; fake vectors carry no semantics
    metadatas=[{"topic": "index"}, {"topic": "quantization"}],
    ids=["doc-1", "doc-2"],
)

store.similarity_search("graph index", k=1)
store.as_retriever(search_kwargs={"k": 1}).invoke("vector compression")
```

The standard `VectorStore` methods — `from_texts`, `from_documents`,
`add_texts`, `add_documents`, `get_by_ids`, `delete`, `similarity_search`,
`similarity_search_with_score`, `similarity_search_by_vector`,
`similarity_search_with_score_by_vector` and `as_retriever` — follow LangChain's
contracts.

## Configuration

```python
store = VSAGVectorStore(
    embedding,                                     # LangChain Embeddings
    index_type="hgraph",                           # hgraph, ivf, brute_force, ...
    metric_type="l2",                              # l2, ip, cosine
    index_param={"max_degree": 32},                # used to construct pyvsag.Index
    search_params={"hgraph": {"ef_search": 100}},  # passed to Index.knn_search()
)
```

- `index_param` and `search_params` accept a dict or a JSON string.
- The index is built by the first non-empty `add_texts()`; the dimension comes
  from the embeddings and must be greater than zero.
- Metadata entries must be mappings or `None`; invalid entries raise `TypeError`.
- `hgraph` and `brute_force` have default search parameters. Other index types
  require `search_params`, for example `{"ivf": {"scan_buckets_count": 16}}`;
  `lazy_hgraph` also needs a nested `index_param`.
- `ef_search` can be overridden per search on `hgraph`; the default is
  `max(k, 100)`.
- `hgraph` is built with `support_remove=true`; `delete()` and upsert rely on it.

## IDs

LangChain string IDs map to store-local `int64` IDs that are never reused. Missing
IDs are generated with `uuid4()`; duplicates within one batch raise `ValueError`.
Adding an existing ID is an upsert: the new vector is added, then the old one
removed. `delete()` ignores unknown IDs, and `delete(ids=None)` removes all
documents.

If removal fails during delete or upsert, the affected documents stay unchanged
and the operation raises `RuntimeError`. Batches may partially succeed; retry the
failed IDs. Unpublished vectors are excluded from search and retried for cleanup
on later writes. Searches expand the candidate count to skip stale vectors.

## Scores

`similarity_search_with_score()` returns the raw VSAG distance, sorted ascending:

| `metric_type` | Score                      |
|---------------|----------------------------|
| `l2`          | squared L2 distance        |
| `ip`          | `1 - dot(query, document)` |
| `cosine`      | `1 - cosine_similarity`    |

Scores are not normalized and are not comparable across metrics. See
[Metric Semantics](../resources/metric_semantics.md).

## Limitations

- Dense `float32` embeddings only; sparse vectors (SINDI) and FP16/BF16 inputs
  are not supported.
- Pyramid indexes require path metadata that the Python bindings do not provide.
- No metadata filtering, MMR, range search or native async. `filter`,
  `where_document`, `score_threshold`, `fetch_k`, `lambda_mult` and search types
  other than `similarity` raise `NotImplementedError`; async methods use
  LangChain's thread-pool fallback; other unknown keyword arguments are ignored.
- `similarity_search_with_relevance_scores()` raises `NotImplementedError`.
- Document text, metadata and ID mappings are kept in memory.
  `store.index.save(path)` persists only the VSAG index.
- Removal is tombstone based; the adapter never compacts or rebuilds.
- Searches and writes are serialized to keep vectors and document mappings consistent.

## Example

[`examples/python/111_langchain_vector_store.py`](https://github.com/antgroup/vsag/blob/main/examples/python/111_langchain_vector_store.py)
