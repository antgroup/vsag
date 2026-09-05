# SINDI_V2

SINDI_V2 is VSAG's term-first sparse-vector index. It uses the same windowed
inverted-list scoring model and `1 - inner_product` distance as
[SINDI](sindi.md), but persists postings by term so a search can read only the
lists needed by the query.

- Factory entry: `sindi_v2`
- Source: `src/algorithm/sindi_v2/`
- Required `dtype`: `sparse`
- Required `metric_type`: `ip`

## Storage layout

Each term payload stores only non-empty windows:

```text
non_empty_window_count
ordered {window_id, posting_count}[]
local_doc_ids[]
alignment padding
encoded_values[]
```

At query time, SINDI_V2 expands the sparse metadata into `window_count + 1`
window offsets. A query loads postings only for its retained terms through
`reader_io`, `mmap_io`, `buffer_io`, `async_io`, or an in-memory backend.

This differs from SINDI's window-first serialization. The two factory entries
therefore require different persisted formats and must not be used
interchangeably when loading an index.

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
        "doc_prune_ratio": 0.1,
        "use_quantization": true,
        "use_reorder": true,
        "remap_term_ids": true,
        "term_io": {
            "type": "buffer_io",
            "file_path": "/tmp/sindi_v2.terms"
        },
        "rerank_io": {
            "type": "block_memory_io"
        }
    }
})";
auto index = vsag::Factory::CreateIndex("sindi_v2", params).value();
index->Build(base);

auto result = index->KnnSearch(
    query, 10,
    R"({"sindi_v2": {
        "n_candidate": 100,
        "query_prune_ratio": 0.1,
        "term_prune_ratio": 0.2,
        "term_retain_threshold": 10000
    }})").value();
```

## Build parameters

Build parameters live under `index_param`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | required | Maximum number of non-zero elements per sparse vector, not the vocabulary size. |
| `term_id_limit` | int | `1000000` | Upper bound on term IDs; maximum 50 000 000. |
| `window_size` | int | `50000` | Documents per window, from 10 000 to 60 000. |
| `doc_prune_ratio` | float | `0.0` | Fraction of lowest-weight document terms removed during build (`[0.0, 1.0)`). |
| `use_quantization` | bool or string | `false` | `false` stores FP32, `true` stores SQ8, and `"fp16"` stores FP16 values. |
| `use_reorder` | bool | `false` | Store high-precision vectors and rerank coarse candidates. |
| `rerank_type` | string | `"fp32"` | Rerank storage type: `fp32` or `dmq8`. |
| `dmq_shared_codebook_threshold` | int | `1024` | Low-frequency term threshold for the shared DMQ codebook. |
| `remap_term_ids` | bool | `false` | Compact sparse or widely separated external term IDs. |
| `avg_doc_term_length` | int | `100` | Memory-estimation hint only. |
| `immutable` | bool | `false` | Select the compact read-only in-memory term DataCell. |
| `term_io` | object | `{"type": "reader_io"}` | Backend used for term-first postings after load. |
| `rerank_io` | object | `{"type": "block_memory_io"}` | Backend used for rerank vectors. |
| `rerank_layout` | uint32 | `0` | Number of leading terms in the top-terms-signature layout; `0` disables it. |

File-backed `term_io` supports `mmap_io`, `buffer_io`, and `async_io`, and
requires `file_path`. If a file-backed `rerank_io` omits `file_path`, VSAG
derives it as `<term_io.file_path>.rerank`.

`rerank_layout > 0` requires `use_reorder: true`. `rerank_type: "dmq8"`
requires `rerank_layout: 0` and the default `block_memory_io` rerank backend.

### Host filtering

SINDI_V2 supports the same `uint32_t` `host_id` build and KNN-query metadata contract as
[SINDI](sindi.md#host-filtering). Both mutable and immutable indexes are supported. It uses the
shared host grouping and routing component while retaining its own term-first posting storage.
Host membership is enforced during posting-window search, including multiple ranges created by
mutable `Add()`. Missing host ID `0`, mutable `Add()` metadata rules, streaming serialization, and
the KNN-only scope are the same as SINDI.

### Date-bucket filtering

SINDI_V2 also supports the [SINDI date-bucket, inclusive range, and hierarchical matching
contract](sindi.md#date-bucket-filtering). The two indexes share one metadata filtering
component for date-only, host-only, and combined date-and-host KNN routing. SINDI_V2 retains its
term-first posting layout and legacy serialization format. Date filtering supports mutable and
immutable indexes with either `use_reorder` setting and is preserved by both legacy and streaming
serialization; range search remains unfiltered. A mutable date-aware SINDI_V2 is build-once and
follows the same initial `Build()` or first-`Add()` and later-`Add()` rejection rules as SINDI.
Exact bucket filtering disables term-level posting pruning in each selected window, so date queries
may scan more postings than host-only queries. Host-only queries on date-enabled indexes share the
boundary-window behavior documented for SINDI.

## Search parameters

Search parameters live under `{"sindi_v2": {...}}`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_candidate` | int | `0` | Coarse candidate count; `0` uses the index default derived from `topk`. |
| `query_prune_ratio` | float | `0.0` | Fraction of the lowest-weight query terms skipped (`[0.0, 1.0)`). |
| `term_prune_ratio` | float | `0.0` | Fraction of the lowest stored values skipped in each term list (`[0.0, 1.0)`). |
| `term_retain_threshold` | uint64 | `0` | Maximum postings for one term across all windows. `0` disables the limit; positive values allow each non-empty window posting list to scan at most `max(1, floor(threshold / window_count))` postings. |

After combining the ratio and threshold limits, SINDI V2 scans at least one posting from every
non-empty term list.

SINDI V2 chooses the heap-insertion strategy automatically from the build-time
`doc_prune_ratio` and search-time `query_prune_ratio`. With the current `0.1`
threshold, it uses distance-array insertion when both ratios are `<= 0.1`; if
either ratio is greater than `0.1`, it uses term-list heap insertion.

## Choosing between SINDI and SINDI_V2

Use SINDI when the full sparse index remains in memory and window-first
serialization is appropriate. Use SINDI_V2 when persisted postings should be
organized by term or when searches should fetch only the query's term lists
from external storage.
