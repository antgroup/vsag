# Index Analysis (`AnalyzeIndexBySearch` & `analyze_index`)

VSAG ships an introspection capability for inspecting an index that has already been built or
loaded, so you can diagnose recall regressions, quantization quality, graph health and search
performance **without** rebuilding the index. This capability is exposed in two ways:

- the C++ API `Index::AnalyzeIndexBySearch` (declared in `include/vsag/index.h`);
- the command-line diagnostic tool `analyze_index`, located under `tools/analyze_index/`.

## The `AnalyzeIndexBySearch` API

```cpp
// include/vsag/index.h
virtual std::string
AnalyzeIndexBySearch(const SearchRequest& request);
```

- **Input**: a `SearchRequest` (query dataset + `topk` + search parameter JSON).
- **Output**: a JSON-formatted string containing dynamic, query-driven metrics.
- **Supported indexes**: currently `HGraph`, `IVF`, and `SINDI`. `Pyramid` only supports static
  analysis through `GetStats()` — it does not yet override `AnalyzeIndexBySearch`. Indexes that do
  not implement this API will throw an exception when called.

It is complementary to `Index::GetStats()`, which reports static structural properties of the
index without needing query data. For graph-based indexes, additional graph-health details such
as degree distribution, entry-point quality, sub-index recall and low-recall hot-spots are
exposed through `GetStats()` rather than through `AnalyzeIndexBySearch`.

### Static metrics from `GetStats()`

#### HGraph metrics

| Metric | Meaning |
| --- | --- |
| `total_count` | Total number of vectors in the index |
| `deleted_count` | Vectors marked for deletion |
| `connect_components` | Connected components in the proximity graph |
| `maximal_component_size` | Size of the largest connected component |
| `in_degree_distribution` | Histogram of graph in-degrees |
| `out_degree_distribution` | Histogram of graph out-degrees |
| `average_degree` | Average graph degree over valid nodes |
| `duplicate_ratio` | Proportion of duplicate vectors in the dataset |
| `avg_distance_base` | Average distance on sampled base vectors |
| `recall_base` | Self-recall on sampled base vectors |
| `time_cost_query` | Average latency when sampled base vectors are searched as queries |
| `proximity_recall_neighbor` | Recall of graph neighbor lists against true nearest neighbors |
| `quantization_bias_ratio` | Quantized-distance bias against exact distance |
| `quantization_inversion_count_rate` | Rate of distance-order inversions caused by quantization |

#### SINDI metrics

| Metric | Meaning |
| --- | --- |
| `total_count` | Total number of sparse vectors in the index |
| `window_count` | Number of SINDI windows |
| `active_term_count.mean` / `min` / `max` | Per-window ratio of non-empty terms to term capacity |
| `active_term_count.avg_count` | Average count of non-empty terms per window |
| `posting_length_distribution.mean` / `max` / `p95` / `p99` | Distribution of non-empty posting-list lengths |
| `posting_length_distribution.long_tail_threshold` | P99 posting-list length used as the long-tail threshold |
| `posting_length_distribution.long_tail_mean` | Ratio of posting lists longer than the P99 threshold |
| `mean_doc_retained.mean` | Average ratio of retained terms after document pruning |
| `recall_base` | Self-recall using sampled base vectors as queries and exact sparse ground truth |
| `doc_prune_recall` | Candidate recall from the doc-pruned index with query pruning disabled |
| `doc_prune_bias_mean` | Average relative distance bias between doc-pruned distance and exact sparse distance |
| `doc_prune_inversion_count_rate` | Candidate-pair order inversion rate introduced by document pruning |
| `quantization_range.min_val` / `max_val` / `diff` | SQ8 quantization range, emitted only when quantization is enabled |
| `quantization_recall` | Candidate recall from quantized coarse scoring, emitted only when quantization is enabled |
| `quantization_bias_ratio` | Average relative distance bias between quantized distance and decoded doc-pruned distance |
| `quantization_inversion_count_rate` | Candidate-pair order inversion rate introduced by quantization |

Metrics that require original base vectors output a `skipped_reason` object when the data is not
available. Original vectors are available inside the index when `use_reorder=true`; otherwise pass
SINDI `base_path` through the analyze parameters or the command-line option described below.

### Dynamic metrics from `AnalyzeIndexBySearch`

#### HGraph metrics

| Metric | Meaning |
| --- | --- |
| `recall_query` | Recall on the supplied query set against true nearest neighbors |
| `avg_distance_query` | Average distance between query vectors and retrieved neighbors |
| `time_cost_query` | Average per-query latency in milliseconds |
| `quantization_bias_ratio_query` | Quantization bias observed during query search |
| `quantization_inversion_count_rate_query` | Query-time ordering errors introduced by quantization |

#### SINDI metrics

| Metric | Meaning |
| --- | --- |
| `recall_query` | Search-result recall against supplied or generated sparse ground truth |
| `mean_latency_ms` | Average per-query latency measured while running `KnnSearch` |
| `time_cost_query` | Alias of `mean_latency_ms`, kept consistent with other analyzers |
| `postings_scanned.query_term_count_after_prune_mean` | Average number of query terms left after query pruning |
| `postings_scanned.query_term_with_posting_mean` | Average number of retained query terms that hit at least one non-empty posting list |
| `postings_scanned.posting_hit_mean` | Average hit ratio of retained query terms against non-empty posting lists |
| `doc_prune_recall` | Recall of doc-pruned pre-rerank candidates against sparse ground truth with query pruning disabled |
| `doc_prune_bias_mean` | Average relative distance bias between doc-pruned distance and exact sparse distance on sampled queries |
| `doc_prune_inversion_count_rate` | Candidate-pair order inversion rate introduced by document pruning on sampled queries |
| `quantization_recall` | Recall of quantized pre-rerank candidates, emitted only when quantization is enabled |
| `quantization_bias_ratio` | Average relative distance bias between quantized distance and decoded doc-pruned distance |
| `quantization_inversion_count_rate` | Candidate-pair order inversion rate introduced by quantization |
| `reorder_recall.before_reorder_recall_k_at_k` | Recall of coarse top-k candidates before precise reorder |
| `reorder_recall.after_reorder_recall_k_at_k` | Recall of final top-k candidates after precise reorder |
| `last_topk_rank_in_heap.mean` / `p95` / `p99` / `max` | Rank distribution of final top-k results inside the pre-rerank candidate heap |

SINDI dynamic recall and distance-quality metrics need ground truth. Pass `groundtruth_path` to
reuse an existing `.dev.gt` file, or pass `base_path` so the analyzer can generate exact sparse
ground truth. `save_groundtruth_path` can persist generated ground truth for later runs. Without
ground truth, those fields return `skipped_reason`; `postings_scanned` still runs because it only
needs the query and index postings.

Quantization-related fields differ by index type — they are not unified across implementations:

| Index | Field | Meaning |
| --- | --- | --- |
| `HGraph` | `quantization_bias_ratio_query` | Quantization bias observed during search |
| `HGraph` | `quantization_inversion_count_rate_query` | Quantization-induced ordering errors during search |
| `IVF` | `quantization_bias_ratio` | Quantization bias observed during search (only when `use_reorder_` is enabled) |
| `IVF` | `quantization_inversion_count_rate` | Quantization-induced ordering errors during search (only when `use_reorder_` is enabled) |

If you also need degree distribution, entry-point analysis or sub-index quality breakdown, look
in the `GetStats()` JSON instead — `AnalyzeIndexBySearch` focuses on dynamic, query-driven
signals.

## The `analyze_index` Tool

`analyze_index` is the user-facing wrapper around the analyzer APIs. It loads a serialized VSAG
index from disk, prints its metadata and `GetStats()` output, and (optionally) runs
`AnalyzeIndexBySearch` against a query file.

### Building

Tools are not built by default — enable them explicitly:

```bash
# via the project Makefile
VSAG_ENABLE_TOOLS=ON make release

# or directly through CMake
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release -j
# Output: ./build-release/tools/analyze_index/analyze_index
```

### Command-line arguments

| Argument | Alias | Required | Description |
| --- | --- | --- | --- |
| `--index_path` | `-i` | **Yes** | Path to the serialized VSAG index file. |
| `--build_parameter` | `-bp` | No | Build parameters (JSON) used when reloading the index. Defaults to the parameters embedded in the index file. |
| `--query_path` | `-qp` | No | Binary query dataset path. If omitted, only static analysis is performed. |
| `--query_data_type` | | No | Query dataset type: `auto`, `dense`, or `sparse`. `auto` uses sparse loading for SINDI. |
| `--base_path` | | No | Optional sparse CSR base dataset for SINDI analysis and ground-truth generation. |
| `--groundtruth_path` | | No | Optional SINDI `.dev.gt` ground-truth file. If present, it is reused. |
| `--save_groundtruth_path` | | No | Optional path for saving generated SINDI ground truth. |
| `--search_parameter` | `-sp` | No | Search parameters (JSON) used during dynamic analysis. |
| `--topk` | `-k` | No | Top-K for dynamic analysis (default `100`). |

The query file format is the simple binary `(uint32 rows, uint32 cols, float32 data...)` layout
consumed by `load_query()` in `tools/analyze_index/analyze_index.cpp`.

For SINDI, query and base datasets use CSR sparse binary layout:
`int64 nrow, int64 ncol, int64 nnz`, followed by `int64 indptr[nrow + 1]`,
`int32 indices[nnz]`, and `float32 data[nnz]`. SINDI ground truth uses `.dev.gt` layout:
`uint32 query_count, uint32 topk`, followed by flattened `int32 ids` and `float32 distances`.
If `--groundtruth_path` is not provided but `--base_path` is available, SINDI analysis generates
ground truth from the original sparse base vectors and can save it through `--save_groundtruth_path`.

### Two analysis modes

**1. Static analysis (no query file)**

```bash
./build-release/tools/analyze_index/analyze_index \
    --index_path /path/to/my_index.hgraph
```

Reports the index name, dimension, data type, metric, build parameters, and `GetStats()` JSON.

**2. Static + dynamic analysis**

```bash
./build-release/tools/analyze_index/analyze_index \
    --index_path /path/to/my_index.ivf \
    --query_path /path/to/queries.bin \
    --search_parameter '{"ivf":{"scan_buckets_count":16}}' \
    --topk 50
```

In addition to the static section, prints a `Search Analyze: { ... }` JSON block produced by
`AnalyzeIndexBySearch`.

When a serialized index only embeds `index_param`, `analyze_index` can still reload it without
`--build_parameter`; missing metadata fields are filled with analyzer defaults where possible.

## Typical Use Cases

- **Recall regression triage**: confirm whether a drop is caused by quantization
  (`quantization_*` metrics), graph structure (`connect_components`,
  `proximity_recall_neighbor`), or query-side parameters (`recall_query` vs. `recall_base`).
- **Capacity / health checks**: detect duplicated data (`duplicate_ratio`), disconnected
  components, or excessive deletions.
- **Parameter tuning**: re-run `AnalyzeIndexBySearch` with different `search_parameter` values to
  pick an operating point that balances `recall_query` and `time_cost_query` — without rebuilding
  the index.
- **What-if experiments**: override `--build_parameter` on load to evaluate alternative settings
  for indexes whose parameters are not embedded in the file.

## References

- API: `Index::AnalyzeIndexBySearch` in `include/vsag/index.h`
- Implementations: `src/analyzer/{analyzer,hgraph_analyzer,pyramid_analyzer}.h`
- Tool source: `tools/analyze_index/`
- Local tool entry point: `tools/analyze_index/README.md`
