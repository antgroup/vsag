# Index Analysis (`AnalyzeIndexBySearch` & `analyze_index`)

VSAG ships an introspection capability for inspecting an index that has already been built or
loaded, so you can diagnose recall regressions, quantization quality, graph health and search
performance **without** rebuilding the index. This capability is exposed in these ways:

- the C++ APIs `Index::GetStats` and `Index::AnalyzeIndexBySearch` (declared in
  `include/vsag/index.h`) for an index object in your process;
- for a serialized index file, one shared implementation, `vsag::analyze_index()` (declared in
  `include/vsag/analyze_index.h`), used by:
  - the `pyvsag-analyze-index` command and the `pyvsag.analyze_index()` Python API, installed
    with `pip install pyvsag`;
  - the C++ command-line tool `analyze_index`, built from `tools/analyze_index/`.

See [Analyzing index files](#analyzing-index-files) for the file-based entry points.

## The `AnalyzeIndexBySearch` API

```cpp
// include/vsag/index.h
virtual std::string
AnalyzeIndexBySearch(const SearchRequest& request);
```

- **Input**: a `SearchRequest` (query dataset + `topk` + search parameter JSON).
- **Output**: a JSON-formatted string containing dynamic, query-driven metrics.
- **Supported indexes**: all factory-registered indexes: `HGraph`, `IVF`, `Pyramid`, `SINDI`,
  `SINDI_V2`, `SIMQ`, `BruteForce`, `WARP`, and `LazyHGraph`.

It is complementary to `Index::GetStats()`, which reports static structural properties of the
index without needing query data. For graph-based indexes, additional graph-health details such
as degree distribution, entry-point quality, sub-index recall and low-recall hot-spots are
exposed through `GetStats()` rather than through `AnalyzeIndexBySearch`.

### Common output contract

Every successful static or dynamic analysis returns a JSON object with an `_analysis` member:

```json
{
  "_analysis": {
    "schema_version": 1,
    "index_type": "brute_force",
    "analysis_type": "search",
    "status": "partial",
    "sample_count": 10,
    "topk": 10,
    "consistency": "weak_snapshot",
    "skipped": { "recall_query": "ground truth was not requested" }
  }
}
```

Existing index-specific fields remain at the JSON root for compatibility. `status` is `complete`
when all applicable metrics were computed, `partial` when individual metrics are listed under
`skipped`, and `not_applicable` when the request is valid but no metric can be computed (for
example, search analysis on an empty index). Invalid arguments and execution failures still throw.
Specialized analyzers may report an unavailable root-level metric as an object containing a
string `skipped_reason`; the common metadata layer mirrors that reason into
`_analysis.skipped.<metric>`. Baseline analyzers write the same public representation directly to
`_analysis.skipped`. Consumers should rely on `_analysis.skipped`, while `skipped_reason` remains
an index-specific compatibility detail.

`BruteForce` and `WARP` provide exact-search baseline counts and latency. `SINDI_V2` and `SIMQ`
provide baseline counts and query latency while index-specific quality metrics are reported as
skipped. `LazyHGraph` delegates to its active BruteForce or HGraph backend and adds its current
`phase` and `transition_threshold`; its reported `index_type` remains `lazy_hgraph`.

Analysis is synchronous and intended for offline diagnostics. It can scan substantial parts of an
index and specialized analyzers may compute exact ground truth. Analysis never changes search or
serialization configuration and is safe with concurrent read-only search. Unless an index offers a
stronger lock-protected view, `_analysis.consistency` is `weak_snapshot`; do not run analysis
concurrently with Add, Remove, Update, or phase transition.

Pyramid query analysis follows the same path-scoping rules as `KnnSearch`. For the default
hierarchy, attach paths with `Dataset::Paths`. For a named hierarchy, use
`Dataset::Paths(hierarchy_name, paths)` and select that hierarchy in the Pyramid search parameters.
A path is required when the selected hierarchy's root is `NO_INDEX`. In a batched query dataset,
the outer path collection has one row per query. The legacy overload stores one path string per
row and interprets `|` as a union; the structured overload stores one or many atomic paths
per row and treats `|` as an ordinary path character. Pyramid calculates ground truth only from
vectors eligible for each query's hierarchy and path union, excluding mark-removed vectors.
Pyramid query analysis currently supports KNN requests only and uses `query_`, `topk_`, and
`params_str_` from `SearchRequest`; filtered and iterator requests are not supported. Run analysis
against a stable index without concurrent add or remove operations.

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
| `build_cache_hit_rate` | Fraction of nodes warm-started from an imported cache during the last `Build()`; emits a `skipped_reason` when the index was not built from a cache imported via `ImportCache()` |
| `build_cache_hit_nodes` / `build_cache_missed_nodes` | Node counts behind `build_cache_hit_rate` (only present when the index was built from an imported cache) |

#### Pyramid metrics

| Metric | Meaning |
| --- | --- |
| `total_count` | Total number of vectors in the index |
| `index_node_structure` | Path-tree node count, depth, nodes by level, and node-status distribution |
| `leaf_node_size_distribution` | Distribution of vector counts across leaf nodes |
| `subindex_quality` | Per-leaf GRAPH/FLAT status and size, plus graph/flat counts and vector coverage |
| `recall_base` | Size-weighted recall across graph-backed nodes; present when sampling and analysis search parameters are available |
| `graph_node_count` / `total_graph_size` | Number and combined size of graph-backed nodes included in recall analysis |
| `skipped_node_count` / `low_recall_nodes` | Nodes excluded from recall analysis and details of nodes below the recall threshold |
| `duplicate_ratio` | Duplicate-vector ratio within the hierarchy |
| `hierarchy_count` / `hierarchies` | Multi-hierarchy count and a per-name object containing the metrics above; single unnamed hierarchies expose them at the top level |

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

#### Pyramid metrics

| Metric | Meaning |
| --- | --- |
| `recall_query` | Recall against exact ground truth within each query's selected hierarchy and path |
| `avg_distance_query` | Average distance of the path-scoped exact ground-truth neighbors |
| `time_cost_query` | Average per-query latency in milliseconds |
| `quantization_error_query` | Query-time distance error introduced by base quantization; emitted when reorder is enabled |
| `quantization_inversion_ratio_query` | Query-time ordering inversion ratio introduced by base quantization; emitted when reorder is enabled |

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
| `Pyramid` | `quantization_error_query` | Quantization distance error during search (only when reorder is enabled) |
| `Pyramid` | `quantization_inversion_ratio_query` | Quantization-induced ordering inversions during search (only when reorder is enabled) |

For static structure and health information, including degree distributions, entry-point
analysis, and Pyramid subindex quality, use the `GetStats()` JSON.

## Analyzing Index Files

`vsag::analyze_index()` analyzes a serialized index file. It detects the serialization format,
reloads the index without building or modifying it, reports its metadata and the `GetStats()`
statistics, and runs `AnalyzeIndexBySearch()` when a query file is given. Three entry points
share this implementation and return the same JSON report:

| Entry point | Comes with | Typical use |
| --- | --- | --- |
| `pyvsag-analyze-index` command | `pip install pyvsag` | Shell scripts and CI jobs |
| `pyvsag.analyze_index()` | `pip install pyvsag` | Python programs and notebooks |
| `vsag::analyze_index()` and the `analyze_index` tool | The C++ library; a source build with tools | C++ applications |

### Installation

```bash
pip install pyvsag==<version>
pyvsag-analyze-index --index_path /path/to/index
```

The wheel contains the command, the Python API and the native library, so analysis needs no
VSAG source tree, CMake, C++ compiler or NumPy. Wheels follow the release matrix of `pyvsag`
(Linux x86_64 and aarch64, CPython 3.6 to 3.14). Install a `pyvsag` release that can read the
index, that is, one not older than the VSAG release that wrote it; the report records the
version that read the file in `compatibility.vsag_version`.

If the scripts directory of the Python environment is not on `PATH`, run
`python -m pyvsag_analyze_index` with the same arguments.

### Command line

```bash
# Static analysis: metadata and statistics.
pyvsag-analyze-index --index_path my_index.vsag > report.json

# Static and dynamic analysis, 10 neighbors per query, a custom search parameter.
pyvsag-analyze-index -i my_index.vsag -qp queries.bin -k 10 \
    -sp '{"hgraph": {"ef_search": 200}}'
```

| Argument | Alias | Default | Description |
| --- | --- | --- | --- |
| `--index_path` | `-i` | Required | Serialized index file. |
| `--build_parameter` | `-bp` | `default` | Build parameters (JSON) used to reload the index. `default` uses the parameters recorded in the file; files that do not record them need this argument (see [Supported files](#supported-files-and-index-types)). |
| `--query_path` | `-qp` | None | Query dataset (see [Input files](#input-files)). Without it, only static analysis runs. This is not Pyramid hierarchy-path metadata. |
| `--query_data_type` | | `auto` | `auto` (sparse for sparse indexes, dense otherwise), `dense` or `sparse`. |
| `--base_path` | | None | SINDI only: sparse CSR base dataset used for exact ground truth and for the metrics that need original vectors. |
| `--groundtruth_path` | | None | SINDI only: existing ground truth for dynamic analysis. Cannot be combined with `--save_groundtruth_path`. |
| `--save_groundtruth_path` | | None | SINDI only: where to save the ground truth generated during dynamic analysis. |
| `--search_parameter` | `-sp` | `default` | Search parameters (JSON) for dynamic analysis. `default` uses `{"hgraph": {"ef_search": max(100, topk)}}` for HGraph and LazyHGraph, `{"pyramid": {"ef_search": max(100, topk)}}` for Pyramid, `{"sindi_v2": {}}` for SINDI_V2 and `{}` for the other index types. |
| `--topk` | `-k` | `100` | Number of neighbors evaluated for each query. |
| `--compact` | | Off | Print the report on one line instead of indented. |
| `--version` | | | Print the version and exit. |

Every option also accepts a dashed spelling, such as `--index-path`. The report is the
only output on stdout. Diagnostics, including the start-up banner and the log messages of the
native library, go to stderr; `VSAG_LOG_LEVEL` (`off`, `error`, `warn`, `info`, ...) sets their
level. A failure prints one line to stderr and nothing to stdout:

```text
pyvsag-analyze-index: error [missing_file]: index_path does not exist: /data/missing.vsag
```

| Exit status | Meaning |
| --- | --- |
| `0` | The report was printed. |
| `1` | The analysis failed; the bracketed [error code](#errors) tells why. |
| `2` | The command line is invalid, for example a missing `--index_path` or a non-numeric `--topk`. |

### Python API

```python
import pyvsag

report = pyvsag.analyze_index(
    "my_index.vsag",
    build_parameter="default",
    query_path="queries.bin",
    query_data_type="auto",
    base_path=None,
    groundtruth_path=None,
    save_groundtruth_path=None,
    search_parameter={"hgraph": {"ef_search": 200}},
    topk=10,
)
print(report["dynamic_analysis"]["result"]["recall_query"])
```

The arguments mean the same as the command-line options. Paths accept `str` or `os.PathLike`
and `None` for "not given"; `build_parameter` and `search_parameter` accept a JSON string, a
`dict` or `None` (same as `"default"`). All arguments after `index_path` are keyword-only. The
function returns a `dict` that `json.dumps()` can serialize, raises `TypeError` for arguments of
the wrong type, and raises `pyvsag.IndexAnalysisError`, a `RuntimeError`, for every other
failure. Its `code` attribute holds the [error code](#errors) and `message` the explanation:

```python
try:
    report = pyvsag.analyze_index("my_index.vsag")
except pyvsag.IndexAnalysisError as error:
    print(error.code, error.message)
```

`pyvsag.ANALYZE_INDEX_SCHEMA_VERSION` is the `schema_version` of the reports. The analysis
releases the GIL while it runs and does not change the log level. Unlike the command, the
Python API does not redirect the native log messages: as for any other `pyvsag` call, `info` and
`debug` messages go to the process stdout and warnings and errors to stderr. Set
`VSAG_LOG_LEVEL=warn` or call `pyvsag.set_logger_off()` to keep them off stdout.

### C++ API

```cpp
#include <vsag/vsag.h>

vsag::AnalyzeIndexOptions options;
options.index_path = "my_index.vsag";
options.query_path = "queries.bin";
options.topk = 10;
auto report = vsag::analyze_index(options);
if (report.has_value()) {
    std::cout << report.value() << std::endl;  // the JSON report
} else {
    std::cerr << report.error().message << std::endl;  // report.error().type tells why
}
```

`AnalyzeIndexOptions` has one field per command-line option; an empty string means "not
given". `vsag::analyze_index()` returns the report as a compact JSON object string, or a
`vsag::Error` whose `type` corresponds to the [error code](#errors).

### Report format

Reports are JSON objects with these members, in this order:

| Member | Content |
| --- | --- |
| `schema_version` | Version of this format, currently `1`. |
| `metadata` | What was loaded: the index type, its dimension, data type, metric, size and parameters. |
| `static_analysis` | The `GetStats()` JSON of the index; see [static metrics](#static-metrics-from-getstats). |
| `dynamic_analysis` | `null` without a query file. Otherwise the analysis inputs and, under `result`, the `AnalyzeIndexBySearch()` JSON; see [dynamic metrics](#dynamic-metrics-from-analyzeindexbysearch). |
| `warnings` | Objects with a `code` and a `message`, for assumptions and overrides that affect how to read the results. |
| `compatibility` | How the file was read: its format, the reading library version and the completeness of its metadata. |

`static_analysis` and `dynamic_analysis.result` keep their own `_analysis` member described in
the [common output contract](#common-output-contract); its `schema_version` versions those
analyzer outputs, independently of the top-level `schema_version`.

| `metadata` member | Meaning |
| --- | --- |
| `index_path`, `file_size_bytes` | The analyzed file. |
| `index_type` | Factory name of the index, for example `hgraph` or `sindi`. |
| `dim` | Vector dimension; `null` for sparse indexes. |
| `data_type`, `metric`, `extra_info_size` | Common build parameters of the loaded index. |
| `num_elements`, `memory_usage_bytes` | `GetNumElements()` and `GetMemoryUsage()` of the loaded index. |
| `parameter_source` | `file` when the recorded parameters were used, `user` when `build_parameter` was. |
| `index_param` | The recorded index parameters, in the layout of the index parameter `ToJson()`, or `index_param` of `build_parameter`. |
| `build_parameter` | The given `build_parameter`, or `null`. |
| `basic_info` | The `basic_info` metadata recorded in the file, without `index_param`; `null` for legacy files. |

| `dynamic_analysis` member | Meaning |
| --- | --- |
| `query_path`, `query_data_type` | The query file and how it was read (`dense` or `sparse`). |
| `query_count`, `query_dim` or `query_cols` | Number of queries, and their dimension (dense) or column count (sparse). |
| `topk` | Neighbors evaluated for each query. |
| `search_parameter`, `search_parameter_source` | The search parameters used, and whether they are the `default` or the `user` value. |
| `base_path`, `groundtruth_path`, `save_groundtruth_path` | Present when given. |
| `result` | The `AnalyzeIndexBySearch()` JSON. |

| Warning `code` | Meaning |
| --- | --- |
| `metadata_defaults` | The file does not record the data type or the metric of its index, and the analyzer assumed `float32` or `l2` (listed in `compatibility.defaulted_fields`). BruteForce, WARP and SIMQ footer files are affected; pass `build_parameter` when the index used another metric or data type. |
| `build_parameter_override` | The index was loaded with `build_parameter` instead of the parameters recorded in the file. |
| `legacy_format` | The file has no VSAG footer and was loaded as a legacy file with `build_parameter`. |
| `memory_io` | The recorded parameters use file-backed IO (`buffer_io`, `async_io`, `mmap_io`, `uring_io`), or `reader_io` in a footer, chunked or HGraph streaming file. The file carries the data inline, so those components were loaded into memory instead, and no recorded file path was opened (`compatibility.memory_io_overrides`). Streaming files with file-backed IO are loaded this way for HGraph, BruteForce and WARP and rejected with `unsupported_index_operation` for the other index types. |
| `empty_index` | The index contains no vectors. |

| `compatibility` member | Meaning |
| --- | --- |
| `vsag_version` | Version of the library that read the file. |
| `serialization_format` | `footer`, `chunked`, `streaming` or `legacy`; see [supported files](#supported-files-and-index-types). |
| `lazy_hgraph_container` | Whether the file was written by a LazyHGraph index. |
| `metadata_complete`, `defaulted_fields` | Whether the analyzer had to assume common parameters, and which values it assumed. |
| `memory_io_overrides` | Parameters whose recorded IO type was replaced with memory IO. |
| `chunked_codec`, `chunk_size` | Chunked files only. |
| `streaming_blocks` | Streaming files only: names of the body blocks. |

An abridged report of an HGraph index with dynamic analysis:

```json
{
  "schema_version": 1,
  "metadata": {
    "index_path": "my_index.vsag",
    "file_size_bytes": 169113,
    "index_type": "hgraph",
    "dim": 16,
    "data_type": "float32",
    "metric": "l2",
    "extra_info_size": 0,
    "num_elements": 500,
    "parameter_source": "file",
    "index_param": {"type": "hgraph", "...": "..."},
    "build_parameter": null,
    "basic_info": {"total_count": 500, "...": "..."}
  },
  "static_analysis": {
    "_analysis": {
      "analysis_type": "stats",
      "consistency": "weak_snapshot",
      "index_type": "hgraph",
      "schema_version": 1,
      "skipped": {"build_cache_hit_rate": "index was not built from an imported cache"},
      "status": "partial"
    },
    "connect_components": 5,
    "recall_base": 0.989,
    "total_count": 500,
    "...": "..."
  },
  "dynamic_analysis": {
    "query_path": "queries.bin",
    "query_data_type": "dense",
    "query_count": 20,
    "query_dim": 16,
    "topk": 10,
    "search_parameter": {"hgraph": {"ef_search": 100}},
    "search_parameter_source": "default",
    "result": {
      "_analysis": {"analysis_type": "search", "sample_count": 20, "status": "complete", "...": "..."},
      "recall_query": 0.995,
      "time_cost_query": 0.788,
      "...": "..."
    }
  },
  "warnings": [],
  "compatibility": {
    "vsag_version": "v1.2.0",
    "serialization_format": "footer",
    "lazy_hgraph_container": false,
    "metadata_complete": true,
    "defaulted_fields": {},
    "memory_io_overrides": []
  }
}
```

### Supported files and index types

The analyzer reads one serialized index file. It recognizes the format from the file itself:

| Format | Written by | Analysis |
| --- | --- | --- |
| `footer` | `Index::Serialize(std::ostream&)`, `pyvsag.Index.save()` | Supported. The parameters recorded in the footer are used unless `build_parameter` is given. |
| `chunked` | `Index::Serialize(SerializeWriter&, chunk_size)` | Supported without compression. Compressed files need the decompressing `DeserializeReader` of the application that wrote them and fail with `unsupported_index_operation`. |
| `streaming` | `Index::SerializeStreaming()` | Supported. The file is loaded with `Index::Load()` and its recorded parameters; `build_parameter` is rejected. |
| `legacy` | VSAG before 0.15, or files without a footer | Only with `build_parameter`, which must name the index type in `"type"` when the file does not record it. |

`BinarySet` and `ReaderSet` serializations consist of several parts and are not single files,
so they are not supported.

| Index type | Static analysis from the file alone | Dynamic analysis |
| --- | --- | --- |
| `hgraph` | Footer, chunked and streaming files. | Dense float32 queries. |
| `ivf` | Footer and streaming files. | Dense float32 queries. |
| `brute_force` | Footer files assume `l2` and `float32` (`metadata_defaults`); streaming files record them. | Dense float32 queries. |
| `pyramid` | Streaming files. Footer files do not record the dimension, metric or data type and need `build_parameter`. | Dense float32 queries, when the root of the searched hierarchy is built. Hierarchies that need per-query paths must be analyzed through the C++ API with `Dataset::Paths`. |
| `sindi` | Footer and streaming files. `base_path` adds the metrics that need original vectors. | Sparse queries; `base_path`, `groundtruth_path` and `save_groundtruth_path` apply. |
| `sindi_v2` | Footer and streaming files. | Sparse queries (baseline metrics). |
| `warp`, `simq` | Footer files, assuming `l2` and `float32` (`metadata_defaults`). | Not supported: multi-vector queries cannot be expressed in query files. |
| `lazy_hgraph` | Needs `build_parameter`: the file does not record the LazyHGraph parameters. | Dense float32 queries. |

Dense query files hold float32 vectors, so dynamic analysis of `int8`, `float16` and `bfloat16`
indexes fails with `unsupported_index_operation`; their static analysis is available, except
that the HGraph statistics do not support `float16` and `bfloat16` indexes yet.

### Errors

| Code | Cause |
| --- | --- |
| `invalid_argument` | An option or a combination of options is invalid: malformed JSON, `topk` not positive, `groundtruth_path` without `query_path`, `base_path` for an index that is not SINDI, a query dimension that differs from the index, `build_parameter` for a streaming file or one that names another index type. |
| `missing_file` | An input file, or the directory of `save_groundtruth_path`, does not exist. |
| `read_error` | An input file cannot be read, or ended while the index was loaded. |
| `invalid_binary` | The index file is not a VSAG index, is truncated or damaged (for example a footer checksum mismatch), or a query, base or ground-truth file does not match its documented layout. |
| `unsupported_index` | The file does not identify a supported index type. |
| `unsupported_index_operation` | The file or analysis is not supported: compressed chunked files, LazyHGraph and Pyramid footer files without `build_parameter`, or dynamic analysis that the index or the query files cannot support. |
| `index_empty` | The file holds an index serialized while empty, without recorded parameters. |
| `no_enough_memory` | The index or the queries do not fit in memory. |
| `internal_error`, `unknown_error` | An unexpected failure inside the index. |

The other error codes of `vsag::ErrorType` can be reported when an index implementation returns
them. Messages name the file or option involved and, where possible, how to fix the problem.

### Input files

All integers and floats are little-endian.

- **Dense query file**: `uint32 rows, uint32 dim`, then `float32 data[rows * dim]`. The file
  size must match the header exactly.
- **Sparse CSR file** (sparse queries and SINDI base datasets): `int64 rows, int64 cols,
  int64 nnz`, then `int64 indptr[rows + 1]`, `int32 indices[nnz]` and `float32 values[nnz]`.
  `indptr` must start at 0, never decrease and end at `nnz`; indices must be in `[0, cols)`.
- **Ground truth** (`.dev.gt`): `uint32 rows, uint32 topk`, then `int32 ids[rows * topk]` and
  `float32 distances[rows * topk]`. It needs one row per query and at least
  `min(topk, num_elements)` neighbors per row.

Input files are validated before the index is loaded, so a malformed file fails fast.

### The C++ `analyze_index` tool

The C++ tool is built from source with the other tools:

```bash
# via the project Makefile
VSAG_ENABLE_TOOLS=ON make release

# or directly through CMake
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release -j
./build-release/tools/analyze_index/analyze_index --index_path my_index.vsag
```

It takes the same options, including the dashed spellings, prints the same report, and uses the
same exit statuses and error format (`analyze_index: error [<code>]: <message>`). It replaces
the log lines printed by earlier versions with the JSON report. `--query_path empty` is still
accepted and means "no query file". On Linux the tool keeps the start-up banner of the library
off stdout; on other platforms set `VSAG_SUPPRESS_INIT_BANNER=1` to do so.

### Limitations

- The whole index is loaded into memory, in addition to the queries.
- Streaming files with file-backed storage are analyzed only when the storage can be loaded into
  memory, which is the case for the base codes, precise codes and raw vectors of HGraph. Any other
  file-backed component, such as BruteForce or WARP codes or a split RaBitQ supplement, fails with
  `unsupported_index_operation`, because loading it would use the path recorded in the file.
- Query files carry vectors only: no Pyramid hierarchy paths, filters, attributes or
  multi-vector queries. Use `AnalyzeIndexBySearch` from C++ for those.
- Analysis is synchronous and meant for offline diagnostics; the costs described in the
  [common output contract](#common-output-contract) apply.
- Footers and streaming headers are validated (checksums and layouts), but the index body is
  read by the deserializer of the index. A damaged body, or a footer-less file loaded with
  `build_parameter`, is only as robust as that deserializer: garbage input can fail with
  `no_enough_memory` or use a lot of memory. Analyze untrusted files where memory is limited.

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
- File analysis API: `vsag::analyze_index` in `include/vsag/analyze_index.h`, implemented in
  `src/analyze_index.cpp`
- Python API and command: `python/pyvsag/index_analysis.py` and
  `python/pyvsag_analyze_index.py`
- Implementations: `src/analyzer/{analyzer,hgraph_analyzer,pyramid_analyzer}.h`
- Tool source: `tools/analyze_index/`
- Local tool entry point: `tools/analyze_index/README.md`
