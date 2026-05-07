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
- **Supported indexes**: currently `HGraph` and `IVF`. `Pyramid` only supports static analysis
  through `GetStats()` — it does not yet override `AnalyzeIndexBySearch`. Indexes that do not
  implement this API will throw an exception when called.

It is complementary to `Index::GetStats()`, which reports static structural properties of the
index without needing query data. For graph-based indexes, additional graph-health details such
as degree distribution, entry-point quality, sub-index recall and low-recall hot-spots are
exposed through `GetStats()` rather than through `AnalyzeIndexBySearch`.

### Static metrics from `GetStats()`

| Metric | Meaning |
| --- | --- |
| `total_count` | Total number of vectors in the index |
| `deleted_count` | Vectors marked for deletion |
| `connect_components` | Connected components in the proximity graph |
| `duplicate_ratio` | Proportion of duplicate vectors in the dataset |
| `avg_distance_base` | Average pairwise distance on a sample of base vectors |
| `recall_base` | Self-recall on a sample of base vectors (sanity check) |
| `proximity_recall_neighbor` | Recall of neighbor lists vs. the true nearest neighbors |
| `quantization_bias_ratio` | Bias introduced by quantized distance vs. exact distance |
| `quantization_inversion_count_rate` | Rate of distance-order inversions caused by quantization |

### Dynamic metrics from `AnalyzeIndexBySearch`

Common query-driven metrics produced by HGraph (IVF currently does **not** emit these recall /
distance / latency fields — see below):

| Metric | Meaning |
| --- | --- |
| `recall_query` | Recall on the supplied query set against true nearest neighbors |
| `avg_distance_query` | Average distance between query vectors and retrieved neighbors |
| `time_cost_query` | Average per-query latency (milliseconds) |

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
| `--search_parameter` | `-sp` | No | Search parameters (JSON) used during dynamic analysis. |
| `--topk` | `-k` | No | Top-K for dynamic analysis (default `100`). |

The query file format is the simple binary `(uint32 rows, uint32 cols, float32 data...)` layout
consumed by `load_query()` in `tools/analyze_index/analyze_index.cpp`.

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
- Tool README: `tools/analyze_index/README.md`
