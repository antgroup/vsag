# Performance Evaluation Tool (`eval_performance`)

`eval_performance` is the command-line performance evaluation tool shipped with VSAG, under
`tools/eval/`. After building, the binary lives at `build-release/tools/eval/eval_performance`. It
is used to compare throughput, latency, and recall across different indexes or parameter
combinations.

## Building

Tools are not built by default — enable them explicitly:

```bash
# via the project Makefile
VSAG_ENABLE_TOOLS=ON make release
# or: make dev

# or directly through CMake
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release -j
# Output: ./build-release/tools/eval/eval_performance
```

HDF5 must be installed on the system (Ubuntu: `apt install libhdf5-dev`; CentOS:
`yum install hdf5-devel`).

## Two Modes

### 1. Command-line mode (quick, one-off experiments)

```bash
./build-release/tools/eval/eval_performance \
    --datapath /tmp/sift-128-euclidean.hdf5 \
    --index_name hgraph \
    --type search \
    --create_params '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}' \
    --search_params '{"hgraph":{"ef_search":60}}' \
    --topk 10
```

Useful flags include `--search_mode` (`knn` / `range` / `knn_filter` / `range_filter`),
`--search-query-count`, `--warmup-query-count`, `--delete-index-after-search`, `--set_immutable`, and the various
`--disable_*` switches that turn off individual metrics. `--set_immutable` defaults to `false`; when
enabled, the evaluator calls `Index::SetImmutable()` after loading/building the index and before
search. An index that does not support this operation causes the evaluation to fail. The reference
template at `tools/eval/eval_template.yaml` shows the complete YAML shape.

### 2. Config-file mode (batch comparisons)

The YAML file is passed directly as a positional argument (no `--config` flag):

```bash
./build-release/tools/eval/eval_performance my_eval.yaml
```

A reference template is available at `tools/eval/eval_template.yaml`. A single configuration can
define multiple named cases, plus an optional `global` section that holds shared settings such as
thread counts, exporters, and an embedded HTTP monitor.

A minimal example:

```yaml
global:
  num_threads_building: 8
  num_threads_searching: 16
  exporters:
    print-directly:
      to: stdout
      format: table
    save-to-file:
      to: "file:///tmp/eval_results.json"
      format: json

eval_case1:
  datapath: /tmp/sift-128-euclidean.hdf5
  type: search
  index_name: hgraph
  create_params: '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}'
  search_params: '{"hgraph":{"ef_search":60}}'
  index_path: /tmp/vsag_eval/hgraph_fp32
  set_immutable: false
  warmup_query_count: 10000
  topk: 10
```

Note: under `global.exporters`, each entry is a **named** exporter (a YAML map), not a list item.

## Concurrency A/B Harness

For a repeatable baseline/candidate comparison over outer query concurrency, use
`tools/eval/run_concurrency_ab.py`. It invokes a separate evaluator binary for each variant one
case at a time, so the search measurement semantics remain those of `eval_performance`, while the
harness fixes the input parameters, runs concurrency `1/2/4/8/16/32`, alternates which variant
leads each pair, and writes one JSON report containing every raw evaluator result.

Prepare one shared serialized index, one evaluator binary, and the matching `libvsag.so` for each
variant first. The harness uses `type: search` and does not build or modify the index:

```bash
python3 tools/eval/run_concurrency_ab.py \
    --spec tools/eval/concurrency_ab.example.json \
    --output /tmp/vsag-concurrency-ab.json \
    --hash-files
```

The spec is JSON so it can be generated and reviewed without an additional YAML library. The
copy at `tools/eval/concurrency_ab.example.json` contains the full shape. `datapath`,
`index_path`, `create_params`, `search_params`, `topk`, `search_query_count`,
`warmup_query_count`, and `set_immutable` are shared by both variants. Each variant object must
contain exactly `eval_binary` and `shared_library`. The harness removes ambient loader overrides,
sets `LD_LIBRARY_PATH` to that library's parent, and runs `ldd` with the same environment. It
rejects equal evaluator realpaths or hashes and equal resolved `libvsag.so` realpaths or hashes;
there is no index-only mode. `set_immutable` defaults to `false`; when enabled, both evaluator
binaries call `Index::SetImmutable()` after deserialization and before warmup/search. Warmup defaults
to 10000 unmeasured KNN queries and uses the same OpenMP/query-repeat loop as measurement. Set
`rounds` to at least 5 for repeated pairs. `--hash-files` adds SHA-256 fingerprints for the dataset
and shared index; evaluator and resolved library fingerprints are always included in the report.

Each entry in `runs` has the following acceptance fields:

```json
{
  "variant": "baseline",
  "pair_number": 1,
  "outer_concurrency": 1,
  "evaluator_binary": "/path/to/baseline/eval_performance",
  "evaluator_realpath": "/path/to/baseline/eval_performance",
  "evaluator_sha256": "...",
  "resolved_libvsag_realpath": "/path/to/baseline/libvsag.so.0.0.0",
  "resolved_libvsag_sha256": "...",
  "index_path": "/path/to/shared.index",
  "errors": 0,
  "error_count": 0,
  "qps": 1234.5,
  "p50_ms": 0.81,
  "p99_ms": 1.42,
  "recall_avg": 0.99,
  "raw_result": {}
}
```

The optional `acceptance` spec section enables paired performance gates. Its defaults are
`qps_min_pct: 15`, `p99_max_regression_pct: 10`, `recall_max_abs_change: 0.01`,
`target_concurrency: [8, 16, 32]`, and `min_valid_pairs: 3`:

```json
"acceptance": {
  "qps_min_pct": 15,
  "p99_max_regression_pct": 10,
  "recall_max_abs_change": 0.01,
  "target_concurrency": [8, 16, 32],
  "min_valid_pairs": 3
}
```

For each `pair_number`, the harness matches the baseline and candidate run from the same round and
outer concurrency. It calculates `qps_change_pct = (candidate / baseline - 1) * 100`, the same
percentage formula for `p99_change_pct`, and `recall_abs_change = abs(candidate - baseline)`. The
`acceptance.by_concurrency` entries report the median of these paired changes and a `pass`, `fail`,
or `not_target` decision. The full paired records are in `acceptance.pairs`:

```json
{
  "status": "pass",
  "thresholds": {
    "qps_min_pct": 15.0,
    "p99_max_regression_pct": 10.0,
    "recall_max_abs_change": 0.01,
    "target_concurrency": [8, 16, 32],
    "min_valid_pairs": 3
  },
  "by_concurrency": [
    {
      "outer_concurrency": 8,
      "target": true,
      "status": "pass",
      "qps_change_pct": 16.2,
      "p99_change_pct": 3.1,
      "recall_abs_change": 0.002
    }
  ],
  "reasons": []
}
```

`acceptance.status` is separate from the harness execution `status` and `errors`. A failed run,
missing metric, unequal sample/success/error counts, incomplete pair, zero baseline for a percentage
metric, or fewer than `min_valid_pairs` valid pairs is recorded in `acceptance.reasons` and fails
the affected concurrency; invalid pairs never enter a median. A healthy non-target concurrency is
`not_target`, while an invalid pair is still `fail` so execution defects are visible. Performance
failure alone leaves a successful harness run as `status: "ok", errors: 0`.

`outer_concurrency` is the evaluator's `num_threads_searching` OpenMP query-loop count. Keep any
`parallel_search_thread_count` inside `search_params` unchanged when comparing variants. Each
YAML case exports JSON to an absolute temporary `file:///...` URI; evaluator stdout/stderr is
retained only as diagnostic text and is never parsed as the result. The report records
`cpu.sched_affinity`, affinity CPU count, logical CPU count, and structured oversubscription warnings
without refusing a requested concurrency. `summary` contains the median of each metric for every
variant/concurrency pair across `rounds`; `runs` preserves execution order and the complete
one-case `eval_performance` JSON under `raw_result`.

The harness `errors` field is a **run-level** error count (`error_scope` is
`evaluator_invocation`): it is zero for a successful evaluator process and one for a timeout,
non-zero exit, launch failure, or invalid JSON. The current native
evaluator aborts on the first query error instead of returning a per-query error count, so a failed
run cannot report the exact number of failed queries. Successful runs still expose
`measurement_successful_query_count` and `error_count` in `raw_result`; paired acceptance requires
all three counts to exist, match, and have zero errors. `recall_avg` is aggregate recall, not a
per-query result-ID comparison. The CLI returns 0 for execution plus acceptance success, 1 for
harness/execution failure, and 2 for a completed run whose performance acceptance fails.

Run the standard-library tests for the harness from WSL with:

```bash
python3 -m unittest discover -s tools/eval -p 'test_run_concurrency_ab.py' -v
```

## Supported Dimensions

- **Efficiency**: QPS, TPS
- **Quality**: average recall and quantile recall (P0/P10/P50/P90...)
- **Latency**: average, P50/P95/P99
- **Resource**: peak memory usage

### Search Measurement Semantics

- **Latency** is the elapsed wall duration of each measured `Index::KnnSearch` call, measured
  with the monotonic `std::chrono::steady_clock`.
- **QPS** is the number of successful queries divided by the wall duration, in seconds, of the
  measured search pass.
- Statistics extraction, recall calculation, and memory sampling run outside the performance
  pass and do not contribute to latency or QPS.
- Before monitors start, `warmup_query_count` KNN queries run with the same query repetition,
  OpenMP thread count, and `schedule(dynamic)` loop as measurement. Warmup is excluded from all
  latency, recall, QPS, and sample counters; any warmup search error fails the invocation.

JSON results include `measurement_method`, `measurement_sample_count`,
`measurement_successful_query_count`, `error_count`, and `measurement_duration(s)` so the measurement population
and method can be audited.

Results produced by older versions that measured intervals between monitor callbacks are not
directly comparable with results that use these semantics.

## Search Modes

`search_mode` accepts `knn`, `range`, `knn_filter`, and `range_filter`.

## Output Formats and Destinations

Each exporter combines a `format` with a `to` destination.

- Formats: `table` (or its alias `text`), `json`, `line_protocol` (for InfluxDB).
- Destinations:
    - `stdout` — print to standard output.
    - `file://<path>` — write (overwrite) to a file.
    - `influxdb://<host>:<port>/<path>?<query>` — POST to an InfluxDB v2 endpoint. Use
      `format: line_protocol` and pass an authentication token via `vars.token` (the value must
      include the `Token ` prefix, e.g. `Token <your-influxdb-token>`).

If no exporter is configured, results are printed to stdout in `table` format by default.

## HTTP Monitor (optional)

When configured, the tool starts an embedded HTTP server for the duration of a batch run and
exposes live progress (current case, total cases, completion %) plus the latest metrics. This is
helpful for long-running evaluations.

```yaml
global:
  http_server:
    enabled: true
    port: 8080
```

## Datasets

Any HDF5 dataset from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
(e.g. `sift-128-euclidean.hdf5`, `gist-960-euclidean.hdf5`) works out of the box.

## References

- Source: `tools/eval/`
- Local tool entry point: `tools/eval/README.md`
- Reference numbers on standard hardware: [Reference Performance](performance.md).
