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
`--search-query-count`, `--recall_target`, `--delete-index-after-search`, and the various
`--disable_*` switches that turn off individual metrics. The reference template at
`tools/eval/eval_template.yaml` shows the complete YAML shape.

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
  topk: 10
  recall_target: 0.9
```

Note: under `global.exporters`, each entry is a **named** exporter (a YAML map), not a list item.

## Supported Dimensions

- **Efficiency**: QPS, TPS
- **Quality**: average recall, quantile recall (P0/P10/P50/P90...), and optional query-level recall
  coverage
- **Latency**: average, P50/P95/P99
- **Resource**: peak memory usage

### Query-level Recall Coverage

Set the optional `recall_target` YAML field or `--recall_target` CLI option to a finite value in
`[0, 1]`. The tool then reports the fraction of recorded query executions whose individual
`recall@k` reaches the target.

`recall_target` is supported only for `knn` and `knn_filter`, the search modes that currently
produce per-query `recall@k`. Configurations that combine it with `range` or `range_filter` are
rejected.

For a search with `topk: k`, reaching target `R` requires at least `ceil(R * k)` correct results.
The per-query result follows the same distance-based or ID-based recall semantics selected for the
evaluation. `query_count` counts executions in the recall-monitor pass, including repeated dataset
queries when `search_query_count` exceeds the dataset query count; it is not a count of distinct
query IDs.

JSON exporters add the following object:

```json
{
  "query_coverage": {
    "recall_target": 0.9,
    "reached_queries": 950,
    "query_count": 1000,
    "rate": 0.95
  }
}
```

If the target is omitted, `query_coverage` is omitted and existing output remains unchanged.

### Search Measurement Semantics

- **Latency** is the elapsed wall duration of each measured `Index::KnnSearch` call, measured
  with the monotonic `std::chrono::steady_clock`.
- **QPS** is the number of successful queries divided by the wall duration, in seconds, of the
  measured search pass.
- Statistics extraction, recall calculation, and memory sampling run outside the performance
  pass and do not contribute to latency or QPS.
- Every measured query is included, including the first query on each worker. The tool does not
  apply an implicit warm-up; run any desired warm-up explicitly before the measured pass.

JSON results include `measurement_method`, `measurement_sample_count`,
`measurement_successful_query_count`, and `measurement_duration(s)` so the measurement population
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

## Prepare calibration and validation inputs

`tools/eval/prepare_query_split.py` creates query subsets for calibrating parameters and
checking them on separate queries. It accepts the unfiltered dense HDF5 layout described in
[Dataset Format](dataset_format.md), with float32 or int8 vectors and the four datasets
`train`, `test`, `neighbors`, and `distances`. Like `eval_performance`, it accepts
`neighbors` stored as any integer type, such as the int32 IDs in ann-benchmarks files, as
long as every ID is a row of `train`. Sparse, multi-vector, and filtering datasets are
rejected.

Install NumPy and h5py in the Python environment used for the helper, then specify the
original query row indices for each subset. For example:

```bash
python3 -m pip install numpy h5py
cat > query_rows.json <<'JSON'
{
  "calibration": [0, 2, 4],
  "validation": [1, 3, 5]
}
JSON
python3 tools/eval/prepare_query_split.py \
  /path/to/sift-128-euclidean.hdf5 query_rows.json /tmp/sift-query-split
```

Choose representative query populations for the workload; the small lists above illustrate
the input format. Both lists must be non-empty. Repeated row indices and identical query
vectors across the two lists are rejected. Query comparison uses exact vector values
(treating positive and negative zero equally). Identical vectors within one list retain
their original row weighting, and unlisted query rows are omitted.

The new output directory contains `calibration.hdf5`, `validation.hdf5`, and
`query_rows.json` with the source path and row mapping. Selected query vectors and their
paired ground-truth rows keep the supplied order, types, and attributes. Each output links
to the original `/train` dataset through a relative HDF5 external link, preserving base row
IDs without copying the base vectors. Keep the source file unchanged and preserve this
relative layout when moving the files. Existing output directories are not overwritten.

Use `calibration.hdf5` as `datapath` while selecting search parameters, then use the frozen
parameters with `validation.hdf5`. An index built from the original base vectors can be
reused for both files.

## Python Usage (`pyvsag.eval`)

The same measurement core is available from Python through the `pyvsag` wheel. HDF5 and YAML
support are optional extras:

```bash
pip install "pyvsag[eval]"   # adds h5py and pyyaml
```

### Programmatic API

```python
import pyvsag
from pyvsag.eval import EvalDataset, evaluate_build, evaluate_search, num_threads_building

dataset = EvalDataset.from_hdf5("sift-128-euclidean.hdf5")      # or EvalDataset.from_arrays(...)
with num_threads_building(4):                                    # set before creating the index
    index = pyvsag.Index("hgraph", build_param_json)
    build = evaluate_build(index, dataset)
search = evaluate_search(index, dataset, top_k=10, search_param=search_param_json)
print(search["qps"], search["recall_avg"])
```

`evaluate_build` / `evaluate_search` return the same result dictionaries as the JSON produced by
`eval_performance`. `EvalDataset.with_ground_truth(k)` computes exact neighbours with numpy when
the dataset has none.

### Command line and YAML jobs

`pyvsag-eval` (or `python -m pyvsag.eval`) runs the YAML job format of `eval_performance`:

```bash
pyvsag-eval job.yaml
```

Supported are the global options and named cases, `build`, `search` and `build,search` actions,
and the `json`, `table`, `text` and `line_protocol` formats with `stdout`, `file://`,
`appendfile://` and `influxdb://` destinations. Notes:

- `build,search` keeps the index in memory instead of going through `index_path`.
- `http_server` is ignored with a warning, and unknown case keys produce a warning.
- Memory metrics read `/proc` and are therefore only meaningful on Linux (0 on macOS).
- `num_threads_building` must be set before the index is created; the job runner does this.

## References

- Source: `tools/eval/`
- Local tool entry point: `tools/eval/README.md`
- Reference numbers on standard hardware: [Reference Performance](performance.md).
- Runnable Python examples: `examples/python/333_feature_eval_api.py` (API) and `examples/python/334_feature_eval_job.py` (HDF5 + YAML job + CLI).
