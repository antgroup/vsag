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
`--search-query-count`, `--delete-index-after-search`, and the various `--disable_*` switches that
turn off individual metrics. The reference template at `tools/eval/eval_template.yaml` shows the
complete YAML shape.

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
```

Note: under `global.exporters`, each entry is a **named** exporter (a YAML map), not a list item.

## Supported Dimensions

- **Efficiency**: QPS, TPS
- **Quality**: average recall and quantile recall (P0/P10/P50/P90...)
- **Latency**: average, P50/P95/P99
- **Resource**: peak memory usage

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
