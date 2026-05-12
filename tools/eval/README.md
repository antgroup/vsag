<!-- agent-hints
canonical: tools/eval/README.md
purpose: Usage guide for the eval_performance tool (build, run, datasets, metrics)
key-facts:
  - Binary built with ENABLE_TOOLS=ON; lives at build-release/tools/eval/eval_performance
  - Consumes HDF5 datasets in the ann-benchmarks layout (see docs/dataset_format.md)
  - Reports throughput, latency, and recall across indexes/params
related:
  - README_zh.md
  - ../../docs/docs/en/src/resources/eval.md
  - ../../docs/eval_performance.md
last-reviewed: 2026-05-12
-->
# VSAG Performance Evaluation Tool

[\[中文\]](README_zh.md)

This is a powerful command-line tool for comprehensive performance benchmarking of the [VSAG](https://github.com/antgroup/vsag) vector retrieval library. It supports two operational modes:

1.  **Command-Line Mode**: For quick, single-run tests using command-line arguments.
2.  **Configuration File Mode**: For complex, multi-case batch testing and result exportation using a YAML configuration file.

## Key Features

- **Multi-Mode Evaluation**: Supports both index `build` and vector `search` evaluation types.
- **Versatile Search Methods**: Supports KNN, Range Search, and their filtered counterparts (KNN with filter and Range Search with filter).
- **Comprehensive Performance Metrics**:
    - **Efficiency**: QPS (Queries Per Second), TPS (Throughput Per Second)
    - **Effectiveness**: Average Recall, Percentile Recall (P0, P10, P50, P90, etc.)
    - **Latency**: Average Latency, Percentile Latency (P50, P95, P99, etc.)
    - **Resources**: Peak Memory Usage
- **Flexible Configuration**: Supports configuring all test parameters via either command-line arguments or a YAML file.
- **Multiple Export Targets**: Results can be sent to stdout, written to a file, or pushed to InfluxDB.
- **Multiple Output Formats**: `table` / `text`, `json`, and `line_protocol` (for InfluxDB).
- **Built-in HTTP Monitor (optional)**: Expose live progress and metrics via an embedded HTTP server while a batch is running.

## Build and Installation

The `tools/` directory is not built by default. Enable it explicitly. Before building, ensure you have a C++17 compiler, CMake, and the HDF5 library installed.

```bash
# 1. Clone the repository
git clone https://github.com/antgroup/vsag.git
cd vsag

# 2. Build with tools enabled
VSAG_ENABLE_TOOLS=ON make release
# or directly via CMake:
# cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
# cmake --build build-release -j

# 3. After compilation, the executable is located at:
#    ./build-release/tools/eval/eval_performance
```

On Ubuntu install HDF5 with `apt install libhdf5-dev`; on CentOS use `yum install hdf5-devel`.

## Usage

The tool can be used in two main ways.

### Mode 1: Via Command-Line Arguments

This mode is suitable for running quick tests on a single index configuration. All parameters are provided through command-line flags.

#### Main Arguments

**Basic Parameters**

- `-d, --datapath` (required): The path to the HDF5 dataset file for evaluation.
- `-t, --type` (required): The evaluation method, choose from `build` or `search`.
- `-n, --index_name` (required): The name of the index to create (e.g., `hgraph`, `hnsw`, `ivf`).
- `-c, --create_params` (required): The parameters for creating the index, in JSON string format (e.g., `'{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}'`).
- `-i, --index_path`: The path to save or load the index (default: `/tmp/performance/index`).

**Search-Related Parameters**

- `-s, --search_params`: The parameters for searching, in JSON string format (e.g., `'{"hgraph":{"ef_search":100}}'`). This is required when `--type` is `search`.
- `--search_mode`: The search mode, choose from `knn`, `range`, `knn_filter`, `range_filter` (default: `knn`).
- `--topk`: The K value for KNN search (default: `10`).
- `--range`: The radius for Range search (default: `0.5`).
- `--search-query-count`: The number of queries to run for the search performance evaluation. Queries from the dataset are reused/cycled to reach this count (default: `100000`).
- `--delete-index-after-search`: Delete the on-disk index after the search is complete to free up storage space (default: `false`).

**Metric Control Parameters (for disabling certain calculations)**

- `--disable_recall`: Disable average recall evaluation.
- `--disable_percent_recall`: Disable percentile recall evaluation (P0, P10, P30, P50, P70, P90).
- `--disable_qps`: Disable QPS evaluation.
- `--disable_tps`: Disable TPS evaluation.
- `--disable_memory`: Disable peak memory usage evaluation.
- `--disable_latency`: Disable average latency evaluation.
- `--disable_percent_latency`: Disable percentile latency evaluation (P50, P80, P90, P95, P99).

#### Examples

1.  **Build an Index**

    Build an HGraph index and save it to the specified path.

    ```bash
    ./eval_performance \
        --type build \
        --datapath /path/to/sift-128-euclidean.hdf5 \
        --index_name hgraph \
        --create_params '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}' \
        --index_path /tmp/my_sift_index
    ```

2.  **Search Vectors**

    Load a pre-built index and perform a KNN search.

    ```bash
    ./eval_performance \
        --type search \
        --datapath /path/to/sift-128-euclidean.hdf5 \
        --index_name hgraph \
        --create_params '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}' \
        --index_path /tmp/my_sift_index \
        --search_params '{"hgraph":{"ef_search":100}}' \
        --topk 10
    ```

### Mode 2: Via YAML Configuration File

This mode is more powerful, allowing you to define and run multiple test cases in a single file and flexibly export the results to different formats and destinations.

#### How to Run

The YAML file path is passed directly as a positional argument (no `--config` flag):

```bash
./eval_performance /path/to/your/config.yaml
```

A reference template is available at [`tools/eval/eval_template.yaml`](eval_template.yaml).

#### YAML File Structure

A YAML file consists of an optional `global` section and one or more named test cases. Every top-level key other than `global` is treated as an independent test case; the case name is user-defined (e.g., `hnsw_m16_ef200`).

##### Global section

```yaml
global:
  num_threads_building: 8        # threads used for building indexes
  num_threads_searching: 16      # threads used for searching
  exporters:                     # map of named exporters (NOT a list)
    <exporter_name>:
      to: <destination>
      format: <format>
      vars:                      # optional, depends on exporter
        <key>: <value>
  http_server:                   # optional HTTP monitor
    enabled: true
    port: 8080
```

###### `exporters`

`exporters` is a **map** of named exporters (each key is just a label). For each exporter you specify:

- `to`: the export destination, one of:
    - `stdout` — print to standard output.
    - `file://<path>` — write (overwrite) to the given file path.
    - `influxdb://<host>:<port>/<path>?<query>` — POST to an InfluxDB v2 endpoint. The `influxdb://` prefix is internally rewritten to `http://`. Requires `format: line_protocol` and a `token` entry in `vars` (the value must include the `Token ` prefix, e.g. `Token <your-influxdb-token>`).
- `format`: the result format, one of `table` (or `text`, equivalent), `json`, `line_protocol`.
- `vars` (optional): additional variables required by the exporter, e.g. `token` for InfluxDB.

If no `exporters` are configured, results are printed to stdout in `table` format by default.

###### `http_server`

When `http_server.enabled` is `true`, an embedded HTTP server is started on `0.0.0.0:<port>` (default `8080`) for the duration of the run. It exposes live progress (current case, total cases, completion %) and the latest metrics, which is useful for long-running batch evaluations.

##### Test cases

Each named case accepts the same fields as the command-line parameters. Common fields:

- `datapath` (required), `type` (required: `build` or `search`), `index_name` (required), `create_params` (required)
- `search_params` (required when `type: search`), `search_mode`, `index_path`, `topk`, `range`
- `search_query_count`, `delete_index_after_search`
- `num_threads_building`, `num_threads_searching` (override the `global` values for this case)
- `disable_recall`, `disable_percent_recall`, `disable_qps`, `disable_tps`, `disable_memory`, `disable_latency`, `disable_percent_latency`

#### YAML Example

The example below defines two search test cases. Results are printed as a table to stdout, also written to a JSON file, and finally pushed to InfluxDB in line-protocol format.

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
    send-to-influxdb:
      to: "influxdb://127.0.0.1:8086/api/v2/write?org=vsag&bucket=eval&precision=ns"
      format: line_protocol
      vars:
        token: "Token <your-influxdb-token>"
  http_server:
    enabled: true
    port: 8080

# Test Case 1: HGraph index
hgraph_fp32_d32_ef300:
  datapath: /path/to/sift-128-euclidean.hdf5
  type: search
  index_name: hgraph
  create_params: '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}'
  search_params: '{"hgraph":{"ef_search":60}}'
  index_path: /tmp/vsag_eval/hgraph_fp32
  search_mode: knn
  topk: 10
  delete_index_after_search: false

# Test Case 2: HGraph with sq8_uniform quantization
hgraph_sq8_d32_ef400:
  datapath: /path/to/sift-128-euclidean.hdf5
  type: search
  index_name: hgraph
  create_params: '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"sq8_uniform","max_degree":32,"ef_construction":400}}'
  search_params: '{"hgraph":{"ef_search":100}}'
  index_path: /tmp/vsag_eval/hgraph_sq8
  topk: 10
```

## License

This tool is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
