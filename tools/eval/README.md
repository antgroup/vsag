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
- **Powerful Result Export**: Supports formatting test results as a table, JSON, or Markdown, and exporting them to standard output (stdout) or a file.

## Build and Installation

Before building, ensure you have a C++17 compiler, CMake, and the HDF5 library installed.

```bash
# 1. Clone the repository (assuming this tool is in a subdirectory of the vsag project)
git clone https://github.com/antgroup/vsag.git
cd vsag

# 2. Create a build directory and compile
make release

# 3. After compilation, the executable is located in the build-release/tools/eval/ directory
#    ./build-release/tools/eval/eval_performance
```

## Usage

The tool can be used in two main ways.

### Mode 1: Via Command-Line Arguments

This mode is suitable for running quick tests on a single index configuration. All parameters are provided through command-line flags.

#### Main Arguments

**Basic Parameters**

- `-d, --datapath` (required): The path to the HDF5 dataset file for evaluation.
- `-t, --type` (required): The evaluation method, choose from `build` or `search`.
- `-n, --index_name` (required): The name of the index to create (e.g., `hnsw`, `ivf-flat`).
- `-c, --create_params` (required): The parameters for creating the index, in JSON string format (e.g., `'{"M":32,"ef_construction":200}'`).
- `-i, --index_path`: The path to save or load the index (default: `/tmp/performance/index`).

**Search-Related Parameters**

- `-s, --search_params`: The parameters for searching, in JSON string format (e.g., `'{"ef_search":100}'`). This is required when `--type` is `search`.
- `--search_mode`: The search mode, choose from `knn`, `range`, `knn_filter`, `range_filter` (default: `knn`).
- `--topk`: The K value for KNN search (default: 10).
- `--range`: The radius for Range search (default: 0.5).
- `--search-query-count`: The number of queries to run for search performance evaluation (default: 100000).
- `--delete-index-after-search`: Delete the index after the search is complete (default: false).

**Metric Control Parameters (for disabling certain calculations)**

- `--disable_recall`: Disable average recall evaluation.
- `--disable_percent_recall`: Disable percentile recall evaluation.
- `--disable_qps`: Disable QPS evaluation.
- `--disable_tps`: Disable TPS evaluation.
- `--disable_memory`: Disable memory usage evaluation.
- `--disable_latency`: Disable average latency evaluation.
- `--disable_percent_latency`: Disable percentile latency evaluation.

#### Examples

1.  **Build an Index**
    Build an HNSW index and save it to the specified path.

    ```bash
    ./eval_performance \
        --type build \
        --datapath /path/to/sift-1m.hdf5 \
        --index_name hnsw \
        --create_params '{"M":32,"ef_construction":200}' \
        --index_path /tmp/my_sift_index
    ```

2.  **Search Vectors**
    Load a pre-built index and perform a KNN search with `ef_search=100`.

    ```bash
    ./eval_performance \
        --type search \
        --datapath /path/to/sift-1m.hdf5 \
        --index_name hnsw \
        --create_params '{"M":32,"ef_construction":200}' \
        --index_path /tmp/my_sift_index \
        --search_params '{"ef_search":100}' \
        --topk 10
    ```

### Mode 2: Via YAML Configuration File

This mode is more powerful, allowing you to define and run multiple test cases in a single file and flexibly export the results to different formats and destinations.

#### How to Run

```bash
./eval_performance /path/to/your/config.yaml
```

#### YAML File Structure

The YAML file consists of an optional `global` section and multiple test cases.

- **`global`**: Defines global configurations.
    - `num_threads_building`: Number of threads to use for building the index.
    - `num_threads_searching`: Number of threads to use for searching.
    - `exporters`: Defines how to export results. This is a list that can contain multiple exporters.
        - `to`: The export destination, choose from `stdout` or `file`.
        - `format`: The export format, choose from `table` (default), `json`, `markdown`.
        - `vars`: Additional variables required by the exporter. For example, when `to` is `file`, the `path` variable is required to specify the file path.

- **Test Cases**: Every top-level key other than `global` represents an independent test case. The case name is user-defined (e.g., `hnsw_m16_ef200`). The parameters under each case are mostly identical to the command-line arguments.

#### YAML Example

Below is an example `config.yaml` that defines two different search test cases and exports the results to the screen as a table, and to files in Markdown and JSON formats.

```yaml
# Global configuration
global:
  num_threads_building: 8
  num_threads_searching: 16
  exporters:
    # Exporter 1: Print results as a table to standard output
    - to: stdout
      format: table
    # Exporter 2: Save results as a Markdown file
    - to: file
      format: markdown
      vars:
        path: "./results.md"
    # Exporter 3: Save results as a JSON file
    - to: file
      format: json
      vars:
        path: "./results.json"

# Test Case 1: HNSW index, M=16, ef_search=100
hnsw_m16_ef100:
  datapath: /path/to/sift-1m.hdf5
  type: search
  index_name: hnsw
  create_params: '{"M":16,"ef_construction":200}'
  search_params: '{"ef_search":100}'
  index_path: /tmp/vsag_eval/hnsw_m16
  topk: 10

# Test Case 2: IVF-FLAT index, nlist=128, nprobe=8
ivf_nlist128_nprobe8:
  datapath: /path/to/sift-1m.hdf5
  type: search
  index_name: ivf-flat
  create_params: '{"nlist":128}'
  search_params: '{"nprobe":8}'
  index_path: /tmp/vsag_eval/ivf_128
  topk: 10
```

## License

This tool is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
