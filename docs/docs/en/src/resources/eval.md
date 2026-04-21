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

### 2. Config-file mode (batch comparisons)

```bash
./build-release/tools/eval/eval_performance --config my_eval.yaml
```

See the template `tools/eval/eval_template.yaml`. A single configuration can define multiple
`eval_caseN` entries, each with its own parameter set.

## Supported Dimensions

- **Efficiency**: QPS, TPS
- **Quality**: average recall and quantile recall (P0/P10/P50/P90...)
- **Latency**: average, P50/P95/P99
- **Resource**: peak memory usage

## Search Modes

`search_mode` accepts `knn`, `range`, `knn_filter`, and `range_filter`.

## Output Formats

- `table` — to stdout or a file.
- `json` — for downstream automation.
- `line_protocol` — directly to InfluxDB (supports token auth).
- `markdown` — handy when posting into an issue or document.

## Datasets

Any HDF5 dataset from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
(e.g. `sift-128-euclidean.hdf5`, `gist-960-euclidean.hdf5`) works out of the box.

## References

- Source: `tools/eval/`
- English README: `tools/eval/README.md`
- Reference numbers on standard hardware: [Reference Performance](performance.md).
