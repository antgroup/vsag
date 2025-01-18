The document introduces how to use the performance tool `eval_performance` for vsag index evaluate.

`eval_performance` is compiled when use `ENABLE_TOOLS=ON` option.

## 1. How To Run

we have to method to run the `eval_performance` tool

### 1.1 Use Command Line Options

example like this:

```shell
eval_performance --datapath /data/sift-128-euclidean.hdf5 
                 --type search
                 --index_name hgraph 
                 --create_params '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"sq8_uniform","max_degree":32,"ef_construction":400}}' 
                 --search_params '{"hgraph":{"ef_search":100}}' 
                 --index_path /data/sift-128-euclidean/index/hgraph_sq8_uniform_degree_32 
                 --topk 10
```

the detail options for command line as following:

```bash
Usage: eval_performance [--help] [--version] --datapath VAR --type VAR --index_name VAR --create_params VAR
 [--index_path VAR] [--search_params VAR] [--search_mode VAR] [--topk VAR] [--range VAR] 
 [--disable_recall VAR] [--disable_percent_recall VAR] [--disable_qps VAR] [--disable_tps VAR] 
 [--disable_memory VAR] [--disable_latency VAR] [--disable_percent_latency VAR]

Optional arguments:
  -h, --help                 shows help message and exits 
  -v, --version              prints version information and exits 
  -d, --datapath             The hdf5 file path for eval [required]
  -t, --type                 The eval method to select, choose from {"build", "search"} [required]
  -n, --index_name           The name of index fot create index [required]
  -c, --create_params        The param for create index [required]
  -i, --index_path           The index path for load or save [nargs=0..1] [default: "/tmp/performance/index"]
  -s, --search_params        The param for search [nargs=0..1] [default: ""]
  --search_mode              The mode supported while use 'search' type, 
                             choose from {"knn", "range", "knn_filter", "range_filter"}
                             [nargs=0..1] [default: "knn"]              
  --topk                     The topk value for knn search or knn_filter search [nargs=0..1] [default: 10]
  --range                    The range value for range search or range_filter search [nargs=0..1] [default: 0.5]
  --disable_recall           Disable average recall eval [nargs=0..1] [default: false]
  --disable_percent_recall   Disable percent recall eval, include p0, p10, p30, p50, p70, p90 [nargs=0..1] [default: false]
  --disable_qps              Disable qps eval [nargs=0..1] [default: false]
  --disable_tps              Disable tps eval [nargs=0..1] [default: false]
  --disable_memory           Disable memory eval [nargs=0..1] [default: false]
  --disable_latency          Disable average latency eval [nargs=0..1] [default: false]
  --disable_percent_latency  Disable percent latency eval, include p50, p80, p90, p95, p99 [nargs=0..1] [default: false]
```

### 1.2 Use Yaml Config File

use like this:

```shell
eval_performance run.yaml
```

the template yaml config as following, eval_case name can be modified

```yaml
eval_case1:
  datapath: "/data/sift-128-euclidean.hdf5"
  type: "search" # build, search
  index_name: "hgraph"
  create_params: '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":400}}'
  search_params: '{"hgraph":{"ef_search":60}}'
  index_path: "/data/sift-128-euclidean/index/hgraph_index"
  topk: 10
  search_mode: "knn" # ["knn", "range", "knn_filter", "range_filter"]
  range: 0.5

eval_case2:

```

actually, the yaml config is just like how to config on command line. the optional config 
is same on both runner methods

## 2. The Meaning Of The Output

After running the `eval_performance`. We will get the result on console. 
Different config has different output.

### 2.1 Build Output
when use `build` type for evaluate, output as following

```json5
{
    "action": "build",
    "dataset_info": {
        "base_count": 1000000,
        "data_type": "float32",
        "dim": 128,
        "filepath": "/data/sift-128-euclidean.hdf5",
        "query_count": 10000
    },
    "duration(s)": 42.376579977,
    "index_info": {
        "dim": 128,
        "dtype": "float32",
        "index_param": {
            "base_quantization_type": "fp32",
            "ef_construction": 400,
            "max_degree": 32
        },
        "metric_type": "l2"
    },
    "memory_peak(KB)": 1795280,
    "tps": 23597.940195805153
}
```

### 2.2 Search Output
when use `search` type for evaluate, output as following

```json5
{
  "action": "search",
  "dataset_info": {
    "base_count": 1000000,
    "data_type": "float32",
    "dim": 128,
    "filepath": "/data/sift-128-euclidean.hdf5",
    "query_count": 10000
  },
  "index_info": {
    "dim": 128,
    "dtype": "float32",
    "hnsw": {
      "ef_construction": 400,
      "max_degree": 16
    },
    "metric_type": "l2"
  },
  "latency_avg(ms)": 0.4475634817999995,
  "latency_detail(ms)": {
    "p50": 0.451057,
    "p80": 0.493573,
    "p90": 0.513236,
    "p95": 0.528784,
    "p99": 0.564937
  },
  "memory_peak(KB)": 1310664,
  "qps": 2234.319913631526,
  "recall_avg": 0.9842299999999834,
  "recall_detail": {
    "p0": 0.5,
    "p10": 0.9,
    "p30": 1.0,
    "p50": 1.0,
    "p70": 1.0,
    "p90": 1.0
  },
  "search_mode": "knn"
}

```

