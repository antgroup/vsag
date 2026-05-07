# 性能评估工具（eval_performance）

`eval_performance` 是 VSAG 自带的命令行性能评估工具，位于 `tools/eval/`，编译后二进制路径为
`build-release/tools/eval/eval_performance`。它可以用于对比不同索引、不同参数组合的吞吐、延迟与召回率。

## 构建

`tools/` 默认不会编译，需要显式开启：

```bash
# 通过项目 Makefile
VSAG_ENABLE_TOOLS=ON make release
# 或：make dev

# 也可直接通过 CMake
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release -j
# 产物：./build-release/tools/eval/eval_performance
```

需要系统安装 HDF5（Ubuntu: `apt install libhdf5-dev`；CentOS: `yum install hdf5-devel`）。

## 两种模式

### 1. 命令行模式（适合单次快速测试）

```bash
./build-release/tools/eval/eval_performance \
    --datapath /tmp/sift-128-euclidean.hdf5 \
    --index_name hgraph \
    --type search \
    --create_params '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}' \
    --search_params '{"hgraph":{"ef_search":60}}' \
    --topk 10
```

常用参数还包括 `--search_mode`（`knn` / `range` / `knn_filter` / `range_filter`）、
`--search-query-count`、`--delete-index-after-search`，以及一系列用于关闭单项指标的
`--disable_*` 开关。完整参数列表见 `tools/eval/README.md`。

### 2. 配置文件模式（适合批量对比）

YAML 文件作为位置参数直接传入（不需要 `--config` 标志）：

```bash
./build-release/tools/eval/eval_performance my_eval.yaml
```

参考模板 `tools/eval/eval_template.yaml`。一份配置可以包含多个具名 case，并通过可选的
`global` 段配置共享参数，例如线程数、导出器以及内嵌的 HTTP 监控服务。

最小示例：

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

注意：`global.exporters` 下每一项都是**具名**的导出器（即 YAML map），并不是数组。

## 支持的评估维度

- **效率**：QPS、TPS
- **效果**：平均召回率、分位召回率（P0/P10/P50/P90...）
- **延迟**：平均延迟、P50/P95/P99 延迟
- **资源**：峰值内存占用

## 搜索模式

`search_mode` 支持 `knn`、`range`、`knn_filter`、`range_filter` 四种。

## 输出格式与导出目标

每个导出器同时指定一种 `format` 与一个 `to` 目标。

- 格式：`table`（或别名 `text`）、`json`、`line_protocol`（用于 InfluxDB）。
- 目标：
    - `stdout` — 输出到标准输出。
    - `file://<path>` — 写入文件（覆盖）。
    - `influxdb://<host>:<port>/<path>?<query>` — POST 到 InfluxDB v2 接口；
      需要使用 `format: line_protocol`，并通过 `vars.token` 传入鉴权令牌
      （值需包含 `Token ` 前缀，例如 `Token <your-influxdb-token>`）。

如未配置任何导出器，结果默认以 `table` 格式打印到 stdout。

## HTTP 监控（可选）

启用后，工具会在批量评估运行期间启动一个内嵌 HTTP 服务，实时暴露当前进度（当前案例、
总案例数、完成百分比）和最新指标，便于长时间任务的状态观察。

```yaml
global:
  http_server:
    enabled: true
    port: 8080
```

## 数据集

可使用 [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) 提供的 HDF5 格式数据集
（如 `sift-128-euclidean.hdf5`、`gist-960-euclidean.hdf5`）。

## 参考

- 源码：`tools/eval/`
- 详细说明：`tools/eval/README.md`
- 标准机型的基准结果见 [标准环境性能参考](performance.md)。
