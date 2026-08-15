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
`--search-query-count`、`--delete-index-after-search`、`--set_immutable`，以及一系列用于关闭
单项指标的 `--disable_*` 开关。`--set_immutable` 默认是 `false`；启用后，评测器会在索引加载或
构建完成、搜索开始前调用 `Index::SetImmutable()`。索引不支持该操作时评测会失败。参考模板
`tools/eval/eval_template.yaml` 展示了完整的 YAML 结构。

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
  set_immutable: false
  topk: 10
```

注意：`global.exporters` 下每一项都是**具名**的导出器（即 YAML map），并不是数组。

## 并发 A/B 验收 Harness

需要对 baseline/candidate 做可复现的外层查询并发对比时，使用
`tools/eval/run_concurrency_ab.py`。它为每个变体分别调用独立的 evaluator binary，并逐个
case 运行，因此搜索计量语义仍由 `eval_performance` 决定；harness 负责固定输入参数，运行
并发 `1/2/4/8/16/32`，让每个并发档位的成对运行交替决定哪一个变体先执行，并把每次评估的
原始 JSON 写入一个报告。

先准备一个共享序列化索引和两个变体各自的 evaluator binary。harness 使用 `type: search`，
不会构建或修改索引：

```bash
python3 tools/eval/run_concurrency_ab.py \
    --spec tools/eval/concurrency_ab.example.json \
    --output /tmp/vsag-concurrency-ab.json \
    --hash-files
```

输入 spec 使用 JSON，因此可以在不额外安装 YAML 库的情况下生成和审查；完整格式见
`tools/eval/concurrency_ab.example.json`。`datapath`、`index_path`、`create_params`、
`search_params`、`topk` 和 `search_query_count` 在两个变体之间共享；两个变体必须分别提供
`eval_binary`。将 `rounds` 设为大于 1 可重复运行成对实验。变体可以显式覆盖 `index_path`
来进行两个索引的对比，但报告会给出 warning；推荐使用顶层共享 `index_path`。
spec 中的 `set_immutable` 也在两个变体之间共享，默认是 `false`。启用时两个 evaluator binary
都会在反序列化完成、搜索开始前调用 `Index::SetImmutable()`，因此启用只读搜索路径本身不会被算作
候选版本独有的改动。
`--hash-files` 会把数据集、两个索引和两个 evaluator binary 的 SHA-256 指纹写入报告，哈希
计算不在搜索计量阶段内，索引 hash 不同时会给出 warning。

`runs` 中每条记录包含以下验收字段：

```json
{
  "variant": "baseline",
  "pair_number": 1,
  "outer_concurrency": 1,
  "evaluator_binary": "/path/to/baseline/eval_performance",
  "index_path": "/path/to/shared.index",
  "errors": 0,
  "qps": 1234.5,
  "p50_ms": 0.81,
  "p99_ms": 1.42,
  "recall_avg": 0.99,
  "raw_result": {}
}
```

可以在 spec 中配置可选的 `acceptance` 性能验收门槛。默认值为
`qps_min_pct: 15`、`p99_max_regression_pct: 10`、`recall_max_abs_change: 0.01`，以及
`target_concurrency: [8, 16, 32]`：

```json
"acceptance": {
  "qps_min_pct": 15,
  "p99_max_regression_pct": 10,
  "recall_max_abs_change": 0.01,
  "target_concurrency": [8, 16, 32]
}
```

Harness 会按 `pair_number` 匹配同一轮、同一外层并发下的 baseline/candidate 运行，计算
`qps_change_pct = (candidate / baseline - 1) * 100`、同样定义的 `p99_change_pct`，以及
`recall_abs_change = abs(candidate - baseline)`。`acceptance.by_concurrency` 给出这些 paired
change 按并发取的中位数，并给出 `pass`、`fail` 或 `not_target` 判定；完整配对记录保存在
`acceptance.pairs` 中：

```json
{
  "status": "pass",
  "thresholds": {
    "qps_min_pct": 15.0,
    "p99_max_regression_pct": 10.0,
    "recall_max_abs_change": 0.01,
    "target_concurrency": [8, 16, 32]
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

`acceptance.status` 与 harness 执行层的 `status`、`errors` 相互独立。失败运行、缺少指标、配对不完整，
或百分比指标的 baseline 为 0 时，都会写入 `acceptance.reasons` 并使对应并发判定为 `fail`，不能静默排除。
健康的非目标并发显示为 `not_target`；但其中的无效配对仍显示为 `fail`，以暴露执行缺陷。
只有性能未达标时，执行层仍保持 `status: "ok", errors: 0`。

`outer_concurrency` 对应评测器的 `num_threads_searching` OpenMP 查询循环线程数。对比变体时，
应保持 `search_params` 中的 `parallel_search_thread_count` 不变。每个 YAML case 都会把 JSON
导出到临时文件；evaluator 的 stdout/stderr 只作为诊断文本保留，不会被当作结果解析。
`summary` 给出每个变体/并发组合在多轮 `rounds` 中各指标的中位数；`runs` 保留真实执行顺序，
并在 `raw_result` 下保留完整的单 case `eval_performance` JSON。

harness 的 `errors` 是**运行级**错误数（`error_scope` 为 `evaluator_invocation`）：评测进程成功时为 0，
超时、非零退出、启动失败或输出不是 JSON 时为 1。当前原生评测器遇到首个查询错误会直接退出，不会返回逐查询错误计数，
因此失败运行不能给出精确的失败查询数。成功运行的 `raw_result` 仍包含
`measurement_successful_query_count`，可用来检查两个变体是否完成了相同的有效查询数量。
`recall_avg` 是聚合召回率，不是逐查询结果 ID 对比；harness 也不会隐式添加预热轮次。

在 WSL 中运行 harness 的标准库测试：

```bash
python3 -m unittest discover -s tools/eval -p 'test_run_concurrency_ab.py' -v
```

## 支持的评估维度

- **效率**：QPS、TPS
- **效果**：平均召回率、分位召回率（P0/P10/P50/P90...）
- **延迟**：平均延迟、P50/P95/P99 延迟
- **资源**：峰值内存占用

### 搜索指标计量语义

- **延迟**是每次被测 `Index::KnnSearch` 调用的墙钟耗时，使用单调时钟
  `std::chrono::steady_clock` 测量。
- **QPS** 是成功查询数除以被测搜索阶段的墙钟时间（秒）。
- 统计信息提取、召回率计算和内存采样在性能测试阶段之外执行，不计入延迟
  或 QPS。
- 所有被测查询均计入结果，包括每个工作线程的首个查询。工具不会隐式预热；
  如有需要，应在被测阶段之前显式执行预热。

JSON 结果会输出 `measurement_method`、`measurement_sample_count`、
`measurement_successful_query_count` 和 `measurement_duration(s)`，用于核对测量方法
及其样本范围。

旧版本使用相邻监控回调间隔计算指标，其结果不能与采用上述语义的结果
直接比较。

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
- 本地工具入口：`tools/eval/README.md`
- 标准机型的基准结果见 [标准环境性能参考](performance.md)。
