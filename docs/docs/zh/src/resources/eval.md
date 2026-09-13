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
`--disable_*` 开关。参考模板 `tools/eval/eval_template.yaml` 展示了完整的 YAML 结构。

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

## 准备校准与验证输入

`tools/eval/prepare_query_split.py` 用于准备参数校准和独立查询验证所需的输入。
它支持[数据集格式](dataset_format.md)中不带过滤条件的稠密 HDF5：向量为 float32 或 int8，
包含 `train`、`test`、`neighbors` 和 `distances` 四个数据集。
稀疏、多向量及带过滤字段的数据集会被拒绝。

在运行工具的 Python 环境中安装 NumPy 和 h5py，再指定两个子集使用的原始查询行号。例如：

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

实际使用时应选择能代表工作负载的查询，上面的短列表只演示输入格式。两个列表均须非空。
工具拒绝重复行号及跨组的相同查询向量；查询比较采用精确向量值，正零与负零视为相同。
同一组内不同原始行上的相同向量保留其行权重，未列出的查询行不进入输出。

新输出目录包含 `calibration.hdf5`、`validation.hdf5`，以及记录源文件路径和行号映射的
`query_rows.json`。查询向量与对应真值按指定顺序复制，保留类型和属性。两个 HDF5 文件
通过相对外部链接引用原文件的 `/train`，保留库向量行号且不重复复制库向量。
源文件应保持不变，移动时保留这些文件的相对位置。工具不会覆盖已有输出目录。

选参时将 `calibration.hdf5` 作为 `datapath`，冻结参数后再使用 `validation.hdf5` 验证。
使用原库向量构建的索引可以供两个文件复用。

## 参考

- 源码：`tools/eval/`
- 本地工具入口：`tools/eval/README.md`
- 标准机型的基准结果见 [标准环境性能参考](performance.md)。
