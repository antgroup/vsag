# VSAG 性能评估工具 (Performance Evaluation Tool)

[\[EN\]](README.md)

这是一个功能强大的命令行工具，用于对 [VSAG](https://github.com/antgroup/vsag) 向量检索库进行全面的性能基准测试。它支持两种运行模式：

1.  **命令行模式**：通过命令行参数进行快速、单一的测试。
2.  **配置文件模式**：通过 YAML 配置文件进行复杂的、多案例的批量测试和结果导出。

## 主要功能

- **多模式评估**：支持索引构建 (`build`) 和向量搜索 (`search`) 两种评估类型。
- **多种搜索方式**：支持 KNN、Range Search，以及带过滤的 KNN 和 Range Search。
- **全面的性能指标**：
    - **效率**：QPS (每秒查询数), TPS (每秒处理向量数)
    - **效果**：平均召回率, 不同百分位的召回率 (P0, P10, P50, P90 等)
    - **延迟**：平均延迟, 不同百分位的延迟 (P50, P95, P99 等)
    - **资源**：峰值内存使用
- **灵活的配置**：支持通过命令行或 YAML 文件配置所有测试参数。
- **多种导出目标**：结果可输出到 stdout、写入文件或推送到 InfluxDB。
- **多种输出格式**：`table` / `text`、`json`、以及用于 InfluxDB 的 `line_protocol`。
- **可选的内置 HTTP 监控**：在批量评估运行期间，通过内嵌 HTTP 服务实时查看进度和指标。

## 编译与安装

`tools/` 默认不会被编译，需要显式开启。在编译之前，请确保您已安装 C++17 编译器、CMake 和 HDF5 库。

```bash
# 1. 克隆仓库
git clone https://github.com/antgroup/vsag.git
cd vsag

# 2. 开启 tools 选项编译
VSAG_ENABLE_TOOLS=ON make release
# 或者直接通过 CMake：
# cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
# cmake --build build-release -j

# 3. 编译完成后，可执行文件位于：
#    ./build-release/tools/eval/eval_performance
```

Ubuntu 安装 HDF5：`apt install libhdf5-dev`；CentOS：`yum install hdf5-devel`。

## 使用方法

该工具有两种主要的使用方式。

### 模式一：通过命令行参数

此模式适用于对单个索引配置进行快速测试。所有参数都通过命令行标志提供。

#### 主要参数

**基础参数**

- `-d, --datapath` (必需): 用于评估的 HDF5 数据集文件路径。
- `-t, --type` (必需): 评估方法，可选 `build` 或 `search`。
- `-n, --index_name` (必需): 要创建的索引名称 (例如 `hgraph`、`hnsw`、`ivf`)。
- `-c, --create_params` (必需): 创建索引所需的参数，格式为 JSON 字符串 (例如 `'{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}'`)。
- `-i, --index_path`: 索引的保存或加载路径 (默认: `/tmp/performance/index`)。

**搜索相关参数**

- `-s, --search_params`: 搜索时所需的参数，格式为 JSON 字符串 (例如 `'{"hgraph":{"ef_search":100}}'`)。当 `--type` 为 `search` 时此项为必需。
- `--search_mode`: 搜索模式，可选 `knn`、`range`、`knn_filter`、`range_filter` (默认: `knn`)。
- `--topk`: KNN 搜索的 K 值 (默认: `10`)。
- `--range`: Range 搜索的半径 (默认: `0.5`)。
- `--search-query-count`: 用于搜索性能评估的查询次数。当数据集查询数不足时会循环复用 (默认: `100000`)。
- `--delete-index-after-search`: 搜索完成后删除磁盘上的索引以释放存储空间 (默认: `false`)。

**指标控制参数 (用于禁用某些计算)**

- `--disable_recall`: 禁用平均召回率评估。
- `--disable_percent_recall`: 禁用百分位召回率评估 (P0, P10, P30, P50, P70, P90)。
- `--disable_qps`: 禁用 QPS 评估。
- `--disable_tps`: 禁用 TPS 评估。
- `--disable_memory`: 禁用峰值内存使用评估。
- `--disable_latency`: 禁用平均延迟评估。
- `--disable_percent_latency`: 禁用百分位延迟评估 (P50, P80, P90, P95, P99)。

#### 示例

1.  **构建索引**

    构建一个 HGraph 索引并保存到指定路径。

    ```bash
    ./eval_performance \
        --type build \
        --datapath /path/to/sift-128-euclidean.hdf5 \
        --index_name hgraph \
        --create_params '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}' \
        --index_path /tmp/my_sift_index
    ```

2.  **搜索向量**

    加载已构建的索引并执行 KNN 搜索。

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

### 模式二：通过 YAML 配置文件

此模式功能更强大，允许您在一个文件中定义和运行多个测试案例，并能灵活地将结果导出到不同格式和位置。

#### 运行方式

YAML 文件作为位置参数直接传入（不需要 `--config` 标志）：

```bash
./eval_performance /path/to/your/config.yaml
```

可参考模板 [`tools/eval/eval_template.yaml`](eval_template.yaml)。

#### YAML 文件结构

YAML 文件由一个可选的 `global` 部分和一个或多个具名的测试案例组成。除 `global` 外，每个顶级键都被视为一个独立的测试案例，案例名称可以自定义（例如 `hnsw_m16_ef200`）。

##### `global` 部分

```yaml
global:
  num_threads_building: 8        # 构建索引时使用的线程数
  num_threads_searching: 16      # 搜索时使用的线程数
  exporters:                     # 具名导出器组成的 map (注意不是数组)
    <exporter_name>:
      to: <destination>
      format: <format>
      vars:                      # 可选，依导出器类型而定
        <key>: <value>
  http_server:                   # 可选的 HTTP 监控服务
    enabled: true
    port: 8080
```

###### `exporters`

`exporters` 是一个**具名导出器的 map**（每个键只是一个标签）。每个导出器需要指定：

- `to`: 导出目标，可选：
    - `stdout` — 输出到标准输出。
    - `file://<path>` — 写入到指定文件（覆盖）。
    - `influxdb://<host>:<port>/<path>?<query>` — POST 到 InfluxDB v2 接口。`influxdb://` 前缀会被内部重写为 `http://`。需要 `format: line_protocol`，并在 `vars` 中提供 `token`（值需包含 `Token ` 前缀，例如 `Token <your-influxdb-token>`）。
- `format`: 导出格式，可选 `table`（或等价的 `text`）、`json`、`line_protocol`。
- `vars`（可选）: 导出器所需的额外变量，例如 InfluxDB 的 `token`。

如果没有配置任何 `exporters`，结果将默认以 `table` 格式打印到 stdout。

###### `http_server`

当 `http_server.enabled` 为 `true` 时，运行期间会在 `0.0.0.0:<port>`（默认 `8080`）上启动一个内嵌的 HTTP 服务，实时暴露当前进度（当前案例、总案例数、完成百分比）和最新指标，便于长时间批量评估时的状态观察。

##### 测试案例

每个具名案例支持的字段与命令行参数一致。常用字段：

- `datapath`（必需）、`type`（必需，`build` 或 `search`）、`index_name`（必需）、`create_params`（必需）
- `search_params`（当 `type: search` 时必需）、`search_mode`、`index_path`、`topk`、`range`
- `search_query_count`、`delete_index_after_search`
- `num_threads_building`、`num_threads_searching`（覆盖 `global` 中针对此案例的值）
- `disable_recall`、`disable_percent_recall`、`disable_qps`、`disable_tps`、`disable_memory`、`disable_latency`、`disable_percent_latency`

#### YAML 示例

下面的示例定义了两个搜索测试案例，并将结果同时以表格形式打印到 stdout、写入到 JSON 文件，以及以 line_protocol 格式推送到 InfluxDB。

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

# 测试案例 1: HGraph 索引
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

# 测试案例 2: 使用 sq8_uniform 量化的 HGraph
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

该工具根据 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) 授权。
