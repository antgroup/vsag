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
- **强大的结果导出**：支持将测试结果格式化为表格 (Table)、JSON 或 Markdown，并导出到标准输出 (stdout) 或文件。

## 编译与安装

在编译之前，请确保您已安装 C++17 编译器、CMake 和 HDF5 库。

```bash
# 1. 克隆仓库 (假设此工具在 vsag 项目的子目录中)
git clone https://github.com/antgroup/vsag.git
cd vsag

# 2. 创建构建目录并编译
make release

# 3. 编译完成后，可执行文件位于 build-release/tools/eval/ 目录下
#    ./build-release/tools/eval/eval_performance
```

## 使用方法

该工具有两种主要的使用方式。

### 模式一：通过命令行参数

此模式适用于对单个索引配置进行快速测试。所有参数都通过命令行标志提供。

#### 主要参数

**基础参数**

- `-d, --datapath` (必需): 用于评估的 HDF5 数据集文件路径。
- `-t, --type` (必需): 评估方法，可选 `build` 或 `search`。
- `-n, --index_name` (必需): 要创建的索引名称 (例如, `hnsw`, `ivf-flat`)。
- `-c, --create_params` (必需): 创建索引所需的参数，格式为 JSON 字符串 (例如, `'{"M":32,"ef_construction":200}'`)。
- `-i, --index_path`: 索引的保存或加载路径 (默认: `/tmp/performance/index`)。

**搜索相关参数**

- `-s, --search_params`: 搜索时所需的参数，格式为 JSON 字符串 (例如, `'{"ef_search":100}'`)。当 `--type` 为 `search` 时此项为必需。
- `--search_mode`: 搜索模式，可选 `knn`, `range`, `knn_filter`, `range_filter` (默认: `knn`)。
- `--topk`: KNN 搜索的 K 值 (默认: 10)。
- `--range`: Range 搜索的半径 (默认: 0.5)。
- `--search-query-count`: 用于搜索性能评估的查询数量 (默认: 100000)。
- `--delete-index-after-search`: 搜索完成后删除索引 (默认: false)。

**指标控制参数 (用于禁用某些计算)**

- `--disable_recall`: 禁用平均召回率评估。
- `--disable_percent_recall`: 禁用百分位召回率评估。
- `--disable_qps`: 禁用 QPS 评估。
- `--disable_tps`: 禁用 TPS 评估。
- `--disable_memory`: 禁用内存使用评估。
- `--disable_latency`: 禁用平均延迟评估。
- `--disable_percent_latency`: 禁用百分位延迟评估。

#### 示例

1.  **构建索引**
    构建一个 HNSW 索引并将它保存到指定路径。

    ```bash
    ./eval_performance \
        --type build \
        --datapath /path/to/sift-1m.hdf5 \
        --index_name hnsw \
        --create_params '{"M":32,"ef_construction":200}' \
        --index_path /tmp/my_sift_index
    ```

2.  **搜索向量**
    加载已构建的索引，并使用 `ef_search=100` 的参数进行 KNN 搜索。

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

### 模式二：通过 YAML 配置文件

此模式功能更强大，允许您在一个文件中定义和运行多个测试案例，并能灵活地将结果导出到不同格式和位置。

#### 运行方式

```bash
./eval_performance /path/to/your/config.yaml
```

#### YAML 文件结构

YAML 文件由一个可选的 `global` 部分和多个测试案例 (case) 组成。

- **`global`**: 用于定义全局配置。
    - `num_threads_building`: 构建索引时使用的线程数。
    - `num_threads_searching`: 搜索时使用的线程数。
    - `exporters`: 定义结果如何导出。这是一个列表，可以配置多个导出器。
        - `to`: 导出目标，可选 `stdout` (屏幕) 或 `file`。
        - `format`: 导出格式，可选 `table` (默认), `json`, `markdown`。
        - `vars`: 导出器所需的额外变量。例如，当 `to` 是 `file` 时，需要提供 `path` 变量指定文件路径。

- **测试案例**: 除了 `global` 之外的每个顶级键都代表一个独立的测试案例。案例名称可以自定义（例如 `hnsw_m16_ef200`）。其下的参数与命令行参数基本一致。

#### YAML 示例

下面是一个示例 `config.yaml`，它定义了两个不同的搜索测试案例，并将结果同时以表格形式打印到屏幕、以 Markdown 和 JSON 格式保存到文件。

```yaml
# 全局配置
global:
  num_threads_building: 8
  num_threads_searching: 16
  exporters:
    # 导出器1: 在屏幕上打印表格
    - to: stdout
      format: table
    # 导出器2: 将结果保存为 Markdown 文件
    - to: file
      format: markdown
      vars:
        path: "./results.md"
    # 导出器3: 将结果保存为 JSON 文件
    - to: file
      format: json
      vars:
        path: "./results.json"

# 测试案例 1: HNSW 索引, M=16, ef_search=100
hnsw_m16_ef100:
  datapath: /path/to/sift-1m.hdf5
  type: search
  index_name: hnsw
  create_params: '{"M":16,"ef_construction":200}'
  search_params: '{"ef_search":100}'
  index_path: /tmp/vsag_eval/hnsw_m16
  topk: 10

# 测试案例 2: IVF-FLAT 索引, nlist=128, nprobe=8
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

该工具根据 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) 授权。
