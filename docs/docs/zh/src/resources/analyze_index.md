# 索引分析（`AnalyzeIndexBySearch` 与 `analyze_index`）

VSAG 提供了对**已构建或已加载索引**进行内省诊断的能力，可以在不重建索引的情况下排查召回率回归、
量化质量、图结构健康度以及查询性能问题。该能力通过两种方式对外暴露：

- C++ 接口 `Index::AnalyzeIndexBySearch`（声明在 `include/vsag/index.h`）；
- 命令行诊断工具 `analyze_index`，位于 `tools/analyze_index/`。

## `AnalyzeIndexBySearch` 接口

```cpp
// include/vsag/index.h
virtual std::string
AnalyzeIndexBySearch(const SearchRequest& request);
```

- **输入**：`SearchRequest`（查询数据集 + `topk` + 搜索参数 JSON）。
- **输出**：JSON 字符串，包含基于查询的动态指标。
- **支持的索引类型**：当前支持 `HGraph` 与 `IVF`。`Pyramid` 仅通过 `GetStats()` 提供静态分析，
  尚未 override `AnalyzeIndexBySearch`。未实现该接口的索引在调用时会抛出异常。

该接口与 `Index::GetStats()` 互为补充：后者无需查询数据，只输出索引的静态结构指标。
对于基于图的索引，度分布、入口点质量、子索引召回率以及低召回热点节点等图健康度信息，
通过 `GetStats()` 而非 `AnalyzeIndexBySearch` 输出。

### `GetStats()` 输出的静态指标

| 指标 | 含义 |
| --- | --- |
| `total_count` | 索引中向量总数 |
| `deleted_count` | 被标记为删除的向量数 |
| `connect_components` | 邻近图中的连通分量数 |
| `duplicate_ratio` | 数据集中重复向量比例 |
| `avg_distance_base` | 基础数据集采样向量的平均距离 |
| `recall_base` | 基础数据集采样向量的自召回率（健康度自检） |
| `proximity_recall_neighbor` | 邻居列表相对真实最近邻的召回率 |
| `quantization_bias_ratio` | 量化距离相对精确距离的偏差比率 |
| `quantization_inversion_count_rate` | 量化导致的距离顺序倒置比率 |

### `AnalyzeIndexBySearch` 输出的动态指标

HGraph 输出的通用查询指标（**IVF 当前不输出** 以下召回 / 距离 / 时延字段，详见下方说明）：

| 指标 | 含义 |
| --- | --- |
| `recall_query` | 用户查询集相对真实最近邻的召回率 |
| `avg_distance_query` | 查询向量与检索结果之间的平均距离 |
| `time_cost_query` | 平均单次查询耗时（毫秒） |

量化相关字段在不同索引下命名不一致：

| 索引 | 字段 | 含义 |
| --- | --- | --- |
| `HGraph` | `quantization_bias_ratio_query` | 搜索阶段观察到的量化偏差 |
| `HGraph` | `quantization_inversion_count_rate_query` | 搜索阶段量化引起的距离顺序倒置率 |
| `IVF` | `quantization_bias_ratio` | 搜索阶段观察到的量化偏差（仅在 `use_reorder_` 启用时输出） |
| `IVF` | `quantization_inversion_count_rate` | 搜索阶段量化引起的距离顺序倒置率（仅在 `use_reorder_` 启用时输出） |

如需度分布、入口点分析或子索引质量分布等图健康度信息，请查看 `GetStats()` 的 JSON 输出——
`AnalyzeIndexBySearch` 仅关注查询驱动的动态信号。

## `analyze_index` 工具

`analyze_index` 是上述分析接口的命令行封装。它从磁盘加载一个已序列化的 VSAG 索引，
打印元数据与 `GetStats()` 结果，并可选地针对查询文件运行 `AnalyzeIndexBySearch`。

### 构建

`tools/` 默认不会编译，需要显式开启：

```bash
# 通过项目 Makefile
VSAG_ENABLE_TOOLS=ON make release

# 也可直接通过 CMake
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release -j
# 产物：./build-release/tools/analyze_index/analyze_index
```

### 命令行参数

| 参数 | 缩写 | 是否必需 | 描述 |
| --- | --- | --- | --- |
| `--index_path` | `-i` | **是** | 待分析的 VSAG 索引文件路径。 |
| `--build_parameter` | `-bp` | 否 | 加载索引时使用的构建参数（JSON）。默认使用索引文件内嵌的原始参数。 |
| `--query_path` | `-qp` | 否 | 查询数据集路径。如果未提供，则只进行静态分析。 |
| `--search_parameter` | `-sp` | 否 | 动态分析时使用的搜索参数（JSON）。 |
| `--topk` | `-k` | 否 | 动态分析的 top-K（默认 `100`）。 |

查询文件格式为 `tools/analyze_index/analyze_index.cpp` 中 `load_query()` 所读取的简单二进制
布局：`(uint32 rows, uint32 cols, float32 data...)`。

### 两种分析模式

**1. 仅静态分析（不提供查询文件）**

```bash
./build-release/tools/analyze_index/analyze_index \
    --index_path /path/to/my_index.hgraph
```

输出索引名、维度、数据类型、距离度量、构建参数，以及 `GetStats()` 的 JSON。

**2. 静态 + 动态分析**

```bash
./build-release/tools/analyze_index/analyze_index \
    --index_path /path/to/my_index.ivf \
    --query_path /path/to/queries.bin \
    --search_parameter '{"ivf":{"scan_buckets_count":16}}' \
    --topk 50
```

除静态信息外，还会额外打印由 `AnalyzeIndexBySearch` 产出的 `Search Analyze: { ... }` JSON 块。

## 典型使用场景

- **召回率回归排查**：根据指标定位问题来源——是量化质量（`quantization_*`）、图结构
  （`connect_components`、`proximity_recall_neighbor`），还是查询端参数
  （对比 `recall_query` 与 `recall_base`）。
- **数据健康度体检**：发现重复数据（`duplicate_ratio`）、断连分量或过多删除等情况。
- **参数调优**：使用不同的 `search_parameter` 反复运行 `AnalyzeIndexBySearch`，
  在 `recall_query` 与 `time_cost_query` 之间选择合适的工作点，无需重建索引。
- **假设性实验**：通过 `--build_parameter` 在加载时覆盖原始构建参数，对未在文件中嵌入参数的索引
  进行不同配置的评估。

## 参考

- 接口：`include/vsag/index.h` 中的 `Index::AnalyzeIndexBySearch`
- 实现：`src/analyzer/{analyzer,hgraph_analyzer,pyramid_analyzer}.h`
- 工具源码：`tools/analyze_index/`
- 工具说明：`tools/analyze_index/README_zh.md`
