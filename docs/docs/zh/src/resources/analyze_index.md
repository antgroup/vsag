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
- **支持的索引类型**：工厂注册的全部索引，即 `HGraph`、`IVF`、`Pyramid`、`SINDI`、
  `SINDI_V2`、`SIMQ`、`BruteForce`、`WARP` 与 `LazyHGraph`。

该接口与 `Index::GetStats()` 互为补充：后者无需查询数据，只输出索引的静态结构指标。
对于基于图的索引，度分布、入口点质量、子索引召回率以及低召回热点节点等图健康度信息，
通过 `GetStats()` 而非 `AnalyzeIndexBySearch` 输出。

### 公共输出契约

每次成功的静态或动态分析都会返回 JSON 对象，并包含 `_analysis` 成员：

```json
{
  "_analysis": {
    "schema_version": 1,
    "index_type": "brute_force",
    "analysis_type": "search",
    "status": "partial",
    "sample_count": 10,
    "topk": 10,
    "consistency": "weak_snapshot",
    "skipped": { "recall_query": "ground truth was not requested" }
  }
}
```

为保持兼容，已有索引专属字段继续位于 JSON 根节点。全部适用指标均已计算时 `status` 为
`complete`；部分指标因依赖不足而位于 `skipped` 时为 `partial`；请求合法但无法计算任何指标
（例如对空索引执行搜索分析）时为 `not_applicable`。非法参数和执行失败仍然抛出异常。

`BruteForce` 与 `WARP` 提供精确搜索基线计数及耗时；`SINDI_V2` 与 `SIMQ` 提供基线计数和
查询耗时，并将尚无真值的专属质量指标标记为 skipped。`LazyHGraph` 委托当前 BruteForce 或
HGraph 后端，并追加 `phase` 与 `transition_threshold`；其 `index_type` 始终为 `lazy_hgraph`。

Analysis 是同步、面向离线诊断的操作，可能扫描大量索引数据，专用 Analyzer 还可能计算精确
真值。Analysis 不修改搜索或序列化配置，可与只读 Search 并发执行。除非索引提供更强的加锁
快照，否则 `_analysis.consistency` 为 `weak_snapshot`；不要让 Analysis 与 Add、Remove、Update
或 phase transition 并发执行。

Pyramid 的查询分析遵循与 `KnnSearch` 相同的路径限定语义。对于默认 hierarchy，
使用 `Dataset::Paths` 为每条 query 设置路径；对于命名 hierarchy，使用
`Dataset::Paths(hierarchy_name, paths)` 设置路径，并在 Pyramid 搜索参数中选择该
hierarchy。当选中 hierarchy 的根节点为 `NO_INDEX` 时必须提供路径。批量 query
数据集在需要或提供路径时，必须为每条 query 提供对应路径。Pyramid 只在每条 query 的
hierarchy 和路径所允许的向量中计算真值集，并排除已标记删除的向量。Pyramid 查询分析
目前只支持 KNN 请求，并使用 `SearchRequest` 中的 `query_`、`topk_` 和 `params_str_`；不支持
带过滤条件或 iterator 的请求。分析期间应保持索引稳定，不要并发执行 add 或 remove。

### `GetStats()` 输出的静态指标

#### HGraph 指标

| 指标 | 含义 |
| --- | --- |
| `total_count` | 索引中向量总数 |
| `deleted_count` | 被标记为删除的向量数 |
| `connect_components` | 邻近图中的连通分量数 |
| `maximal_component_size` | 最大连通分量大小 |
| `in_degree_distribution` | 图入度分布直方图 |
| `out_degree_distribution` | 图出度分布直方图 |
| `average_degree` | 有效节点的平均图度数 |
| `duplicate_ratio` | 数据集中重复向量比例 |
| `avg_distance_base` | 基础数据集采样向量的平均距离 |
| `recall_base` | 基础数据集采样向量的自召回率 |
| `time_cost_query` | 使用采样 base 向量作为查询时的平均耗时 |
| `proximity_recall_neighbor` | 邻居列表相对真实最近邻的召回率 |
| `quantization_bias_ratio` | 量化距离相对精确距离的偏差比率 |
| `quantization_inversion_count_rate` | 量化导致的距离顺序倒置比率 |
| `build_cache_hit_rate` | 上一次 `Build()` 中从导入缓存完成 warm-start 的节点占比；当索引并非由 `ImportCache()` 导入的缓存构建时，输出 `skipped_reason` |
| `build_cache_hit_nodes` / `build_cache_missed_nodes` | `build_cache_hit_rate` 背后的命中 / 未命中节点数（仅在索引由导入缓存构建时存在） |

#### Pyramid 指标

| 指标 | 含义 |
| --- | --- |
| `total_count` | 索引中的向量总数 |
| `index_node_structure` | 路径树节点数、深度、逐层节点数及节点状态分布 |
| `leaf_node_size_distribution` | 各叶节点向量数量的分布 |
| `subindex_quality` | 各叶子 GRAPH/FLAT 子索引的状态与规模，以及图/暴搜数量和向量覆盖率 |
| `recall_base` | 各图节点按规模加权的召回率；仅在采样和分析搜索参数可用时输出 |
| `graph_node_count` / `total_graph_size` | 参与召回分析的图节点数量及总规模 |
| `skipped_node_count` / `low_recall_nodes` | 未参与召回分析的节点数，以及低于召回阈值的节点详情 |
| `duplicate_ratio` | 当前 hierarchy 内的重复向量比例 |
| `hierarchy_count` / `hierarchies` | 多 hierarchy 数量及按名称组织的上述指标；单个匿名 hierarchy 直接在顶层输出 |

#### SINDI 指标

| 指标 | 含义 |
| --- | --- |
| `total_count` | 稀疏索引中的向量总数 |
| `window_count` | SINDI window 数量 |
| `active_term_count.mean` / `min` / `max` | 每个 window 中非空 term 数占 term capacity 的比例统计 |
| `active_term_count.avg_count` | 每个 window 的平均非空 term 数 |
| `posting_length_distribution.mean` / `max` / `p95` / `p99` | 非空 posting list 长度分布 |
| `posting_length_distribution.long_tail_threshold` | 作为长尾阈值的 P99 posting list 长度 |
| `posting_length_distribution.long_tail_mean` | 长度超过 P99 阈值的 posting list 比例 |
| `mean_doc_retained.mean` | doc prune 后每个文档平均保留的 term 比例 |
| `recall_base` | 使用采样 base 向量作为 query、基于精确 sparse 真值集计算的自召回 |
| `doc_prune_recall` | 禁用 query prune 时，doc-pruned 索引返回候选相对真值 top-k 的召回 |
| `doc_prune_bias_mean` | doc-pruned 距离相对原始精确 sparse 距离的平均相对偏差 |
| `doc_prune_inversion_count_rate` | doc prune 在候选集合内导致的距离顺序倒置比例 |
| `quantization_range.min_val` / `max_val` / `diff` | SQ8 量化范围，仅在开启量化时输出 |
| `quantization_recall` | 量化粗筛候选相对真值 top-k 的召回，仅在开启量化时输出 |
| `quantization_bias_ratio` | 量化距离相对解码后 doc-pruned 距离的平均相对偏差 |
| `quantization_inversion_count_rate` | 量化在候选集合内导致的距离顺序倒置比例 |

依赖原始 base 向量的 SINDI 指标在数据不可用时会输出 `skipped_reason`。当
`use_reorder=true` 时，索引内可读取原始向量；否则需要通过 analyze 参数或下方命令行参数
传入 SINDI `base_path`。

### `AnalyzeIndexBySearch` 输出的动态指标

#### HGraph 指标

| 指标 | 含义 |
| --- | --- |
| `recall_query` | 用户查询集相对真实最近邻的召回率 |
| `avg_distance_query` | 查询向量与检索结果之间的平均距离 |
| `time_cost_query` | 平均单次查询耗时，单位毫秒 |
| `quantization_bias_ratio_query` | 查询阶段观察到的量化距离偏差 |
| `quantization_inversion_count_rate_query` | 查询阶段量化导致的距离顺序倒置率 |

#### Pyramid 指标

| 指标 | 含义 |
| --- | --- |
| `recall_query` | 搜索结果相对每条 query 所选 hierarchy 和路径内精确真值集的召回率 |
| `avg_distance_query` | 路径范围内精确真值邻居的平均距离 |
| `time_cost_query` | 平均单次查询耗时，单位毫秒 |
| `quantization_error_query` | base 量化引入的查询距离误差；开启 reorder 时输出 |
| `quantization_inversion_ratio_query` | base 量化引入的查询顺序倒置率；开启 reorder 时输出 |

#### SINDI 指标

| 指标 | 含义 |
| --- | --- |
| `recall_query` | 搜索结果相对用户提供或自动生成 sparse 真值集的召回率 |
| `mean_latency_ms` | 调用 `KnnSearch` 时测得的平均单 query 耗时 |
| `time_cost_query` | `mean_latency_ms` 的别名，用于和其他 analyzer 保持输出习惯一致 |
| `postings_scanned.query_term_count_after_prune_mean` | query prune 后平均剩余 query term 数 |
| `postings_scanned.query_term_with_posting_mean` | 剩余 query term 中平均有多少 term 命中至少一个非空 posting list |
| `postings_scanned.posting_hit_mean` | 剩余 query term 命中非空 posting list 的平均比例 |
| `doc_prune_recall` | 禁用 query prune 时，doc-pruned 粗筛候选相对 sparse 真值集的召回 |
| `doc_prune_bias_mean` | 抽样 query 上 doc-pruned 距离相对原始精确 sparse 距离的平均相对偏差 |
| `doc_prune_inversion_count_rate` | 抽样 query 上 doc prune 在候选集合内导致的顺序倒置比例 |
| `quantization_recall` | 量化粗筛候选召回，仅在开启量化时输出 |
| `quantization_bias_ratio` | 量化距离相对解码后 doc-pruned 距离的平均相对偏差 |
| `quantization_inversion_count_rate` | 量化在候选集合内导致的顺序倒置比例 |
| `reorder_recall.before_reorder_recall_k_at_k` | 精排前粗筛 top-k 候选相对真值 top-k 的召回 |
| `reorder_recall.after_reorder_recall_k_at_k` | 精排后最终 top-k 相对真值 top-k 的召回 |
| `last_topk_rank_in_heap.mean` / `p95` / `p99` / `max` | 最终 top-k 结果在精排前候选堆中的最差名次分布 |

SINDI 动态召回和距离质量指标需要真值集。可通过 `groundtruth_path` 复用已有 `.dev.gt`，
也可通过 `base_path` 让 analyzer 基于原始 sparse base 生成精确真值集；`save_groundtruth_path`
可保存生成结果便于后续复用。没有可用真值集时，这些字段会输出 `skipped_reason`；
`postings_scanned` 只依赖 query 和索引 posting，仍可正常输出。

量化相关字段在不同索引下命名不一致：

| 索引 | 字段 | 含义 |
| --- | --- | --- |
| `HGraph` | `quantization_bias_ratio_query` | 搜索阶段观察到的量化偏差 |
| `HGraph` | `quantization_inversion_count_rate_query` | 搜索阶段量化引起的距离顺序倒置率 |
| `IVF` | `quantization_bias_ratio` | 搜索阶段观察到的量化偏差（仅在 `use_reorder_` 启用时输出） |
| `IVF` | `quantization_inversion_count_rate` | 搜索阶段量化引起的距离顺序倒置率（仅在 `use_reorder_` 启用时输出） |
| `Pyramid` | `quantization_error_query` | 搜索阶段的量化距离误差（仅在开启 reorder 时输出） |
| `Pyramid` | `quantization_inversion_ratio_query` | 搜索阶段量化引起的顺序倒置率（仅在开启 reorder 时输出） |

静态结构和健康度信息（包括度分布、入口点分析与 Pyramid 子索引质量）请查看
`GetStats()` 的 JSON 输出。

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
| `--query_path` | `-qp` | 否 | 查询向量数据集的文件路径，不是 Pyramid hierarchy 路径元数据。如果未提供，则只进行静态分析。 |
| `--query_data_type` | | 否 | 查询数据类型：`auto`、`dense` 或 `sparse`。`auto` 会对 SINDI 使用 sparse 加载。 |
| `--base_path` | | 否 | SINDI 分析可选的 sparse CSR 原始 base 数据集路径。 |
| `--groundtruth_path` | | 否 | SINDI 可选的 `.dev.gt` 真值集路径；提供后直接复用。 |
| `--save_groundtruth_path` | | 否 | SINDI 自动生成真值集时的可选保存路径。 |
| `--search_parameter` | `-sp` | 否 | 动态分析时使用的搜索参数（JSON）。 |
| `--topk` | `-k` | 否 | 动态分析的 top-K（默认 `100`）。 |

查询文件格式为 `tools/analyze_index/analyze_index.cpp` 中 `load_query()` 所读取的简单二进制
布局：`(uint32 rows, uint32 cols, float32 data...)`。

该 dense query 格式只包含向量。当前工具无法向加载后的 `Dataset` 附加默认或命名
Pyramid hierarchy 路径。因此，工具无法表达按路径限定的 Pyramid query，也无法在选中
hierarchy 的根节点为 `NO_INDEX` 时运行动态分析。通过 `GetStats()` 执行的 Pyramid 静态分析
仍可正常使用。如需按路径执行动态分析，请调用 C++ 接口并为 query 数据集设置路径。

SINDI 的 query/base 数据使用 CSR sparse 二进制布局：`int64 nrow, int64 ncol, int64 nnz`，
随后是 `int64 indptr[nrow + 1]`、`int32 indices[nnz]` 和 `float32 data[nnz]`。SINDI 真值集
使用 `.dev.gt` 布局：`uint32 query_count, uint32 topk`，随后是展开后的 `int32 ids` 与
`float32 distances`。如果没有提供 `--groundtruth_path` 但提供了 `--base_path`，SINDI 分析会
基于原始 sparse base 生成真值集，并可通过 `--save_groundtruth_path` 保存复用。

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

当序列化索引只内嵌 `index_param` 时，`analyze_index` 也可以在不提供 `--build_parameter` 的情况下
加载；缺失的 metadata 字段会尽可能使用 analyzer 默认值补齐。

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
- 本地工具入口：`tools/analyze_index/README_zh.md`
