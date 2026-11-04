# 索引分析（`AnalyzeIndexBySearch` 与 `analyze_index`）

VSAG 提供了对**已构建或已加载索引**进行内省诊断的能力，可以在不重建索引的情况下排查召回率回归、
量化质量、图结构健康度以及查询性能问题。该能力通过以下方式对外暴露：

- C++ 接口 `Index::GetStats` 与 `Index::AnalyzeIndexBySearch`（声明在 `include/vsag/index.h`），
  用于进程内的索引对象；
- 针对序列化索引文件的统一实现 `vsag::analyze_index()`（声明在 `include/vsag/analyze_index.h`），
  它被以下入口复用：
  - 通过 `pip install pyvsag` 安装的 `pyvsag-analyze-index` 命令与 `pyvsag.analyze_index()`
    Python API；
  - 基于 `tools/analyze_index/` 构建的 C++ 命令行工具 `analyze_index`。

基于文件的入口请参见[分析索引文件](#分析索引文件)。

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
专用 Analyzer 可以把不可用的根级指标表示为包含字符串 `skipped_reason` 的对象；公共元数据层
会将该原因同步到 `_analysis.skipped.<metric>`。基线 Analyzer 则直接写入相同的公共表示
`_analysis.skipped`。使用方应以 `_analysis.skipped` 为准，而将 `skipped_reason` 视为索引专属的
兼容性细节。

`BruteForce` 与 `WARP` 提供精确搜索基线计数及耗时；`SINDI_V2` 与 `SIMQ` 提供基线计数和
查询耗时，并将尚无真值的专属质量指标标记为 skipped。`LazyHGraph` 委托当前 BruteForce 或
HGraph 后端，并追加 `phase` 与 `transition_threshold`；其 `index_type` 始终为 `lazy_hgraph`。

Analysis 是同步、面向离线诊断的操作，可能扫描大量索引数据，专用 Analyzer 还可能计算精确
真值。Analysis 不修改搜索或序列化配置，可与只读 Search 并发执行。除非索引提供更强的加锁
快照，否则 `_analysis.consistency` 为 `weak_snapshot`；不要让 Analysis 与 Add、Remove、Update
或 phase transition 并发执行。

Pyramid 的查询分析遵循与 `KnnSearch` 相同的路径限定语义。对于默认 hierarchy，
使用 `Dataset::Paths` 设置路径；对于命名 hierarchy，使用
`Dataset::Paths(hierarchy_name, paths)` 设置路径，并在 Pyramid 搜索参数中选择该
hierarchy。当选中 hierarchy 的根节点为 `NO_INDEX` 时必须提供路径。批量 query 数据集的
外层路径集合每行对应一条 query：旧重载每行保存一个路径字符串，并将 `|` 解释为并集；
结构化重载每行可保存一条或多条原子路径，并将 `|` 当作普通路径字符。Pyramid 只在
每条 query 的 hierarchy 和路径并集所允许的向量中计算真值集，并排除已标记删除的向量。
Pyramid 查询分析目前只支持 KNN 请求，并使用 `SearchRequest` 中的 `query_`、`topk_` 和
`params_str_`；不支持带过滤条件或 iterator 的请求。分析期间应保持索引稳定，不要并发执行
add 或 remove。

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

## 分析索引文件

`vsag::analyze_index()` 用于分析一个已序列化的索引文件。它会识别序列化格式，在不构建、不修改索引的
前提下重新加载索引，输出索引元数据与 `GetStats()` 统计信息；提供查询文件时还会运行
`AnalyzeIndexBySearch()`。以下三个入口共用这一套实现，并返回相同的 JSON 报告：

| 入口 | 获取方式 | 典型用途 |
| --- | --- | --- |
| `pyvsag-analyze-index` 命令 | `pip install pyvsag` | Shell 脚本与 CI 任务 |
| `pyvsag.analyze_index()` | `pip install pyvsag` | Python 程序与 Notebook |
| `vsag::analyze_index()` 与 `analyze_index` 工具 | C++ 库；开启工具的源码构建 | C++ 应用 |

### 安装

```bash
pip install pyvsag==<version>
pyvsag-analyze-index --index_path /path/to/index
```

wheel 内已包含命令、Python API 与原生库，分析时不需要 VSAG 源码、CMake、C++ 编译器或 NumPy。
wheel 的发布平台与 `pyvsag` 的发布矩阵一致（Linux x86_64 与 aarch64，CPython 3.6 至 3.14）。
请安装能够读取该索引的 `pyvsag` 版本，即不低于写出该索引的 VSAG 版本；报告中的
`compatibility.vsag_version` 记录了读取文件所用的版本。

如果 Python 环境的脚本目录不在 `PATH` 中，可使用相同参数运行 `python -m pyvsag_analyze_index`。

### 命令行

```bash
# 静态分析：元数据与统计信息
pyvsag-analyze-index --index_path my_index.vsag > report.json

# 静态 + 动态分析：每个 query 评估 10 个近邻，并指定搜索参数
pyvsag-analyze-index -i my_index.vsag -qp queries.bin -k 10 \
    -sp '{"hgraph": {"ef_search": 200}}'
```

| 参数 | 缩写 | 默认值 | 描述 |
| --- | --- | --- | --- |
| `--index_path` | `-i` | 必需 | 待分析的序列化索引文件。 |
| `--build_parameter` | `-bp` | `default` | 重新加载索引时使用的构建参数（JSON）。`default` 表示使用文件中记录的参数；未记录参数的文件需要提供该参数（见[支持的文件](#支持的文件与索引类型)）。 |
| `--query_path` | `-qp` | 无 | 查询数据集（见[输入文件](#输入文件)）。未提供时只进行静态分析。它不是 Pyramid hierarchy 路径元数据。 |
| `--query_data_type` | | `auto` | `auto`（稀疏索引按 sparse 读取，其余按 dense 读取）、`dense` 或 `sparse`。 |
| `--base_path` | | 无 | 仅 SINDI：用于计算精确真值以及需要原始向量的指标的 sparse CSR base 数据集。 |
| `--groundtruth_path` | | 无 | 仅 SINDI：动态分析复用的已有真值文件，不能与 `--save_groundtruth_path` 同时使用。 |
| `--save_groundtruth_path` | | 无 | 仅 SINDI：保存动态分析期间生成的真值的路径。 |
| `--search_parameter` | `-sp` | `default` | 动态分析使用的搜索参数（JSON）。`default` 对 HGraph 与 LazyHGraph 使用 `{"hgraph": {"ef_search": max(100, topk)}}`，对 Pyramid 使用 `{"pyramid": {"ef_search": max(100, topk)}}`，对 SINDI_V2 使用 `{"sindi_v2": {}}`，对其余索引类型使用 `{}`。 |
| `--topk` | `-k` | `100` | 每个 query 评估的近邻数。 |
| `--compact` | | 关闭 | 将报告输出为一行，而不是缩进格式。 |
| `--version` | | | 输出版本号后退出。 |

所有选项都同时接受连字符写法，例如 `--index-path`。stdout 上只有报告；诊断信息
（包括原生库的启动横幅与日志）写入 stderr，可通过 `VSAG_LOG_LEVEL`（`off`、`error`、`warn`、
`info` 等）设置其级别。失败时 stderr 输出一行错误，stdout 不输出任何内容：

```text
pyvsag-analyze-index: error [missing_file]: index_path does not exist: /data/missing.vsag
```

| 退出码 | 含义 |
| --- | --- |
| `0` | 已输出报告。 |
| `1` | 分析失败，方括号中的[错误码](#错误)说明原因。 |
| `2` | 命令行无效，例如缺少 `--index_path` 或 `--topk` 不是数字。 |

### Python API

```python
import pyvsag

report = pyvsag.analyze_index(
    "my_index.vsag",
    build_parameter="default",
    query_path="queries.bin",
    query_data_type="auto",
    base_path=None,
    groundtruth_path=None,
    save_groundtruth_path=None,
    search_parameter={"hgraph": {"ef_search": 200}},
    topk=10,
)
print(report["dynamic_analysis"]["result"]["recall_query"])
```

各参数的含义与命令行选项相同。路径参数接受 `str` 或 `os.PathLike`，`None` 表示未提供；
`build_parameter` 与 `search_parameter` 接受 JSON 字符串、`dict` 或 `None`（等同于 `"default"`）。
`index_path` 之后的参数都只能以关键字形式传入。函数返回可被 `json.dumps()` 序列化的 `dict`；
参数类型错误时抛出 `TypeError`，其余失败均抛出 `pyvsag.IndexAnalysisError`（`RuntimeError` 的
子类），其 `code` 属性为[错误码](#错误)，`message` 为说明：

```python
try:
    report = pyvsag.analyze_index("my_index.vsag")
except pyvsag.IndexAnalysisError as error:
    print(error.code, error.message)
```

`pyvsag.ANALYZE_INDEX_SCHEMA_VERSION` 是报告的 `schema_version`。分析期间会释放 GIL，且不会修改
日志级别。与命令不同，Python API 不会重定向原生日志：与其他 `pyvsag` 调用一样，`info` 与 `debug`
日志写入进程的 stdout，警告与错误写入 stderr。设置 `VSAG_LOG_LEVEL=warn` 或调用
`pyvsag.set_logger_off()` 即可让 stdout 中不出现这些日志。

### C++ API

```cpp
#include <vsag/vsag.h>

vsag::AnalyzeIndexOptions options;
options.index_path = "my_index.vsag";
options.query_path = "queries.bin";
options.topk = 10;
auto report = vsag::analyze_index(options);
if (report.has_value()) {
    std::cout << report.value() << std::endl;  // JSON 报告
} else {
    std::cerr << report.error().message << std::endl;  // report.error().type 说明原因
}
```

`AnalyzeIndexOptions` 的每个字段对应一个命令行选项，空字符串表示未提供。`vsag::analyze_index()`
返回紧凑格式的 JSON 对象字符串；失败时返回 `vsag::Error`，其 `type` 与[错误码](#错误)对应。

### 报告格式

报告是一个 JSON 对象，依次包含以下成员：

| 成员 | 内容 |
| --- | --- |
| `schema_version` | 报告格式版本，当前为 `1`。 |
| `metadata` | 加载结果：索引类型、维度、数据类型、距离度量、规模与参数。 |
| `static_analysis` | 索引的 `GetStats()` JSON，见[静态指标](#getstats-输出的静态指标)。 |
| `dynamic_analysis` | 未提供查询文件时为 `null`；否则包含分析输入，以及位于 `result` 下的 `AnalyzeIndexBySearch()` JSON，见[动态指标](#analyzeindexbysearch-输出的动态指标)。 |
| `warnings` | 由 `code` 与 `message` 组成的对象，说明会影响结果解读的假设与覆盖。 |
| `compatibility` | 文件的读取方式：格式、读取所用库版本以及元数据是否完整。 |

`static_analysis` 与 `dynamic_analysis.result` 保留各自的 `_analysis` 成员（见
[公共输出契约](#公共输出契约)），其 `schema_version` 表示这些 analyzer 输出的版本，与顶层
`schema_version` 相互独立。

| `metadata` 成员 | 含义 |
| --- | --- |
| `index_path`、`file_size_bytes` | 被分析的文件。 |
| `index_type` | 索引的工厂名称，例如 `hgraph` 或 `sindi`。 |
| `dim` | 向量维度；稀疏索引为 `null`。 |
| `data_type`、`metric`、`extra_info_size` | 已加载索引的通用构建参数。 |
| `num_elements`、`memory_usage_bytes` | 已加载索引的 `GetNumElements()` 与 `GetMemoryUsage()`。 |
| `parameter_source` | 使用文件中记录的参数时为 `file`，使用 `build_parameter` 时为 `user`。 |
| `index_param` | 文件中记录的索引参数（采用索引参数 `ToJson()` 的布局），或 `build_parameter` 中的 `index_param`。 |
| `build_parameter` | 传入的 `build_parameter`，未传入时为 `null`。 |
| `basic_info` | 文件中记录的 `basic_info` 元数据（不含 `index_param`）；legacy 文件为 `null`。 |

| `dynamic_analysis` 成员 | 含义 |
| --- | --- |
| `query_path`、`query_data_type` | 查询文件及其读取方式（`dense` 或 `sparse`）。 |
| `query_count`、`query_dim` 或 `query_cols` | query 数量，以及维度（dense）或列数（sparse）。 |
| `topk` | 每个 query 评估的近邻数。 |
| `search_parameter`、`search_parameter_source` | 实际使用的搜索参数，以及它是 `default` 还是 `user` 提供的值。 |
| `base_path`、`groundtruth_path`、`save_groundtruth_path` | 传入时才会出现。 |
| `result` | `AnalyzeIndexBySearch()` 的 JSON。 |

| 警告 `code` | 含义 |
| --- | --- |
| `metadata_defaults` | 文件未记录索引的数据类型或距离度量，分析器假设为 `float32` 或 `l2`（列于 `compatibility.defaulted_fields`）。BruteForce、WARP 与 SIMQ 的 footer 文件受此影响；若索引使用其他度量或数据类型，请传入 `build_parameter`。 |
| `build_parameter_override` | 索引使用 `build_parameter` 加载，而不是文件中记录的参数。 |
| `legacy_format` | 文件没有 VSAG footer，已借助 `build_parameter` 按 legacy 文件加载。 |
| `memory_io` | 记录的参数使用了文件型 IO（`buffer_io`、`async_io`、`mmap_io`、`uring_io`），或者 footer、chunked、HGraph streaming 文件使用了 `reader_io`。文件内联携带全部数据，因此这些组件改为加载到内存，且不会打开任何记录的文件路径（见 `compatibility.memory_io_overrides`）。使用文件型 IO 的 streaming 文件：HGraph、BruteForce 与 WARP 以同样方式加载，其余索引类型以 `unsupported_index_operation` 拒绝。 |
| `empty_index` | 索引不包含任何向量。 |

| `compatibility` 成员 | 含义 |
| --- | --- |
| `vsag_version` | 读取文件所用的库版本。 |
| `serialization_format` | `footer`、`chunked`、`streaming` 或 `legacy`，见[支持的文件](#支持的文件与索引类型)。 |
| `lazy_hgraph_container` | 文件是否由 LazyHGraph 索引写出。 |
| `metadata_complete`、`defaulted_fields` | 分析器是否需要假设通用参数，以及假设的取值。 |
| `memory_io_overrides` | 记录的 IO 类型被替换为内存 IO 的参数。 |
| `chunked_codec`、`chunk_size` | 仅 chunked 文件。 |
| `streaming_blocks` | 仅 streaming 文件：各 body block 的名称。 |

以下是一个带动态分析的 HGraph 索引报告（已省略部分内容）：

```json
{
  "schema_version": 1,
  "metadata": {
    "index_path": "my_index.vsag",
    "file_size_bytes": 169113,
    "index_type": "hgraph",
    "dim": 16,
    "data_type": "float32",
    "metric": "l2",
    "extra_info_size": 0,
    "num_elements": 500,
    "parameter_source": "file",
    "index_param": {"type": "hgraph", "...": "..."},
    "build_parameter": null,
    "basic_info": {"total_count": 500, "...": "..."}
  },
  "static_analysis": {
    "_analysis": {
      "analysis_type": "stats",
      "consistency": "weak_snapshot",
      "index_type": "hgraph",
      "schema_version": 1,
      "skipped": {"build_cache_hit_rate": "index was not built from an imported cache"},
      "status": "partial"
    },
    "connect_components": 5,
    "recall_base": 0.989,
    "total_count": 500,
    "...": "..."
  },
  "dynamic_analysis": {
    "query_path": "queries.bin",
    "query_data_type": "dense",
    "query_count": 20,
    "query_dim": 16,
    "topk": 10,
    "search_parameter": {"hgraph": {"ef_search": 100}},
    "search_parameter_source": "default",
    "result": {
      "_analysis": {"analysis_type": "search", "sample_count": 20, "status": "complete", "...": "..."},
      "recall_query": 0.995,
      "time_cost_query": 0.788,
      "...": "..."
    }
  },
  "warnings": [],
  "compatibility": {
    "vsag_version": "v1.2.0",
    "serialization_format": "footer",
    "lazy_hgraph_container": false,
    "metadata_complete": true,
    "defaulted_fields": {},
    "memory_io_overrides": []
  }
}
```

### 支持的文件与索引类型

分析器读取单个序列化索引文件，并根据文件内容识别其格式：

| 格式 | 写出方式 | 分析支持 |
| --- | --- | --- |
| `footer` | `Index::Serialize(std::ostream&)`、`pyvsag.Index.save()` | 支持。除非传入 `build_parameter`，否则使用 footer 中记录的参数。 |
| `chunked` | `Index::Serialize(SerializeWriter&, chunk_size)` | 支持未压缩文件。压缩文件需要写出它的应用所提供的解压 `DeserializeReader`，会以 `unsupported_index_operation` 失败。 |
| `streaming` | `Index::SerializeStreaming()` | 支持。使用 `Index::Load()` 与文件中记录的参数加载；不接受 `build_parameter`。 |
| `legacy` | VSAG 0.15 之前的版本，或没有 footer 的文件 | 仅在传入 `build_parameter` 时支持；文件未记录索引类型时，需在 `"type"` 中指明。 |

`BinarySet` 与 `ReaderSet` 序列化由多个部分组成，不是单个文件，因此不受支持。

| 索引类型 | 仅凭文件即可进行的静态分析 | 动态分析 |
| --- | --- | --- |
| `hgraph` | footer、chunked 与 streaming 文件。 | dense float32 query。 |
| `ivf` | footer 与 streaming 文件。 | dense float32 query。 |
| `brute_force` | footer 文件假设 `l2` 与 `float32`（`metadata_defaults`）；streaming 文件会记录二者。 | dense float32 query。 |
| `pyramid` | streaming 文件。footer 文件未记录维度、距离度量与数据类型，需要 `build_parameter`。 | 被搜索 hierarchy 的根节点已构建时支持 dense float32 query。需要按 query 指定路径的 hierarchy 请通过 C++ API 配合 `Dataset::Paths` 分析。 |
| `sindi` | footer 与 streaming 文件。`base_path` 可补充需要原始向量的指标。 | sparse query；适用 `base_path`、`groundtruth_path` 与 `save_groundtruth_path`。 |
| `sindi_v2` | footer 与 streaming 文件。 | sparse query（基线指标）。 |
| `warp`、`simq` | footer 文件，假设 `l2` 与 `float32`（`metadata_defaults`）。 | 不支持：查询文件无法表达多向量 query。 |
| `lazy_hgraph` | 需要 `build_parameter`：文件未记录 LazyHGraph 参数。 | dense float32 query。 |

dense 查询文件存放 float32 向量，因此 `int8`、`float16` 与 `bfloat16` 索引的动态分析会以
`unsupported_index_operation` 失败；它们的静态分析可用，但 HGraph 统计目前尚不支持 `float16`
与 `bfloat16` 索引。

### 错误

| 错误码 | 原因 |
| --- | --- |
| `invalid_argument` | 选项或选项组合无效：JSON 格式错误、`topk` 不是正数、只提供 `groundtruth_path` 而没有 `query_path`、对非 SINDI 索引传入 `base_path`、query 维度与索引不一致、对 streaming 文件传入 `build_parameter`，或 `build_parameter` 指明了另一种索引类型。 |
| `missing_file` | 输入文件或 `save_groundtruth_path` 所在目录不存在。 |
| `read_error` | 输入文件无法读取，或在加载索引时提前结束。 |
| `invalid_binary` | 索引文件不是 VSAG 索引、已被截断或损坏（例如 footer 校验和不匹配），或者查询、base、真值文件不符合文档描述的布局。 |
| `unsupported_index` | 文件未标识受支持的索引类型。 |
| `unsupported_index_operation` | 文件或分析方式不受支持：压缩的 chunked 文件、未传入 `build_parameter` 的 LazyHGraph 与 Pyramid footer 文件，或索引、查询文件无法支持的动态分析。 |
| `index_empty` | 文件中的索引在为空时序列化，且未记录参数。 |
| `no_enough_memory` | 索引或查询无法装入内存。 |
| `internal_error`、`unknown_error` | 索引内部发生了意外错误。 |

当索引实现返回 `vsag::ErrorType` 中的其他错误时，也可能出现对应的错误码。错误信息会指出相关的
文件或选项，并尽可能说明修复方法。

### 输入文件

所有整数与浮点数均为小端序。

- **dense 查询文件**：`uint32 rows, uint32 dim`，随后是 `float32 data[rows * dim]`。文件大小必须与
  文件头严格一致。
- **sparse CSR 文件**（稀疏 query 与 SINDI base 数据集）：`int64 rows, int64 cols, int64 nnz`，
  随后是 `int64 indptr[rows + 1]`、`int32 indices[nnz]` 与 `float32 values[nnz]`。`indptr` 必须
  从 0 开始、单调不减并以 `nnz` 结束；索引必须位于 `[0, cols)`。
- **真值文件**（`.dev.gt`）：`uint32 rows, uint32 topk`，随后是 `int32 ids[rows * topk]` 与
  `float32 distances[rows * topk]`。每个 query 需要一行，每行至少包含 `min(topk, num_elements)`
  个近邻。

输入文件会在加载索引之前完成校验，格式错误的文件会尽早失败。

### C++ `analyze_index` 工具

C++ 工具随其他工具一起从源码构建：

```bash
# 通过项目 Makefile
VSAG_ENABLE_TOOLS=ON make release

# 也可直接通过 CMake
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release -j
./build-release/tools/analyze_index/analyze_index --index_path my_index.vsag
```

它接受相同的选项（包括连字符写法），输出相同的报告，并使用相同的退出码与错误格式
（`analyze_index: error [<code>]: <message>`）。旧版本打印日志行，现在改为输出 JSON 报告。
`--query_path empty` 仍然可用，表示不提供查询文件。在 Linux 上，工具会把库的启动横幅挡在 stdout
之外；在其他平台上请设置 `VSAG_SUPPRESS_INIT_BANNER=1`。

### 限制

- 整个索引以及 query 会被加载到内存中。
- 使用文件型存储的 streaming 文件，只有在存储可以改为加载到内存时才能分析，目前仅限 HGraph 的基础编码、
  精确编码与原始向量。其他文件型组件（例如 BruteForce、WARP 的编码，或 RaBitQ 分离存储的 supplement）
  会返回 `unsupported_index_operation`，因为加载它们会使用文件中记录的路径。
- 查询文件只包含向量：不支持 Pyramid hierarchy 路径、过滤条件、属性或多向量 query。这些场景
  请在 C++ 中使用 `AnalyzeIndexBySearch`。
- 分析是同步的，适用于离线诊断；[公共输出契约](#公共输出契约)中描述的开销同样适用。
- footer 与 streaming header 会被校验（校验和与布局），但索引主体由该索引自身的反序列化逻辑读取。
  主体已损坏的文件，或借助 `build_parameter` 加载的无 footer 文件，其健壮性取决于该反序列化逻辑：
  遇到无效数据时可能以 `no_enough_memory` 失败或占用大量内存。请在有内存限制的环境中分析不可信文件。

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
- 文件分析接口：`include/vsag/analyze_index.h` 中的 `vsag::analyze_index`，实现位于
  `src/analyze_index.cpp`
- Python API 与命令：`python/pyvsag/index_analysis.py` 与
  `python/pyvsag_analyze_index.py`
- 实现：`src/analyzer/{analyzer,hgraph_analyzer,pyramid_analyzer}.h`
- 工具源码：`tools/analyze_index/`
- 本地工具入口：`tools/analyze_index/README_zh.md`
