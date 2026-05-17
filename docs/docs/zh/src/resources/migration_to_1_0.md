# 升级到 VSAG 1.0

> **状态：** 活文档 —— 章节会在 PR
> [#2070](https://github.com/antgroup/vsag/pull/2070) 中逐步补全，
> 整体进度跟踪在 [issue #2069](https://github.com/antgroup/vsag/issues/2069)。

本页汇总了从 **VSAG 0.18.x**（及更早版本）平滑升级到 **VSAG 1.0** 所需了解的
全部内容。请在重新编译或重新部署前先阅读本页。

> 1.0 之后版本之间的兼容性规则在 [API 稳定性](api_stability.md) 中说明；
> 本页只覆盖 0.18 → 1.0 的一次性迁移。

## 一图速览

| 主题 | 1.0 中的状态 | 操作 |
|------|-------------|------|
| `hnsw` 索引 | 已弃用，仍可用 | 新增部署改用 [HGraph](../indexes/hgraph.md) |
| `diskann` 索引 | 已弃用，仍可用 | 新增部署改用 [IVF](../indexes/ivf.md) 或 [内存-磁盘混合索引](../advanced/hybrid_index.md) |
| `Index::KnnSearch(query, k, SearchParam&)` | 已弃用的重载 | 改用 `Index::SearchWithRequest(SearchRequest)` |
| `SearchParam::allocator` | 已弃用字段 | 改用 `SearchRequest::search_allocator_` |
| `Index::CalDistanceById`（批量） | 保留（拼写错误的名字） | 继续使用；正确拼写的 `CalcDistancesById` 在规划中（见 [#2068](https://github.com/antgroup/vsag/issues/2068)） |
| 0.18.x 序列化产物 | 1.0 可读 | 升级后建议重新序列化以采用新的布局优化 |
| 公共 C ABI | 稳定 | 无需操作 |

下文逐行展开，给出可直接套用的代码片段。

## 已弃用的索引

### `hnsw` → `hgraph`

`hnsw` 是从 hnswlib 继承下来的图索引。1.0 中为了向后兼容仍然保留，但
**已标记为弃用**；新增部署请使用 [HGraph](../indexes/hgraph.md)，它是
`hnsw` 的超集：

- 同样的分层图拓扑，同样的 `max_degree` / `ef_construction` /
  `ef_search` 调参。
- 统一的 `index_param` 构建参数 schema，量化选项更丰富（`fp32`、
  `fp16`、`bf16`、`sq8`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、
  `rabitq`）。
- 可选的重排（`use_reorder` + `precise_quantization_type`）、去重、
  `Remove()`、以及基于 ELP 的运行期自动调参。

构建期映射：

```diff
- auto index = vsag::Factory::CreateIndex("hnsw", R"({
-     "dim": 768,
-     "dtype": "float32",
-     "metric_type": "ip",
-     "hnsw": {
-         "max_degree": 32,
-         "ef_construction": 400
-     }
- })").value();
+ auto index = vsag::Factory::CreateIndex("hgraph", R"({
+     "dim": 768,
+     "dtype": "float32",
+     "metric_type": "ip",
+     "index_param": {
+         "base_quantization_type": "fp32",
+         "max_degree": 32,
+         "ef_construction": 400
+     }
+ })").value();
```

搜索期映射：

```diff
- auto result = index->KnnSearch(query, k, R"({"hnsw": {"ef_search": 100}})").value();
+ auto result = index->KnnSearch(query, k, R"({"hgraph": {"ef_search": 100}})").value();
```

两个易错点：

1. 构建子对象的 key 从 `"hnsw"` 变成 `"index_param"`，并且
   `base_quantization_type` 是必填字段。
2. 搜索子对象的 key 也从 `"hnsw"` 变成 `"hgraph"`。

### `diskann` → `ivf` 或 内存-磁盘混合索引

`diskann` 提供了内存放 PQ、磁盘放原始向量的混合检索能力。1.0 中
**已弃用**；请按以下顺序选择替代项：

- [IVF](../indexes/ivf.md) —— 适合大规模分区检索；当数据可以完全放入
  内存时，是 `diskann` 的自然替代。
- [内存-磁盘混合索引](../advanced/hybrid_index.md) —— 当确实需要把
  部分索引下沉到 NVMe（语料巨大、内存预算紧张）时再使用。

优先尝试 IVF；只有在确实测量到内存是瓶颈时，再退回磁盘混合配置。

### `hnsw` / `diskann` 不再作为首选示例

网站文档中 [创建索引](../guide/create_index.md)、
[索引参数](index_parameters.md)、[序列化](../advanced/serialization.md)
等页面在 1.0 中默认以 `hgraph` 作为示例。原有的示例代码
`examples/cpp/101_index_hnsw.cpp`、`examples/cpp/102_index_diskann.cpp`
仍保留以供参考。

## 后续章节大纲（待补充）

- 已弃用的搜索 API 与 `SearchRequest` 迁移。
- `CalDistanceById` 拼写问题与 `CalcDistancesById` 迁移路径。
- 序列化格式兼容性声明。
- 默认值与行为变化。
- 构建系统 / 打包相关说明。
- 升级操作清单。
