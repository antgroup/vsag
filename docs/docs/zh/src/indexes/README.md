# 索引

VSAG 提供了一系列索引实现，它们共享同一套构建式 API、同一种序列化格式、同一组操作
（`Build`、`Add`、`KnnSearch`、`RangeSearch`、`Remove`、`Serialize` / `Deserialize` 等），
差异在于底层使用的数据结构与折中取舍。

本节覆盖当前活跃维护的索引：

| 索引 | 文档 | 适用场景 |
|------|------|---------|
| `hgraph` | [HGraph](hgraph.md) | 通用高召回图索引，量化选项丰富 |
| `ivf` | [IVF](ivf.md) | 基于分桶的检索，适合高吞吐批查询与超大规模语料 |
| `sindi` | [SINDI](sindi.md) | 稀疏向量（BM25 / 学习稀疏）上的内积检索 |
| `pyramid` | [Pyramid](pyramid.md) | 多租户 / 标签分区的层级索引 |

`brute_force` 作为精确检索基线也可使用（见
[创建索引](../guide/create_index.md) 与 `examples/cpp/105_index_brute_force.cpp`）。

`hnsw` 与 `diskann` 保留用于向后兼容，但已 **弃用**；新部署请优先选择 `hgraph`（图索引）
或 `ivf`（分桶索引）。

## 参数约定

所有索引共享以下顶层构建字段：

| 字段 | 可选值 | 说明 |
|------|--------|------|
| `dim` | 正整数 | 向量维度；构建后不可变 |
| `dtype` | `float32` / `float16` / `bfloat16` / `int8` / `sparse` | `sparse` 仅 SINDI 使用 |
| `metric_type` | `l2` / `ip` / `cosine` | 查询时必须保持一致（SINDI 仅支持 `ip`） |

索引特有的构建参数放在 `index_param` 子对象中；查询参数放在以索引名命名的子对象中
（例如 `hgraph`、`ivf`、`sindi`、`pyramid`）。具体参数定义在各索引页面内给出，也可查阅
[索引参数](../resources/index_parameters.md) 进行总览。
