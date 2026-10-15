# 索引参数

本页汇总 VSAG 各索引类型的常用参数。完整枚举请参考源码：

- 构建参数键：`src/constants.cpp`
- 公开常量：`include/vsag/constants.h`
- 每个索引的示例：`examples/cpp/*_index_*.cpp`（例如 `103_index_hgraph.cpp`）

## 通用参数

所有索引在构建时都需要提供以下顶层字段：

| 字段 | 取值 | 说明 |
|------|------|------|
| `dim` | 正整数 | 向量维度，构建后不可更改 |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` | 向量数据类型，决定索引内部表示 |
| `metric_type` | `l2` / `ip` / `cosine` | 距离度量 |

## HGraph

HGraph 的构建参数使用通用的 `index_param` 键（参见 `examples/cpp/103_index_hgraph.cpp`）；
`hgraph` 键则保留给搜索期参数。

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 400
    }
}
```

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `max_degree` | 16~48 | 每节点最大出边数 |
| `ef_construction` | 200~500 | 构建阶段候选集大小，越大召回越高、构建越慢 |
| `alpha` | `1.0` | 最终 robust pruning 的系数；PiPNN 要求有限且不小于 `1.0` |
| `base_quantization_type` | `fp32` / `fp16` / `bf16` / `sq8` / `sq4` / `pq` | 主存储的量化策略 —— 支持的全部取值见[量化章节](../quantization/README.md) |

可选构建参数 `mci_knng_source` 会启用 MCI 挂件，并选择候选图来源：
`hgraph`（默认）或 `odescent`。构建和过滤器要求见
[MCI 挂件指南](../advanced/hgraph_mci_companion.md)。

MCI 维护参数包括 `mci_incremental_degree_min`、`mci_incremental_degree_n_divisor`、
`mci_incremental_degree_mcs_divisor`、`mci_delete_clique_size_threshold` 和
`mci_delete_node_mct_threshold`。物理删除要求 `support_force_remove: true`；
默认值与限制详见[维护指南](../advanced/hgraph_mci_companion.md)。

可选搜索参数 `use_hybrid_traversal` 将稀疏邻居和团邻居合并遍历；
`hybrid_vob`、`hybrid_filter_cost_ratio`、`mci_seed_coverage` 和 `mci_seed_max_count`
控制每次展开开销及 seed 预算。详见 [MCI 挂件指南](../advanced/hgraph_mci_companion.md#动态邻居遍历)。

### PiPNN 构建参数

当 HGraph 的 `graph_type` 为 `pipnn` 时，还会从 `index_param` 接受以下构建参数。
PiPNN 仅适用于 HGraph；Pyramid 索引不支持 `graph_type: "pipnn"`。

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `pipnn_max_leaf_size` | `1024` | 分区叶子的最大点数，范围 `[2, 2147483647]` |
| `pipnn_min_leaf_size` | `64` | 合并过小叶子时的目标规模，必须为正数且不超过 `pipnn_max_leaf_size` |
| `pipnn_leader_sample_rate` | `0.005` | leader 采样比率，范围 `(0, 1]`；每个分区使用 `2` 到 `1000` 个 leader，且不超过分区点数 |
| `pipnn_fanout` | `[10, 2]` | 连续分区层级的正数最近 leader fanout；更深层使用 `1` |
| `pipnn_leaf_neighbor_count` | `5` | 每个点在每个叶子中贡献的最近候选数，必须为正数；过小叶子至少使用 `4` |
| `pipnn_hash_plane_count` | `12` | 方向 hash 位数，范围 `[1, 15]`，且 `max_degree <= 2^pipnn_hash_plane_count` |
| `pipnn_reservoir_size` | `64` | 最终剪枝前每个点的候选容量，必须为正数；实际容量为 `max(value, max_degree)` 且不能超过 `65535` |

现有的 `alpha` 字段控制 PiPNN 的最终 robust pruning。`ef_construction` 只调节 NSW，
对 PiPNN 不生效。

搜索时：

```json
{"hgraph": {"ef_search": 100}}
```

`hgraph` 搜索参数还接受 `brute_force_threshold`（`[0.0, 1.0]` 区间的 float，
默认 `0.0`）。当取值 `> 0` 且当前请求的 filter 的 `ValidRatio()` 不超过该
阈值时，HGraph 会跳过图遍历，直接在通过过滤的 id 上做精确暴扫。详见
[HGraph 索引文档](../indexes/hgraph.md#高选择性过滤下的暴搜回退brute_force_threshold)。

## LazyHGraph

LazyHGraph 的构建参数可以放在顶层 `lazy_hgraph` 对象中（推荐，语义更清晰），也可以放在
通用的 `index_param` 对象中。`hgraph` 子对象会转交给转换后的内部 HGraph。

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "lazy_hgraph": {
        "transition_threshold": 1000,
        "hgraph": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100
        }
    }
}
```

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `transition_threshold` | `1000` 或按业务规模设置 | 从精确 flat 搜索转换到 HGraph 的正整数向量数量阈值 |
| `hgraph` | HGraph 构建对象 | graph 阶段的参数；见 [HGraph](../indexes/hgraph.md) |

LazyHGraph 只支持 `dtype: "float32"`。搜索参数使用 `hgraph` 对象，例如
`{"hgraph": {"ef_search": 100}}`。详见 [LazyHGraph 索引文档](../indexes/lazy_hgraph.md)。

`hgraph` 搜索参数还接受以下 filter 相关参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `skip_ratio` | float | `0.2` | 控制带 filter 搜索时跳过候选检查的比例，取值范围为 `[0.0, 1.0]`。值越大，跳过越激进，搜索越快但可能影响召回。 |
| `skip_strategy` | string | `"deterministic_accumulative"` | 跳过策略。支持 `"random"` 和 `"deterministic_accumulative"`。 |

## IVF

```json
{
    "ivf": {
        "nlist": 4096,
        "base_quantization_type": "sq8",
        "nprobe": 32
    }
}
```

## Brute Force

```json
{"brute_force": {}}
```

无需额外参数。

## Pyramid

Pyramid 支持按 tag 组织多棵子图：

```json
{
    "pyramid": {
        "tag_dim": 1,
        "max_degree": 24,
        "ef_construction": 300
    }
}
```

Pyramid 不支持 `graph_type: "pipnn"`；上文 PiPNN 构建参数仅适用于 HGraph。

## SINDI（稀疏向量）

```json
{
    "sindi": {
        "top_k": 32,
        "doc_prune_ratio": 0.1
    }
}
```

## 运行期参数

除构建参数外，`Index::Tune` 与 `SearchParam` 可在运行时调整 `ef_search`、`nprobe` 等参数。参考
[优化器](../advanced/optimizer.md) 与各 `examples/cpp/3xx_feature_*.cpp` 示例。
