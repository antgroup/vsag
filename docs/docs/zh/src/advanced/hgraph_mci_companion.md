# HGraph MCI 挂件

HGraph 可以选择构建一个 MCI（Maximal Clique Index）挂件，用于带过滤条件的 KNN
搜索。这个挂件把团信息存放在 HGraph 索引内部，并复用 HGraph 的向量存储。它不是
独立索引类型：创建索引时仍然使用 `hgraph`，不要使用 `mci`。

当主要负载是过滤搜索，并且过滤后只保留较小比例的向量时，可以启用这个功能。搜索
时，HGraph 会比较 `Filter::ValidRatio()` 和阈值，自动选择普通 HGraph 搜索或 MCI
挂件搜索。

## 构建配置

MCI 构建参数直接放在 `index_param` 下，不再使用嵌套对象。满足以下任一条件即
启用挂件：`use_mci` 为 true，或出现 `mci_mcs`、`mci_clique_max`、`mci_alpha`、
`mci_knng_source` 中任意一个。`mci_knng_source` 用于
选择构团所需 KNN 图的来源：可以从已经构建好的 HGraph bottom graph 派生，也可以
单独构建一张 ODescent 图。

```json
{
  "dtype": "float32",
  "metric_type": "l2",
  "dim": 128,
  "index_param": {
    "base_quantization_type": "fp32",
    "max_degree": 32,
    "ef_construction": 400,
    "mci_mcs": 200,
    "mci_clique_max": 50,
    "mci_alpha": 1.2,
    "mci_knng_source": "odescent"
  }
}
```

`mci_knng_source` 默认为 `hgraph`，保持现有行为不变。设为 `odescent` 时，
MCI 会直接基于已存储向量构建专用 KNN 图。若内部配置了外部 KNN 图文件路径，
外部文件的优先级高于此选项。

| 参数 | 作用 |
| --- | --- |
| `use_mci` | 设为 `true` 时使用默认构建参数启用 MCI。 |
| `mci_mcs` | 构建团时使用的候选邻居数量。 |
| `mci_clique_max` | 全量构建时保留的极大团的最小大小。团始终完整保存，因此这是下限而非上限。 |
| `mci_knng_source` | KNN 图来源：`hgraph`（默认）或 `odescent`。 |
| `mci_alpha` | 团构建扩展系数。 |
| `mci_incremental_join_ratio_threshold` | Add 时加入已有团的阈值。 |
| `mci_incremental_added_mct` | 新节点最多加入的已有团数量。 |
| `mci_incremental_clique_max` | 增量创建新团时的最大团大小。 |

## 搜索配置

搜索参数放在 `hgraph` 搜索对象下：

```json
{
    "hgraph": {
      "ef_search": 120,
      "use_mci": true,
      "mci_seed_ratio": 0.1,
      "hgraph_valid_ratio_threshold": 0.2
    }
}
```

`use_mci` 在搜索时默认为 true，可以设为 false 来仅对本次查询关闭原有 MCI 路径。
下文的混合遍历使用独立的启用开关。
`hgraph_valid_ratio_threshold` 是搜索路由阈值：`ValidRatio()` 低于该阈值时走 MCI，
否则走 HGraph。默认值是 `0.05`。

seed 数量按 `ceil(sqrt(当前向量总数) * mci_seed_ratio)` 计算，并且至少为 1。
`mci_seed_ratio` 默认值为 `0.1`，必须是有限的非负数。最终 seed 数量不会超过
满足过滤条件的点数。

MCI 路径要求使用 bitset 过滤器，或通过 `GetValidIds()` 提供非空有效 ID 列表的
自定义 `Filter`，并提供合理的 `ValidRatio()`。仅有函数回调的过滤器回退到 HGraph。
采样只解析选中的标签，不会为过期或重复 ID 补充 seed；若没有可用 seed，则回退到 HGraph。

## 动态邻居遍历

`use_hybrid_traversal` 默认为 `false`。启用后，在同一次遍历中处理每个展开节点的
两类邻居：先计算稀疏 HGraph 邻居的距离并加入候选堆，保持图的连通性；团成员则
先判断过滤条件，再计算距离，并受每次节点展开的虚拟开销预算约束。

当 `considered * (hybrid_filter_cost_ratio + local_selectivity) >= hybrid_vob` 时，
停止当前节点的团遍历。`local_selectivity` 是本次展开中已考虑团成员的过滤命中率，
`hybrid_filter_cost_ratio` 是单次过滤相对于单次距离计算的开销比。`hybrid_vob`
以距离计算开销为单位，默认 `0.0`（不提前停止）；开销比默认 `1.0`。
两个参数都必须是有限的非负数。

```json
{
    "hgraph": {
      "ef_search": 600,
      "use_hybrid_traversal": true,
      "hybrid_vob": 32.0,
      "mci_seed_ratio": 1.0,
      "mci_seed_coverage": 1.0
    }
}
```

混合遍历复用 MCI 的有效 ID 和 bitset 采样器。seed 预算下限为
`max(1, ceil(sqrt(N) * mci_seed_ratio))`。仅当覆盖目标
`ceil(mci_seed_coverage * valid_count)` 不超过 N 且满足 `mci_seed_max_count`
上限时，才用该目标提高预算。超出上限时直接放弃覆盖项，而非截断到上限。
覆盖比例默认 `1.0`，上限默认 `32768`（`0` 表示无限制）；两者必须非负，覆盖比例
还必须有限。ratio 和 coverage 都为零时仍保留一个 seed。这两个新参数只影响混合
遍历，原有 MCI 路径的 seed 预算保持不变。

当有效 ID 列表证明 seed 已覆盖全部有效点时，直接从 seed 返回结果并跳过展开。
这对遍历使用的距离是精确的；量化编码仍有其通常的近似误差和重排行为。bitset
采样器无法证明穷尽性，因此不会跳过展开。没有可用 seed 列表的过滤器从粗搜索入口开始。

混合遍历独立于搜索参数 `use_mci`。当前节点数没有可用挂件、范围搜索或存在属性
过滤执行器时，回退到已有路径。启用的暴力搜索阈值优先于混合遍历。
此 `1.0` 回移保留单查询接口，不引入 main 的批量搜索 API。

查询结果的 `GetStatistics()` 通过 `mci_hybrid_route`（运行混合遍历时为 `"hybrid"`）、
`hybrid_seed_budget`、`hybrid_seeded_entries`、`hybrid_expansion_skipped`、
`hybrid_expanded_nodes`、`hybrid_mci_members_considered`、`hybrid_dist_computations`
及 `hybrid_mci_stopped_early` 报告行为。

## Add、序列化和统计

通过扁平构建参数启用 MCI 后，`HGraph::Add()` 会在每个新点插入 HGraph 后更新
MCI 挂件。它会先尝试加入合适的已有团；如果没有好的候选团，则为新点创建
一个小的增量团。

注意：MCI 索引不应通过在空索引上不断调用 `Add()` 从 0 开始构建。增量添加路径适用于
已有初始索引后，再追加少量向量的场景。

团数据会作为 HGraph 索引的一部分序列化。加载 HGraph 索引时，MCI 挂件会自动恢复。

在此 `1.0` 基础回移版本中，经过增量 `Add()` 扩展的索引重新加载后，伴随索引仍使用
原始节点数，因此搜索回退到 HGraph 遍历；向量数据仍可查询。恢复增量 MCI 的可用性
需要单独回移 #3012 的维护能力。本基础版本也不支持物理删除。

`GetStats()` 会包含以下 MCI 质量字段：

- `mci_has_index`
- `mci_total_nodes`
- `mci_covered_nodes`
- `mci_total_clique_count`
- `mci_total_membership_count`
- `mci_avg_membership_per_node`
- `mci_avg_clique_size`
- `mci_max_clique_size`
- `mci_memory_usage`

## 示例

最小构建和过滤搜索流程见
[`examples/cpp/324_feature_hgraph_mci_companion.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/324_feature_hgraph_mci_companion.cpp)。

## 可选过滤位图

自定义过滤器可通过 `GetValidBitmap(uint64_t* size)` 提供按标签索引的逐字节位图，
零表示过滤掉该标签，任意非零值表示保留。位图须在整个搜索期间保持有效且不变，
并与 `CheckValid()` 的结果一致。仅当所有有效标签均等于对应内部 ID，且位图覆盖
整个索引时，MCI 才使用此优化；否则继续调用 `CheckValid()`。现有过滤器无需修改。
搜索统计项 `mci_bitmap_fast_path` 表示是否使用了位图。浮点和量化 MCI 搜索均支持此优化。
