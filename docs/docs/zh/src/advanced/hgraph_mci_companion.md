# HGraph MCI 挂件

HGraph 可以选择构建一个 MCI（Maximal Clique Index）挂件，用于带过滤条件的 KNN
搜索。这个挂件把团信息存放在 HGraph 索引内部，并复用 HGraph 的向量存储。它不是
独立索引类型：创建索引时仍然使用 `hgraph`，不要使用 `mci`。

当主要负载是过滤搜索，并且过滤后只保留较小比例的向量时，可以启用这个功能。搜索
时，HGraph 会比较 `Filter::ValidRatio()` 和阈值，自动选择普通 HGraph 搜索或 MCI
挂件搜索。

## 构建配置

MCI 构建参数直接放在 `index_param` 下，不再使用嵌套对象。满足以下任一条件即
启用挂件：`use_mci` 为 true，或出现任意 MCI 构建参数。`mci_knng_source` 用于
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

`use_mci` 在搜索时默认为 true，可以设为 false 来仅对本次查询关闭 MCI。
`hgraph_valid_ratio_threshold` 是搜索路由阈值：`ValidRatio()` 低于该阈值时走 MCI，
否则走 HGraph。默认值是 `0.05`。

seed 数量按 `ceil(sqrt(当前向量总数) * mci_seed_ratio)` 计算，并且至少为 1。
`mci_seed_ratio` 默认值为 `0.1`，必须是有限的非负数。最终 seed 数量不会超过
满足过滤条件的点数。

当该目标不超过 `mci_seed_max_count`、也不超过向量总数时，`mci_seed_coverage` 会把
seed 数量提高到 `ceil(mci_seed_coverage * 合法点数)`；否则覆盖项整体丢弃，而不是
截断。`mci_seed_coverage` 默认 `1.0`，`mci_seed_max_count` 默认 `32768`（`0` 表示
不限）。seed 数量不会低于 1，所以播种无法被完全关闭——把两项都设为 `0` 只会留下
一个 seed。

正因为覆盖项是整体丢弃而不是钳制，当合法集超过 `mci_seed_max_count` 时
`mci_seed_coverage` 就不起作用了：这类宽过滤查询的 seed 数量会静默退回到
`ceil(sqrt(N) * mci_seed_ratio)` 这个下限。想扩大“精确播种”的适用范围就要调大
上限，代价是更长的播种阶段。默认值 `32768` 让枚举本身在绝对量级上仍然便宜——
32768 个 inner id 是 128 KiB，可以留在 cache 里——所以饱和跳过依然划算，而不会把
宽谓词变成接近全扫描规模的播种阶段。

MCI 挂件依赖过滤器提供合理的 `ValidRatio()`。bitset 和函数过滤器也可以
使用，但自定义 `Filter` 能给搜索规划提供更准确的选择率信息。

## 动态邻居遍历

`use_hybrid_traversal`（默认 `false`）把上面的“二选一”路由换成一次遍历：对每个
被展开的向量，在同一趟里同时处理它的两个邻居来源——

- 先按距离优先处理 HGraph 稀疏邻居，并把它们压入候选堆，这样无论选择率多低，
  搜索子图都保持连通；
- 再按谓词优先处理同一向量的团成员，这部分由一个虚拟开销预算提前截断。

当 `considered * (hybrid_filter_cost_ratio + local_selectivity)` 达到 `hybrid_vob`
时，团部分停止。其中 `hybrid_filter_cost_ratio` 是 `O_filter / O_dist`（一次谓词
过滤相对于一次距离计算的代价），`local_selectivity` 是当前邻居遍历中谓词的实时
命中率。`hybrid_vob` 非正时关闭提前停止。这两个参数默认值分别是 `0.0` 和 `1.0`，
`hybrid_vob` 的单位是 `O_dist`。

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

遍历的 seed 来自谓词的合法集，与 MCI 路由使用完全相同的预算公式和采样器，因此两条
路线的对比不会被 seed 策略干扰。当该预算最终覆盖了全部合法点时，所有合法距离都已
算出，遍历直接由 seed 给出答案；此时跳过展开是因为它可证明是多余的，而不是近似。
`GetStatistics()` 通过 `mci_hybrid_route`（遍历实际运行时为 `"hybrid"`）、
`hybrid_seed_budget`、`hybrid_seeded_entries`、`hybrid_expansion_skipped`、
`hybrid_expanded_nodes`、`hybrid_mci_members_considered`、
`hybrid_dist_computations` 和 `hybrid_mci_stopped_early` 报告这些行为。

## Add、序列化和统计

通过扁平构建参数启用 MCI 后，`HGraph::Add()` 会在每个新点插入 HGraph 后更新
MCI 挂件。它会先尝试加入合适的已有团；如果没有好的候选团，则为新点创建
一个小的增量团。

注意：MCI 索引不应通过在空索引上不断调用 `Add()` 从 0 开始构建。增量添加路径适用于
已有初始索引后，再追加少量向量的场景。

团数据会作为 HGraph 索引的一部分序列化。加载 HGraph 索引时，MCI 挂件会自动恢复。

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
