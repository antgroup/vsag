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
| `mci_clique_max` | 全量构建时的最大团大小。 |
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

MCI 挂件依赖过滤器提供合理的 `ValidRatio()`。bitset 和函数过滤器也可以
使用，但自定义 `Filter` 能给搜索规划提供更准确的选择率信息。

## 搜索遍历与提前停止

带过滤的 MCI 搜索先用种子向量填满候选队列，然后展开已收集候选所属的团：

1. 从有效 label 中采样种子，数量由 `mci_seed_ratio` 控制；
2. 对每个尚未展开的候选，逐个遍历它所属的团，把其中"有效且未访问"的成员算距离；
3. 候选队列中没有未展开项时结束。

第 2 步正是 MCI 对高选择性过滤有价值的原因：一个团一次就能覆盖一整片邻域。但它也是浪费的来
源，因为团之间高度重叠。在 5M 向量的量化索引上（`mci_mcs=200`、`mci_clique_max=50`），一个点平
均属于 22.4 个团、团平均 102 个成员，所以当可达邻域已经算完之后，后续展开只是在走已经访问过的
成员。该索引上每次查询会展开 1.1 万–2.7 万个团，而每个团只产出 0.68 个新点。

因此 MCI 在**连续 32 个候选都没有发现新东西**（既没有新候选也没有新结果）后停止展开。同一索引、
`ef_search=600`、`mci_seed_ratio=10`：

| | recall@100 | 平均延迟 | 每次查询展开的团数 |
| --- | --- | --- | --- |
| 无提前停止 | 0.9869 | 66 ms | 1.1 万–2.7 万 |
| 有提前停止 | 0.9856 | 31 ms | 0.7 千–2.0 千 |

收益最大的是"种子已经覆盖全部有效向量"的那类查询：这些查询的 recall 完全不变，延迟下降 4–11
倍。有效集合大于种子预算的查询仍会继续展开，上表 recall 的小幅差异就来自它们。

提前停止限制的是**工作量**而不是 recall：遍历仍然没有基于距离下界的终止条件，因此 `mci_seed_ratio`
和 `ef_search` 仍是决定"探索多少有效集合"的旋钮。

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
