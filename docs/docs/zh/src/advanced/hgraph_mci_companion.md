# HGraph MCI 挂件

HGraph 可以选择构建一个 MCI（Maximal Clique Index）挂件，用于带过滤条件的 KNN
搜索。这个挂件把团信息存放在 HGraph 索引内部，并复用 HGraph 的向量存储。它不是
独立索引类型：创建索引时仍然使用 `hgraph`，不要使用 `mci`。

当主要负载是过滤搜索，并且过滤后只保留较小比例的向量时，可以启用这个功能。搜索
时，HGraph 会比较 `Filter::ValidRatio()` 和阈值，自动选择普通 HGraph 搜索或 MCI
挂件搜索。

## 构建配置

在 `index_param.mci` 下启用挂件：

```json
{
  "dtype": "float32",
  "metric_type": "l2",
  "dim": 128,
  "index_param": {
    "base_quantization_type": "fp32",
    "max_degree": 32,
    "ef_construction": 400,
    "alpha": 1.2,
    "mci": {
      "use_mci": true,
      "mcs": 200,
      "clique_max": 50,
      "alpha": 1.2,
      "seed_count": 64,
      "hgraph_valid_ratio_threshold": 0.2,
      "knng_path": "/path/to/knng_200.bin"
    }
  }
}
```

`knng_path` 是可选参数。提供时，HGraph 会基于外部 KNNG 构建团；否则会从已经构建好
的 HGraph 图和向量数据中派生 KNN 图。

| 参数 | 作用 |
| --- | --- |
| `use_mci` | 开启或关闭 MCI 挂件。 |
| `mcs` | 构建团时使用的候选邻居数量。 |
| `clique_max` | 全量构建时的最大团大小。 |
| `alpha` | 团构建扩展系数。 |
| `seed_count` | MCI 搜索默认使用的过滤 seed id 数量。 |
| `hgraph_valid_ratio_threshold` | `ValidRatio()` 低于该阈值时走 MCI，否则走 HGraph。 |
| `knng_path` | 可选的二进制 KNNG 文件。 |
| `incremental_join_ratio_threshold` | Add 时加入已有团的阈值。 |
| `incremental_added_mct` | 新节点最多加入的已有团数量。 |
| `incremental_clique_max` | 增量创建新团时的最大团大小。 |

## 搜索配置

搜索参数放在 `hgraph` 搜索对象下：

```json
{
  "hgraph": {
    "ef_search": 120,
    "use_mci": true,
    "seed_ratio": 0.002,
    "hgraph_valid_ratio_threshold": 0.2
  }
}
```

`seed_ratio` 会乘以当前向量数量得到 seed 数量。如果同时设置 `seed_count` 和
`seed_ratio`，优先使用 `seed_ratio`。

MCI 挂件依赖过滤器提供合理的 `ValidRatio()`。bitset 和函数过滤器也可以使用，但自
定义 `Filter` 能给搜索规划提供更准确的选择率信息。

## Add、序列化和统计

启用 `use_mci` 后，`HGraph::Add()` 会在每个新点插入 HGraph 后更新 MCI 挂件。它会先
尝试加入合适的已有团；如果没有好的候选团，则为新点创建一个小的增量团。

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
