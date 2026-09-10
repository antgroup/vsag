# HGraph MCI 挂件

HGraph 可以选择构建一个 MCI（Maximal Clique Index）挂件，用于带过滤条件的 KNN
搜索。这个挂件把团信息存放在 HGraph 索引内部，并复用 HGraph 的向量存储。它不是
独立索引类型：创建索引时仍然使用 `hgraph`，不要使用 `mci`。

完整实现说明、内存分析及 10k/3m 结果见
[HGraph MCI 增删实现与测试报告](hgraph_mci_mutation.md)。
代码入口与配置见 [代码与配置指南](hgraph_mci_usage.md)。

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
| `mci_delete_clique_size_threshold` | 受影响团删除后的有效大小低于该值时才废弃，默认 `3`。 |
| `mci_delete_node_mct_threshold` | 废弃团中的点预计有效团数低于该值时才修复，默认 `3`。 |

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

## Add、删除、序列化和统计

通过扁平构建参数启用 MCI 后，`HGraph::Add()` 先完成 HGraph 插入批次，
再逐个更新成功插入点的 MCI。它会先尝试加入合适的已有团；
如果没有好的候选团，则为新点创建一个小的增量团。

建议先用 `Build()` 构建初始索引，再通过增量添加路径追加向量。
空索引上的 `Add()` 可以触发构建，但不建议用大量小批 Add 替代全量 Build。

ADD 需要新建团时，与全量 `BuildMCICliques` 共用局部图构建、极大团枚举和选择核心。
增量团大小上限参与决定构团门槛，不再以两个成员作为 alpha 扩张的停止标准。
优先加入已有团的快捷路径不变；高 alpha 回退仍可能生成较小的团。

`MARK_REMOVE` 会同步更新 MCI 挂件。删除一个点后，若受影响团剩余的有效成员数
不小于 `mci_delete_clique_size_threshold`，该团会被保留；只有更小的团才会被废弃。
MCI 仅收集这些废弃团中预计有效团数低于 `mci_delete_node_mct_threshold` 的有效点，
并通过与 `Add()` 相同的增量加团/构团流程修复它们，因此新增关系和团仍然复用
Add 的 delta 存储。这样无需重建完整的一跳邻居集合。
`MARK_REMOVE` 仍是默认模式：只做逻辑删除，不回收向量槽位。

启用 `index_param.support_force_remove: true` 后，可调用
`index->Remove(ids, vsag::RemoveMode::FORCE_REMOVE)` 物理删除指定向量。
MCI 模式下会自动启用图的反向边，要求使用 flat 图存储；物理删除与
`deduplicate_storage`、重复向量分组及属性倒排存储仍不兼容。
已有未启用该选项的索引需要重新构建。

物理删除先修补 HGraph 边，再用尾部向量填入删除位置，同时更新标签和图引用。
MCI 根据同一 ID 映射重建两向 CSR，保留未被指定删除的软删除标记，并通过 Add
共用的增量构团流程修复小团中覆盖不足的点，最后自动 flush 并缩小向量/图存储。
同批重复 ID 只计一次，不存在的 ID 不计数；已软删除的 ID 也可以显式物理删除。
成功后，物理槽位数减少，后续 Add 从新的尾部追加，不再积累本次删除的旧槽位。

FORCE_REMOVE 与 Add、MARK_REMOVE、Flush 串行。ID 搬移及最终缩容阶段阻塞查询；
修复阶段释放 force-remove 锁，MCI 仍未发布，查询可能回退到 HGraph。
该操作不承诺事务回滚：出错前已经完成的物理删除可能保留，未完成的 MCI 不会发布给
快速搜索。CSR 替换本身在分配成功后才提交，但操作期间新旧缓冲区会同时占用内存。
实际 RSS 还受分配器和 IO 分块大小影响，不能保证与索引统计内存同比例下降。

调用 `index->Flush()` 可将 MCI 的增量元数据合并到两向 CSR：合并新建 delta 团和
基础团追加成员，移除废弃团及已删除成员，压紧团 ID 并重建 node → clique 关系。
flush 会清空 delta，但保留向量 inner ID 和删除标记；重复执行不改变有效成员关系。
它不重建 HGraph，也不写入磁盘，持久化仍需调用 `Serialize()`。
其他索引类型以及未启用 MCI 的 HGraph 会返回不支持操作的错误。

flush 与 Add/Remove 串行执行，构建和发布新 CSR 期间持有团存储独占锁，查询可能等待。
所有替换缓冲区分配成功后才发布，分配失败不会破坏原有团数据。
临时内存需要同时容纳新旧 CSR；不要跨 flush 保存并复用团 ID。

MCI 查询只获取一次共享锁来固定 CSR、delta 和删除标记，直接遍历成员而不复制列表。
连续 FP32 向量在 Add/Remove 后、flush 前后均可直接计算距离；统计字段
`mci_raw_float_csr` 也包含直接遍历 delta 的快速查询。其他向量布局复用相同的遍历和
候选队列，通过其原有距离接口计算。删除集合在一次 MCI 查询期间固定，先检查用户
过滤条件，再检查删除集合，避免每访问一个点都获取删除集合锁。

Add/MARK_REMOVE 期间 MCI 可能暂时不可用，并发查询仍按已有语义回退到 HGraph，
不保证修改期间的完整召回；图入口点已删除时，回退结果可能为空。
单独执行 flush 不会让 MCI 暂时不可用，查询在必要时等待团存储锁。

团数据会作为 HGraph 索引的一部分序列化。加载 HGraph 索引时，
MCI 挂件会自动恢复。

`GetStats()` 会包含以下 MCI 质量字段：

- `mci_has_index`
- `mci_total_nodes`
- `mci_covered_nodes`
- `mci_total_clique_count`
- `mci_retired_clique_count`
- `mci_inactive_node_count`
- `mci_total_membership_count`
- `mci_avg_membership_per_node`
- `mci_avg_clique_size`
- `mci_max_clique_size`
- `mci_memory_usage`

## 示例

最小构建和过滤搜索流程见
[`examples/cpp/324_feature_hgraph_mci_companion.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/324_feature_hgraph_mci_companion.cpp)。

## 增删性能结果

测试方法与历史结果见[增删测试报告](hgraph_mci_mutation.md)。本 PR 不包含 benchmark 和配套脚本。
