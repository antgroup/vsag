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

浮点快速路径和通用全量构建路径共用最终补覆盖流程。极大团枚举结束后，仍未被覆盖的点
会与图邻居组成兜底团；没有可用邻居时使用单点团。此流程放宽团内距离约束，按团大小上限
截断时保留当前点，并且只统计实际保存的成员关系。

| 参数 | 作用 |
| --- | --- |
| `use_mci` | 设为 `true` 时使用默认构建参数启用 MCI。 |
| `mci_mcs` | 构建团时使用的候选邻居数量。 |
| `mci_clique_max` | 全量构建时的最大团大小。 |
| `mci_knng_source` | KNN 图来源：`hgraph`（默认）或 `odescent`。 |
| `mci_alpha` | 团构建扩展系数。 |

### 增量维护参数

以下参数仍放在 `index_param` 下，控制 ADD 以及删除后存活点的修复。
删除修复复用 ADD 的加入已有团和新建团逻辑，不会将已有向量重新插入 HGraph。

这里的度数是覆盖当前点的所有有效团中，其他存活成员的去重并集大小，
不等于 HGraph 出度，也不等于当前点所属的团数。度数停止目标为：

```cpp
target = N <= 1 ? 0 :
    min(N - 1,
        max(mci_incremental_degree_min,
            min(N / mci_incremental_degree_n_divisor,
                mcs / mci_incremental_degree_mcs_divisor)));
```

`N` 为当前存活向量数，包含已完成图插入的 ADD 批次，不包含已标记删除的点；
`mcs` 对应 `mci_mcs`，除法向下取整。这是停止目标，而非保证达到的最低度数或硬性上限：
加入一个完整团可能超过目标；候选耗尽或构团无进展时，也可能在目标以下停止。

| 参数 | 默认值与取值范围 | 含义 |
| --- | --- | --- |
| `mci_incremental_join_ratio_threshold` | 默认 `0.6`；范围 `[0, 1]`。 | 加入已有团的重叠比例阈值，比例为当前点的 KNN 候选与团有效成员的交集数量，除以该团有效成员数量。例如团有 10 个有效成员，其中 6 个在 KNN 候选中，比例为 `0.6`：阈值 `0.6` 时满足条件，`0.7` 时不满足。还需满足团未满、加入后能增加新邻居等条件；该比例不是与每个成员逐一验证距离约束。降低阈值会放宽加入条件，提高阈值则更严格，可能使更多度数缺口依赖新建团补足。可比较 `0.5 / 0.6 / 0.7`，同时观察 recall、ADD 耗时和成员关系总量。 |
| `mci_incremental_degree_min` | 默认 `50`；正整数。 | 度数目标公式中的下限，最终仍受 `N-1` 限制。例如 `N=10000、mcs=200` 时，默认目标为 `50`；将本参数改为 `70`，目标变为 `70`。但 `N≈3m、mcs=200` 时，下限为 `50` 或 `70`，目标都是 `100`。低度数点较多时可尝试提高下限，验证 recall 是否改善；降低下限可减少达到目标所需的维护工作，但可能减少搜索连接。调参前应先确认本参数是否实际决定目标。 |
| `mci_incremental_degree_n_divisor` | 默认 `10000`；正整数。 | 控制规模项 `N / divisor`。例如 `N=800000、mcs=200、degree_min=50` 时，默认目标为 `80`；将除数改为 `20000`，规模项变为 `40`，最终目标受下限约束，为 `50`。减小除数会提高规模项，增大除数会降低规模项，适合调整目标随数据规模增长的速度；效果仍受度数下限和 MCS 项限制，被其他项限制时可能没有变化。 |
| `mci_incremental_degree_mcs_divisor` | 默认 `2`；正整数。 | 控制候选规模项 `mcs / divisor`，与规模项取较小值后，再应用度数下限。例如 `N≈3m、mcs=200、degree_min=50` 时，除数为 `2`，目标为 `100`；改为 `4`，目标为 `50`。减小除数可以提高目标，增大除数可以降低目标，但仍受规模项和下限限制。本参数不改变 KNN 候选数量；要改变候选规模，应调整 `mci_mcs`。 |
| `mci_incremental_clique_max` | 默认 `50`；整数且不小于 `2`。 | 同时限制增量新建团的大小和向已有团追加成员后的大小。例如设为 `50` 时，49 人团可以追加到 50 人；已有 50 人团不会再接受新点。因此，全量构建得到的满 50 人团在默认设置下不能直接追加。实际新建团可能小于上限。提高上限可让部分原本已满的团继续追加，并允许更大的新团；降低上限可能需要更多团补足度数。应结合 `mci_clique_max`、成员关系总量和查询开销一起评估。 |
| `mci_delete_clique_size_threshold` | 默认 `30`；正整数。 | 对本批删除影响到的团，统计整批删除后剩余的有效成员数，严格小于阈值才废弃整个团：默认剩 29 个成员时废弃，剩 30 个时保留。废弃的是团及其成员关系，不是剩余存活向量；也不会扫描废弃无关的小团。这不是构建团大小上限。提高阈值会扩大废弃范围，可能增加修复成本和结构变化；降低阈值会保留更多小团。可围绕 `30` 按 `10` 小步增减，对比废弃团数、修复点数、删除耗时和 recall，不假定越大越好。若构建团大小上限低于此阈值，删除触及的这类团都会被废弃，应配套评估。 |
| `mci_delete_node_mct_threshold` | 默认 `3`；正整数。 | 从被废弃团的存活成员中，筛选预计有效覆盖团数严格小于阈值的点作为修复候选。统计时扣除本批即将废弃的团，并对候选点去重。例如默认剩余覆盖团数为 0、1、2 时进入候选，为 3 时不进入；这里统计的是团数，不是邻居度数。提高阈值会扩大候选范围，降低阈值会缩小范围；设为 `1` 时只选择预计失去全部团覆盖的候选点。可按 `3 → 4 → 5 → 6` 比较修复成本和查询质量。实际修复前会再次检查覆盖情况，进入候选不代表一定新建团。 |

以上调参方向来自代码机制，不是已验证的最优参数。建议固定数据阶段、增删 ID、
ground truth 和其他配置，每次只调整一项，同时比较查询质量、吞吐、增删耗时及内存。
例如 `N≈3m、mcs=100、degree_min=50` 时，默认目标为 `50`；仅修改存活点数除数，
只要规模项仍不低于 50，目标就不会改变。
显式配置仍优先于默认值，加载已有索引需要匹配其序列化参数；历史实验中的阈值 3
不代表当前删除团大小阈值 30 的性能结果。

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
如果去重邻居度数仍不足，则继续为新点创建增量团。

建议先用 `Build()` 构建初始索引，再通过增量添加路径追加向量。
空索引上的 `Add()` 可以触发构建，但不建议用大量小批 Add 替代全量 Build。

ADD 需要新建团时，与全量 `BuildMCICliques` 共用局部图构建、极大团枚举和选择核心。
增量团大小上限参与决定构团门槛，不再以两个成员作为 alpha 扩张的停止标准。
JOIN 与构团使用上述增量维护参数计算去重邻居度数停止目标；达到目标、候选耗尽或
构团无进展时停止。高 alpha 回退仍可能生成较小的团。

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
