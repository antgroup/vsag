# HGraph MCI 增删实现与测试报告

本文记录当前工作树中 MCI 的 ADD、MARK_REMOVE、FORCE_REMOVE、Flush 实现，
以及 2026-09-07 的 Codefilter 10k、3m 测试结果。
参数入门见 [HGraph MCI 挂件](hgraph_mci_companion.md)。

## 1. 结论与适用范围

- ADD 与删除后的团修复共用增量构团逻辑及三类 delta 存储。
- 删除采用“小团才废弃、覆盖不足点才修复”的选择性策略，
  不再废弃所有包含被删点的团，也不重建全部一跳邻居。
- MARK_REMOVE 保留向量槽位；FORCE_REMOVE 修复图、搬移尾点、重映射 MCI，
  并缩小存储。Flush 只合并团元数据，不等价于物理删除向量。
- 3m 数据集实际包含 3,241,378 点。删除约 20% 后，
  索引统计内存从 6,323.25 降至 5,215.30 MiB，减少 17.52%。
- 五阶段的 FORCE_REMOVE 测试均满足物理点数正确、存活点全部被 MCI 覆盖、
  无残留删除标记、直接 FP32 快速路径比例为 100%。
- 全部加回后索引内存为 6,411.62 MiB，略高于初始值；
  检索速度和召回率同时发生变化，不能只比较相同 ef 下的 QPS。

本报告对应分支 `feat/mci-delete` 的未提交实现，基线提交为
`f41d6cd143380886ba193f9cb6d7864ba1c790f8`。
仅检出该提交不能复现本文新增行为，需要包含本次工作树修改。

**测试使用 HGraph 的 `graph_type=nsw`，即单层 NSW + MCI，
不是单独的 `hnsw` 索引，也不是多层 HNSW 配置。**

## 2. 索引构建与存储布局

### 2.1 构建流程

先构建 HGraph 并存储向量，再构建 MCI 伴随结构：

```text
输入向量
  → HGraph 图与向量存储
  → MCI 候选 KNN 图
  → 团构建与覆盖选择
  → clique → node CSR + node → clique CSR
```

`mci_knng_source=hgraph` 从现有 HGraph 生成构团候选；
`odescent` 使用单独构建的候选图。外部 KNN 图文件有更高优先级。
连续 FP32 数据走 `BuildMCICliques` 构建路径，其他布局有通用构建路径。
MCI 复用 HGraph 向量存储，不再保存一份完整向量数据。

这里的“团”是受候选集、距离扩展及大小上限约束的搜索组织结构。
不能把当前增量算法理解为对全局固定图枚举所有严格极大团，
也不能把“全部节点被覆盖”理解为“任意查询都达到精确召回”。

### 2.2 两向 CSR 与增量层

| 结构 | 内容 |
| --- | --- |
| `p_maxc_`、`maxcs_` | 基础团到节点的 CSR 偏移和成员数组。 |
| `p_node_to_cid_`、`node_to_cids_` | 节点到基础团的 CSR 偏移和团 ID 数组。 |
| `delta_cliques_` | 新建团的完整成员。 |
| `delta_clique_extra_` | 追加到基础团的成员，不修改基础 CSR。 |
| `delta_node_to_cids_` | 增量产生的 node → clique 关系。 |
| `inactive_nodes_` | 不可用节点标记，主要对应逻辑删除。 |
| `retired_cliques_` | 已废弃团标记。 |

设基础团数量为 `B`。`cid < B` 时，团成员来自基础 CSR 与 extra 的并集；
`cid >= B` 时，成员来自 `delta_cliques_[cid - B]`。
节点所属团来自基础反向 CSR 和 `delta_node_to_cids_`，并排除废弃团。
所有有效成员查询都排除 inactive 节点。

这使得“Add 后立即 Delete”“Delete 修复出新团后再次 Add/Delete”
都能看到统一的有效团视图，而不是只读取最初构建的 CSR。

需要区分三种 ID：外部 label、内部向量 inner ID、团 ID。
FORCE_REMOVE 可以改变 inner ID，Flush 可以改变团 ID；
外部 label 是调用者使用的身份，不应缓存内部 ID 作为稳定句柄。

## 3. ADD 实现策略

### 3.1 批次与发布顺序

`HGraph::Add()` 的主要顺序是：

1. 获取 MCI 修改互斥锁，校验输入并准备向量/图插入批次。
2. 若已有 MCI，将其标记为暂时不可用。
3. 执行 `insert_add_batch`，先把成功行插入 HGraph。
4. 对成功行逐个执行 `incremental_update_mci_clique`。
5. 全部增量更新完成后发布新的 MCI 总点数，并更新内存统计。

因此，批量 ADD 不是每插入一个图节点就立即发布一份新 MCI；
它先执行图插入批次，再逐行更新 MCI。图插入可使用构建线程池，
当前这一层增量 MCI 更新循环是串行的。

若原先没有可用 MCI，则走构建 MCI 的分支；force-enabled 索引还会重新应用
仍保留的软删除标记。支持空索引恢复并不意味着建议用大量小批 ADD
替代一次全量 Build。

### 3.2 获取候选邻居

新点 `u` 通过 `incremental_update_mci_clique(u, vector)` 调用 `search_mci_knn` 获取候选：

- 通过普通 HGraph 查询生成候选，显式指定 `use_mci=false`，避免递归依赖 MCI。
- 目标数量最多为 `mci_mcs`；去掉自身及不在本轮可见 inner ID 范围的点。
- 同批新点按当前 inner ID 的可见边界处理，不把所有后续批内点直接视为已处理点。
- 候选不足时扩大返回数量；搜索 ef 至少为 100。
- 去重后，如候选超过目标数量，按精确向量距离排序截断。

这是近似候选生成，不是扫描全部向量的精确 KNN。
批次大小、ID 分配和插入顺序可能影响最终团结构及召回率。

### 3.3 优先加入已有团

设 `K(u)` 为候选邻居集合，`C` 为有效候选团。
从邻居的所属团关系中统计重叠，计算：

```text
join_ratio(u, C) = |K(u) ∩ C| / |C 的有效成员|
```

仅考虑非空且未达到增量大小上限的团。
比例不小于 `mci_incremental_join_ratio_threshold` 时进入候选列表。
默认阈值为 `0.6`；按重叠成员数降序、团 ID 升序排序，
最多尝试 `mci_incremental_added_mct` 个团，默认 `3`。

成功追加时：

- 基础团：写 `delta_clique_extra_[cid]`。
- 增量团：直接追加到 `delta_cliques_[cid - B]` 的完整成员。
- 两者都写入该节点的 `delta_node_to_cids_`。

追加接口排除无效节点、废弃团、重复成员和超限团。
只要至少成功加入一个已有团，本轮就不再创建新团。
`added_mct=3` 是最多尝试加入的团数，不保证最终拥有三个团。
此分支按重叠比例选择团，不重新验证全团所有成员对的连通关系。

### 3.4 无合适团时新建增量团

`build_incremental_mci_clique` 以 `u` 为起点：

1. 按到 `u` 的距离排序候选邻居。
2. 根据最近邻距离、metric 和 alpha 得到扩张距离界限。
3. 贪心加入在界限内、且与已选成员距离也满足界限的候选。
4. 未达到目标最小规模时继续扩大 alpha，并受终止条件与大小上限约束。
5. 无候选时建立单点团；有候选但构团不足时可退化为自身和最近邻组成的二点团。
6. 通过 `AppendNewClique` 写完整成员，并为所有成员写反向 delta 关系。

该过程强调局部组织质量和覆盖，不承诺找到全局最大团或唯一极大团。

## 4. 删除的共同策略：小团触发、低覆盖点修复

### 4.1 与旧方案的区别

旧思路是收集被删点的一跳团邻居，废弃所有包含它的团，
再重建直到全部邻居被覆盖。当前实现不再采用这个全量局部重建策略。

当前只在团变得过小时修复。两个阈值均为严格“小于”判断：

- `T_size = mci_delete_clique_size_threshold`，默认 `3`。
- `T_mct = mci_delete_node_mct_threshold`，默认 `3`。

### 4.2 快照与候选筛选

`PrepareDelete(D, T_size, T_mct)` 不直接修改数据，计算：

1. `affected`：覆盖删除集合 `D` 的有效团，包含 base、extra 和 delta。
2. `retired`：这些团去掉 `D` 后，有效成员数小于 `T_size` 的团。
3. `candidates`：`retired` 中未被删除的成员，去重。
4. `repair`：候选点中，排除本轮废弃团后预计所属有效团数小于 `T_mct` 的点。

`CommitDelete` 随后设置节点 inactive 和团 retired 标记。
真正修复某点之前再次检查它是否存活、当前覆盖团数是否仍低于阈值，
避免前面构出的团已经覆盖它时重复修复。

### 4.3 “调用 ADD”的准确含义

纯 FP32 配置（float32 输入，底层和 precise 编码均为 FP32）的删除修复调用的是：

```text
repair_mci_clique(v)：获取已有点的查询向量
  → incremental_update_mci_clique(v, vector, total)
  → search_mci_knn(v, vector, total)：HGraph 搜索，use_mci=false
  → incremental_update_mci_clique(v, knn_ids, total)
  → try_join_mci_clique / build_incremental_mci_clique
```

**不是对已有点调用公开 `Index::Add()`，不会再次插入它的向量或 label。**
修复结果与 ADD 共用 delta，后续查询和修改都能看到这些关系。

修复现在与 ADD 共享从 KNN 搜索到 delta 更新的完整 MCI 流程，不再逐修复点
扫描全部存活向量并全排序。KNN 上限仍为 `mci_mcs`，初始请求 `k+1`（排除自身），
`ef_search=max(query_k,100)`，过滤后候选不足时沿用 ADD 的扩大请求机制。
这是近似图搜索，不保证精确 KNN，也不意味着删除修复已并行化。

已有点优先使用连续 FP32 数据；对于 block_memory_io 等非连续存储，读取该点的
FP32 编码并恢复查询向量，然后进入同一流程。
本次不修改量化和其他 dtype 的候选生成：这些配置仍保留原有全扫描修复路径。

新点 ADD 保留插入前缀的候选范围；已有 FP32 修复点使用整个当前索引，搜索过滤已删除点
并排除自身。共享建团函数也使用显式传入的范围，不再用旧点 `node_id + 1`
限制修复团的大小，避免低 ID 点只能形成过小的团。

FORCE_REMOVE 在底层图修复和 ID 重映射完成后释放物理删除独占锁，再调用共享搜索；
增删串行锁仍持有、MCI 仍暂不可用。搜索自行获取读锁，修复后重新获取独占锁再
Flush 和收缩内存，避免递归获取同一把锁。MARK_REMOVE 保持逻辑删除语义。

本文历史性能结果来自这次统一流程之前的实现：旧版修复逐点全扫描和全排序，
约为 `O(R × (N × d + N log N))`。历史耗时和 recall 不代表修改后的表现，需重新测试。

### 4.4 质量边界

`T_mct` 是修复触发阈值，不是修复后必须达到的覆盖下限。
成功加入一个团即可结束一次共享增量更新。
现有实现没有重新统计全部一跳点度数、没有检查所有大团的连通分量，
也没有证明删点后全局图始终连通。
FORCE_REMOVE 的图边修复与 MCI 团覆盖修复是两个不同层面的操作。

## 5. MARK_REMOVE：保持逻辑删除语义

调用 `Remove(ids)` 或显式指定 `MARK_REMOVE`：

1. 在 label 表中标记待删除点，更新有效点数和删除计数。
2. 把 MCI 标为暂时不可用。
3. 对去重后的待删除 inner ID 逐个执行快照、提交和选择性修复。
4. 发布更新后的 MCI，并更新内存统计。

不搬移向量，不缩小物理槽位，不执行 FORCE_REMOVE 的图边修补。
基础 CSR 中可以继续保存过期关系，由标记过滤；修复结果进入 delta。
后续 Add 通常从物理尾部追加，不会因为逻辑删除就自动回收旧槽位。

因此，`GetNumElements()` 减少不代表物理向量数量或内存已经减少。
本报告的 `index_elements` 是有效点数，`mci_total_nodes` 才反映对应物理槽位规模。

## 6. FORCE_REMOVE：物理删除并缩容

### 6.1 启用条件与 API

构建时设置 `index_param.support_force_remove=true`，再调用：

```cpp
auto result = index->Remove(ids, vsag::RemoveMode::FORCE_REMOVE);
if (!result.has_value()) {
    // 检查 result.error()；不要假定操作已自动回滚。
}
```

MCI + FORCE_REMOVE 要求 flat 图存储，并自动启用底层图及上层图参数的反向边。
这里 flat 指图存储类型，不等同于是否使用多层拓扑。
不支持 `deduplicate_storage`、重复向量分组或属性倒排存储的组合。
测试中的自定义 label 过滤器不是内置属性倒排存储。
未启用物理删除能力的已有索引需要重建。

### 6.2 批次算法

1. 获取 MCI 修改互斥锁，再获取物理删除独占锁，阻塞并发查询。
2. 按外部 label 解析目标，允许显式删除已软删除点。
   重复目标去重，不存在的目标不计入删除数量。
3. 按原始 inner ID 降序排列；预分配 `old_to_new` 与 `current_to_old` 映射。
4. 对整个目标集合调用 `PrepareDelete`，再设置 MCI 不可用并提交删除标记。
5. 逐点调用 `force_remove_one`，修复图并把尾点搬到删除位置。
6. 更新映射；删除点映射为无效 ID，存活点映射到压紧后的 `[0, N')`。
7. `RemapNodes` 重建 MCI 两向 CSR，同时重映射未指定删除的软删除标记。
8. 将修复点映射到新 ID，按需执行共享的增量团修复。
9. 自动 `Flush`，把这些修复结果也合并进 CSR。
10. 缩小向量、图及有关逐点存储，更新容量边界，再发布 MCI 和内存统计。

降序删除保证尾点搬移不会使尚未处理的目标离开它的原始位置。
例如删除 ID 2、7（总点数 10），可先用尾点 9 填入 7，
再用当前尾点 8 填入 2，最终物理范围为 `[0, 8)`。
外部 label 跟随实际向量更新，而非跟随旧槽位保留。

MARK_REMOVE 对 MCI 逐点计算快照，FORCE_REMOVE 对同一调用的目标集合计算快照；
两种方式及不同批次大小可能产生不同修复顺序和团结构。

### 6.3 图修补与尾点搬移

图修补收集被删点的出邻居和入邻居。
对受影响邻居，将它原有邻居与被删点的出邻居合并为候选，
排除被删点、自身及越界引用，再通过原有选边启发式限制最大度数。
最后清空被删点的邻接关系。

`move_id` 同步移动向量编码、单独持有的原始向量、精排编码、附加信息、
底层/上层图引用、label 映射及入口点信息。
label 表还处理软删除、同 label 再次 Add 后的旧槽位、可选 source ID 等情况，
避免搬移旧墓碑覆盖同 label 当前存活实例的映射。

### 6.4 回收范围与失败语义

成功后，指定点的物理槽位消失；不在本次目标中的其他软删除槽位仍被保留。
后续 ADD 从缩小后的尾部开始，增长时仍可能按容量粒度预留额外空间。

该操作不是事务：出错前已完成的向量删除、图修补或搬移可能保留。
MCI 完整可用前不重新发布快速视图，不能承诺异常时保持原始索引不变。
两向 CSR 的单次替换会先完成新缓冲分配再交换，但这不使整个 FORCE_REMOVE 原子化。
重映射期间需要同时容纳新旧缓冲，峰值内存可能先上升再下降。

## 7. Flush、直接搜索与锁

### 7.1 Flush 的职责

`index->Flush()` 在内存中执行 MCI 元数据压紧：

1. 枚举基础团及增量团，合并基础成员和 extra。
2. 排除废弃团、inactive 成员和空团。
3. 重新编号有效团，生成两向 CSR。
4. 所有新缓冲分配成功后再交换，清空 delta 和 retired 内容。

普通 Flush 保留向量 inner ID 和 inactive 标记，不重建 HGraph、不删除向量、
不写磁盘；持久化仍需 Serialize。重复 Flush 不改变有效成员关系，
但不能跨 Flush 保留并复用团 ID。
清空 delta 也不意味着元数据零开销：仍需要与点数/基础团数对应的空行描述符。

FORCE_REMOVE 的 `RemapNodes` 与 Flush 共用 CSR 压紧核心，前者额外应用节点映射。
FORCE_REMOVE 会自动再 Flush 一次修复结果；ADD 和 MARK_REMOVE 不自动 Flush。

### 7.2 查询路径

过滤查询根据 `ValidRatio()` 和路由阈值选择 HGraph 或 MCI。
MCI 查询的有效邻接视图直接由以下部分组成：

```text
节点 → base 反向 CSR + delta_node_to_cids_ → 排除 retired 团
团   → base 正向 CSR + extra，或 delta_cliques_ → 排除 inactive 点
候选 → 用户过滤条件 → label 删除集合 → 距离计算与候选队列
```

`CliqueDataCellSearchView` 用一次共享锁固定查询使用的 CSR、delta 和标记，
避免每访问一团就复制成员列表或再次获取存储锁。
连续 FP32 向量直接计算距离，其他编码使用原距离接口；
两者共用遍历和候选队列，不因 delta 非空就放弃快速路径。
名为 `mci_raw_float_csr` 的统计也包含直接遍历 delta 的 FP32 查询。

label 删除集合通过一次查询级只读视图固定，不再为每个访问节点获取删除集合锁。
先检查用户过滤条件再检查删除集合，减少低选择率过滤中的额外工作。
这不是无锁搜索，而是缩小锁获取频率和消除部分临时复制。

### 7.3 并发语义

| 保护机制 | 用途 |
| --- | --- |
| `mci_mutation_mutex_` | 串行化 MCI 的 Add、两种 Remove 和 Flush。 |
| `force_remove_mutex_` | 查询共享持有，物理删除独占持有，固定 inner ID 生命周期。 |
| `persistent_codes_mutex_` | 防止直接距离计算期间向量缓冲被移动或缩容。 |
| 团存储共享/独占锁 | 固定查询视图，保护增量修改与 CSR 替换。 |
| label 表及删除集合锁 | 保护身份映射、删除状态及查询级删除视图。 |

FORCE_REMOVE 的主要顺序为修改互斥锁 → 物理删除独占锁 → 局部存储锁。
ADD 初期还持有物理删除共享锁；进入调用公开搜索的 MCI 候选阶段前会释放该共享锁，
仍保留修改互斥锁排除物理删除，避免同线程递归获取共享锁。

Add/MARK_REMOVE 期间 MCI 可能暂时不可用，查询可以回退到 HGraph，
不保证修改期间的完整召回；回退入口点失效时甚至可能返回空结果。
FORCE_REMOVE 阻塞查询直至搬移及修复结束。
Flush 不主动使 MCI 不可用，但查询可能等待 CSR 替换的独占锁。
以上不是跨批次事务或任意并发操作下的快照一致性承诺。

### 7.4 序列化

MCI 的基础 CSR、delta 和删除相关状态随 HGraph 序列化。
force-enabled MCI 加载时，会利用保存的 inactive 状态恢复 label 删除集合和计数，
按实际物理点数校验范围，并避免旧软删除槽位遮蔽同 label 的存活实例。
不要把 Flush 当成 Serialize，也不要只保存一份 CSR 就认为包含了全部增量状态。

## 8. 内存口径与 128 MiB 分块现象

| 指标 | 含义 |
| --- | --- |
| `index_memory_bytes` | `GetMemoryUsage()` 汇总的索引分配容量，不是进程 RSS。 |
| `vector_memory_bytes` | 本次 FP32 测试的 `basic_flatten_codes` 容量。 |
| `graph_memory_bytes` | bottom graph 与 route graph 的统计内存，包含相关图元数据。 |
| `mci_memory_bytes` | MCI 数组、增量行和标记等的容量统计。 |
| `stage_rss_bytes` | 阶段查询末尾采样的进程驻留内存。 |

索引总量还包含 label 表、锁等其他结构，因此不严格等于上面三项之和。
`vector_memory_bytes` 不是适用于所有量化、精排或独立 raw-vector 配置的向量总内存。
本文所有 MiB 均为字节数除以 `2^20`。

本轮向量使用 `memory_io`，图保留默认 `block_memory_io`。
默认内存块为 128 MiB，计费为 `块数 × 块大小`，Shrink 只能释放完整块。
因此 10k 和 8k 点的图都仍需要保留至少一个块，掩盖了向量缩容效果。
这首先是实际分配容量问题，不应只归因于操作系统没有退还 RSS。

大数据集更容易跨越整块释放边界，但图修补也会改变反向边、容器容量等。
所以图统计可能在某个删除阶段上升；Add 后同样点数也不保证回到初始图内存。
RSS 另外包含完整输入 HDF5 数据、查询过滤器、临时缓冲和分配器保留页。
回收了索引内存，不代表 RSS 在同一时刻等量下降。

## 9. 测试设计与复现

### 9.1 环境和共同参数

测试日期为 2026-09-07，使用 Release 构建。
机器为 Intel Xeon Platinum 8269CY，52 核/104 逻辑 CPU，约 754 GiB 内存。
构建和检索各设置 16 线程；不把机器总核数当作实际检索线程数。

| 参数 | 值 |
| --- | --- |
| dtype / metric / dim | float32 / cosine / 384 |
| 图配置 | HGraph，`graph_type=nsw`，max_degree=32，ef_construction=200 |
| 向量 / 图 IO | memory_io / 默认 block_memory_io |
| MCI 候选 | mcs=50，knng_source=hgraph |
| MCI 大小 / alpha | clique_max=50，incremental_clique_max=50，alpha=1.2 |
| 增量加入 | ratio_threshold=0.6，added_mct=3（默认） |
| 删除阈值 | clique_size=3，node_mct=3 |
| 查询 | top-k=10，200 条查询，ef=40/80/160 |
| QPS 计时 | 每个 ef 执行 50,000 次查询，循环使用上述 200 条查询 |
| 预热 / seed | 预热 64 次，mci_seed_ratio=0.1 |
| 路由 / 删除随机种子 | route_threshold=1.0，random_seed=20260907 |
| Flush | 不额外调用；仅 FORCE_REMOVE 内部自动压紧 |

单次运行，不是多次重复的中位数，也没有置信区间。
同一批 200 条查询重复执行，不能等同于 50,000 条不同查询或冷缓存吞吐。

### 9.2 数据集、过滤与阶段

程序先将 HDF5 数据加载到内存，再全量 Build，不使用 `--max-base` 截取。
按 `train_labels == test_labels` 做过滤。
3m 文件有 1,278 条测试查询，本轮选择 200 条；10k 文件的过滤真值被重新精确计算，
3m 文件使用通过范围/标签校验的提供真值。

为了保证五阶段沿用同一份 top-k 真值，删除候选排除所测查询的真值点，
其余点用固定种子选取。这不是允许真值点被删除的完全随机删除实验。
加回的是同一批被删除向量及外部 ID。

| 阶段 | 操作 | 10k 存活点 | 3m 存活点 |
| --- | --- | ---: | ---: |
| S0 | 全量 Build | 10,000 | 3,241,378 |
| S1 | 删除原始数据的约 10% | 9,000 | 2,917,240 |
| S2 | 再删除原始数据的约 10% | 8,000 | 2,593,102 |
| S3 | 加回第一批 | 9,000 | 2,917,240 |
| S4 | 加回第二批 | 10,000 | 3,241,378 |

3m 每阶段操作 324,138 点，是原始点数的四舍五入 10%，
不是对当时剩余点数再乘 10%。
10k 使用每次 API 调用 10 点；3m 使用整批 324,138 点。
批次差异会改变压紧次数、修复决策和结构，不能把两者作为严格的规模延迟对照。

10k 比较 FORCE_REMOVE 与 MARK_REMOVE，两者独立构建；
force 开启反向边，初始内存已经不同，应优先比较各自运行内的变化。
3m 只测 FORCE_REMOVE，没有 3m MARK_REMOVE 或额外 Flush 对照。

### 9.3 复现命令

在仓库根目录执行。脚本默认构建 Release 的 `mci_mutation_benchmark`；
已有匹配当前源码的二进制时可设置 `MCI_SKIP_BUILD=1`。

```bash
MCI_DATASET_PATH=/root/data/codefilter-3m-384-angular-f32.hdf5 \
MCI_RESULT_DIR=/tmp/mci-3m-reproduce \
bash scripts/perf_reports/run_hgraph_mci_mutation.sh \
    --force-remove --mutation-batch-size 324138 \
    --query-count 200 --search-count 50000 \
    --build-threads 16 --search-threads 16 --ef-search-values 40,80,160
```

```bash
MCI_DATASET_PATH=/root/data/codefilter-10k-384-angular-f32.hdf5 \
MCI_RESULT_DIR=/tmp/mci-10k-force-reproduce \
bash scripts/perf_reports/run_hgraph_mci_mutation.sh \
    --force-remove --mutation-batch-size 10 --search-count 50000
```

```bash
MCI_DATASET_PATH=/root/data/codefilter-10k-384-angular-f32.hdf5 \
MCI_RESULT_DIR=/tmp/mci-10k-mark-reproduce \
bash scripts/perf_reports/run_hgraph_mci_mutation.sh \
    --mutation-batch-size 10 --search-count 50000
```

上述命令复现基准程序指标；额外 `stage_rss_bytes` 来自进程外部采样，
不是原 C++ CSV 自动生成的列。
3m 每 100 ms 采样一次，以 ef=160 阶段末尾约 1 秒的 RSS 中位数作为阶段值；
10k 每约 5 ms 采样，以末尾约 200 ms 中位数作为阶段值。
未强制执行分配器 trim，采样峰值也不保证捕获每个瞬间的分配峰值。

## 10. 测试结果

### 10.1 3m：索引与向量内存

| 阶段 | 物理点数 | 索引 MiB | 向量 MiB | 图 MiB | MCI MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 3,241,378 | 6,323.25 | 4,749.00 | 926.83 | 349.19 |
| S1 | 2,917,240 | 5,692.63 | 4,273.30 | 862.39 | 262.76 |
| S2 | 2,593,102 | 5,215.30 | 3,798.49 | 900.01 | 229.55 |
| S3 | 2,917,240 | 5,762.74 | 4,273.50 | 905.89 | 321.77 |
| S4 | 3,241,378 | 6,411.62 | 4,749.00 | 1,043.43 | 332.09 |

S2 相比 S0：索引减少 **1,107.94 MiB（17.52%）**。
删除向量有效载荷约 949.62 MiB，统计向量容量减少约 950.51 MiB；
两者之差主要是初始预留容量也被缩小。

S4 与 S0 点数相同，向量容量同为 4,749.00 MiB；
索引多约 88.37 MiB，图多约 116.60 MiB，MCI 反而少约 17.10 MiB。
因此剩余差异主要落在图等结构，不能解释成被删除向量仍在累积。

### 10.2 3m：RSS、耗时与 delta

| 阶段 | RSS MiB | 本阶段增删秒数 | 总团数 | delta 新团数 |
| --- | ---: | ---: | ---: | ---: |
| S0 | 12,301.81 | — | 408,484 | 0 |
| S1 | 11,697.05 | 393.30 | 408,424 | 0 |
| S2 | 11,210.89 | 338.56 | 408,364 | 0 |
| S3 | 11,709.60 | 551.62 | 523,628 | 115,264 |
| S4 | 12,353.88 | 513.30 | 601,107 | 192,743 |

全量 Build 用时 593.99 秒；全流程 2,572.34 秒，约 42.9 分钟。
采样到的进程峰值 RSS 为 13,088.96 MiB。
全部阶段的有效点数、物理点数和 MCI 覆盖点数相等，inactive 标记数均为 0。
delta 新团数不包括基础团 extra 成员，因此不能作为全部 delta 工作量的替代指标。

第一批删除及第一批 ADD 各做过一次短暂的只读调试器附加，
停顿包含在对应增删耗时中；计时检索期间未附加调试器。
抽样分别看到图邻接修复和 ADD 的 MCI 候选 KNN 查询。
硬件 perf 采样因权限不可用，故这些结果不是完整 CPU 时间占比分析。

### 10.3 3m：完整 QPS–recall@10 数据

| 阶段 | ef_search | QPS | Recall@10 |
| --- | ---: | ---: | ---: |
| S0 | 40 | 10,030.73 | 84.90% |
| S0 | 80 | 6,540.38 | 88.90% |
| S0 | 160 | 4,004.39 | 92.25% |
| S1 | 40 | 10,369.07 | 84.30% |
| S1 | 80 | 6,988.16 | 89.65% |
| S1 | 160 | 4,524.68 | 92.50% |
| S2 | 40 | 12,228.89 | 84.15% |
| S2 | 80 | 8,140.06 | 88.75% |
| S2 | 160 | 5,222.28 | 92.35% |
| S3 | 40 | 12,238.82 | 84.25% |
| S3 | 80 | 7,897.30 | 88.75% |
| S3 | 160 | 5,006.40 | 92.65% |
| S4 | 40 | 12,308.17 | 84.10% |
| S4 | 80 | 8,294.05 | 87.80% |
| S4 | 160 | 5,345.38 | 90.90% |

15 个点的 MCI 路由和直接 FP32 路径比例均为 100%。
这表明存在 delta 时仍能走快速搜索，不证明 delta 没有任何访问成本。
S4 的 ef=80 召回比 S0 低 1.10 个百分点，ef=160 低 1.35 个百分点。
不能仅凭 QPS 上升宣称等召回性能提高；需要更密的 ef 扫描和独立重复实验。

### 10.4 10k：FORCE_REMOVE 与 MARK_REMOVE 对照

| 阶段 | FORCE 物理点 | MARK 物理点 | FORCE 索引 MiB | MARK 索引 MiB |
| --- | ---: | ---: | ---: | ---: |
| S0 | 10,000 | 10,000 | 146.57 | 145.34 |
| S1 | 9,000 | 10,000 | 144.65 | 145.36 |
| S2 | 8,000 | 10,000 | 143.14 | 145.36 |
| S3 | 9,000 | 11,000 | 145.22 | 147.24 |
| S4 | 10,000 | 12,000 | 146.87 | 148.84 |

两种模式有效点数均为 10k → 9k → 8k → 9k → 10k，物理点数并不相同。

| 阶段 | FORCE 向量 MiB | MARK 向量 MiB | FORCE RSS MiB | MARK RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| S0 | 15.00 | 15.00 | 205.46 | 202.34 |
| S1 | 13.18 | 15.00 | 205.51 | 202.49 |
| S2 | 11.72 | 15.00 | 204.09 | 202.49 |
| S3 | 13.50 | 16.50 | 205.51 | 205.72 |
| S4 | 15.00 | 18.00 | 207.69 | 206.79 |

10k 的 FORCE 图内存从 S0 的 129.33 变为 S2 的 129.62 MiB，
至少一个 128 MiB 块没有释放，故总内存只下降约 3.43 MiB。
向量确实已经删除，不能据此否定物理缩容。

| 阶段 | FORCE QPS | FORCE recall | MARK QPS | MARK recall |
| --- | ---: | ---: | ---: | ---: |
| S0 | 59,700.36 | 99.15% | 59,658.00 | 98.85% |
| S1 | 64,371.72 | 97.45% | 59,928.16 | 98.85% |
| S2 | 69,515.67 | 97.75% | 60,476.66 | 97.00% |
| S3 | 67,242.11 | 97.45% | 55,990.72 | 98.85% |
| S4 | 66,083.50 | 99.15% | 58,258.47 | 98.85% |

上表为 ef=80，完整 ef=40/80/160 数据见归档 CSV。
物理 ID 置换、团结构及 seed 选择可以改变近似遍历结果，
不要求 FORCE_REMOVE 与 MARK_REMOVE 具有完全相同的 recall。

## 11. 验证范围与待改进项

实现阶段已有 81 个相关单元测试、8,657,906 个断言通过，覆盖：

- FP32/INT8、移动后向量距离与 label 一致性。
- 重复/不存在 ID、软删转物理删、同 label 再添加、删空后再添加。
- base/extra/delta 与标记混合后的重映射、序列化和计数恢复。
- 并发查询与 FORCE_REMOVE/Add/Flush，以及 CSR 分配失败时的替换原子性。
- 入口点删除、墓碑尾点搬移、存储 Shrink 后的逻辑数量。

相关命令：

```bash
./build/tests/unittests \
    '[ut][hgraph],[ut][MCISearcher],[ut][LabelTable],[ut][HGraphParameter],[ut][FlattenDataCell]'
```

早期物理删除测试还重复运行了 15 次；相关 clang-format-15 和定向 clang-tidy-15
检查通过。这些是实现阶段的记录，本次文档编写没有重新运行整个 C++ 测试集。
没有测量全库覆盖率，不能声称已验证仓库要求的 90% 覆盖率，也未完成全功能测试。

优先改进方向（尚未在本报告实现）：

1. 大批 ADD 的候选生成与串行团更新：分离可并行候选计算与有序发布。
2. 删除修复的全量精确距离和完整排序：评估 top-k 选择、批量距离或近似候选，
   但必须重新验证质量，不能默认等价。
3. 图的尾块收缩和反向边容器压紧：减少同点数下的多余预留。
4. 增加同 recall 对照、更多查询、重复运行、3m MARK/Flush 对照及不保护真值的删除实验。
5. 若需要严格连通性、修复后最小覆盖数或更强异常恢复，应单独定义并实现这些保证。

## 12. 结果归档与源码导航

本文保留测试结果汇总；本 PR 不包含原始 CSV 结果文件。

原始连续 RSS、日志和本机复现辅助脚本分别位于
`/tmp/mci-force-3m-wO5WR3/`、`/tmp/mci-force-verify-5G5UPQ/final/`，
它们是本机临时产物，不是可移植路径。
之前的交互图数值经过 CSV 核对，但浏览器预览因依赖下载超时未验证；
本文表格和归档 CSV 是独立可读的结果依据。

| 文件 | 主要职责 |
| --- | --- |
| [hgraph_build.cpp][src-build] | Build、Add 批次及 MCI 发布。 |
| [hgraph_mci.cpp][src-mci] | 候选搜索、加团/构团、删除修复、Force 和 Flush。 |
| [hgraph_modify.cpp][src-modify] | Remove 分发、图边修补、尾点搬移及缩容。 |
| [clique_datacell.cpp][src-cell] | 三类 delta、删除快照、提交、两向 CSR 压紧。 |
| [clique_datacell.h][src-view] | 固定 base/delta/标记的查询视图与直接遍历。 |
| [mci_searcher.cpp][src-search] | 直接距离计算及候选队列。 |
| [label_table.cpp][src-label] | label 搬移、删除状态和恢复。 |
| [hgraph_serialize.cpp][src-serialize] | 持久化恢复及内存分项。 |
| [memory_block_io.cpp][src-block] | 按完整块分配和缩容。 |
| [mci_mutation_benchmark.cpp][src-bench] | 数据加载、过滤真值、五阶段检索与 CSV。 |

[src-build]: ../../../../../src/algorithm/hgraph/hgraph_build.cpp
[src-mci]: ../../../../../src/algorithm/hgraph/hgraph_mci.cpp
[src-modify]: ../../../../../src/algorithm/hgraph/hgraph_modify.cpp
[src-cell]: ../../../../../src/datacell/clique_datacell.cpp
[src-view]: ../../../../../src/datacell/clique_datacell.h
[src-search]: ../../../../../src/impl/searcher/mci_searcher.cpp
[src-label]: ../../../../../src/impl/label_table/label_table.cpp
[src-serialize]: ../../../../../src/algorithm/hgraph/hgraph_serialize.cpp
[src-block]: ../../../../../src/io/memory_block_io/memory_block_io.cpp
[src-bench]: ../../../../../tools/eval/mci_mutation_benchmark.cpp

## 13. ADD / Delete 阈值扫描脚本

使用 [sweep_mci_thresholds.py][src-sweep] 扫描已有阈值，不修改索引算法。
默认先在 10k 上做单因素实验（OAT），避免直接在 3m 上运行完整参数网格。

| 扫描参数 | 基线 | 默认候选值 |
| --- | ---: | --- |
| `join_ratio` | 0.6 | 0.4、0.6、0.8 |
| `added_mct` | 3 | 1、3、6 |
| `clique_max`（仅增量） | 50 | 25、50、100 |
| `delete_size` | 3 | 3、4、5、6 |
| `delete_mct` | 3 | 3、5、8 |

低于删除大小阈值才废弃团，不是小于等于。
删除大小阈值从基线 3 按步长 1 提升，依次测试 4、5、6，不扩大到 20/40。
10k 初始团可能接近 50 点，这些阈值可能不触发小团修复；应如实报告无变化。
提高 `delete_mct` 在没有小团被废弃时可能没有作用，需要后续组合实验验证交互。

增量团上限与全量构建的 `mci_clique_max=50` 分开控制，避免同时改变初始索引。
`mci_mcs` 仍固定为 50，因此把增量上限提高到 100 不保证新团有 100 个成员；
候选数和已有团的追加行为也会限制实际大小。
`added_mct` 是最多尝试加入团数，`delete_mct` 是修复触发条件，均不是覆盖保证。

### 13.1 执行流程

1. 默认基线加上各维度的非基线取值，共 12 组；每组重复 3 次，共 36 次独立运行。
2. 每次完整执行五阶段，扫描 ef=40/80/160/320，共 720 个检索测量点。
3. 每个重复轮次中，所有配置共用相同删除随机种子；轮次之间种子递增。
4. 配置运行顺序按固定种子打乱，但运行本身串行，避免实验互相争抢 CPU。
5. 默认构建 1 线程以减少初始构建差异，检索 16 线程；
   不保证索引逐字节一致，也没有将删除随机种子当成 HGraph 的构建种子。
6. 默认每批修改 1,000 点，200 条 recall 查询，每个 ef 计时 10,000 次查询。

这些设置与第 10 节历史测试的构建线程、批次大小和查询次数不同，
应比较本次扫描的内部基线，不能直接比较历史 QPS。
过滤及真值保护规则与原基准程序相同，未固定每阶段的 seed 集合。

先确保当前源码的二进制已构建，再检查计划并启动：

```bash
cmake --build build-release --target mci_mutation_benchmark -j 8
python3 scripts/perf_reports/sweep_mci_thresholds.py --dry-run
python3 scripts/perf_reports/sweep_mci_thresholds.py \
    --output-dir /tmp/mci-threshold-oat
```

脚本不自动构建；会检查二进制是否支持新的阈值参数。
省略输出目录时自动创建独立临时目录，不覆盖已有非空目录。
每组超时默认为 600 秒；失败日志保留，其余配置继续，最终返回非零退出码。

仅对比删除团大小阈值 3/4/5/6，固定其他参数（4 组、12 次运行）：

```bash
python3 scripts/perf_reports/sweep_mci_thresholds.py \
    --only baseline,delete_size-4,delete_size-5,delete_size-6 \
    --output-dir /tmp/mci-threshold-delete-step1
```

完成单因素实验后，显式指定候选范围进行组合实验：

```bash
python3 scripts/perf_reports/sweep_mci_thresholds.py --design grid \
    --join-ratios 0.6,0.8 --added-mcts 3,6 --clique-maxes 50 \
    --delete-sizes 4,5,6 --delete-mcts 3,5 --dry-run
```

网格之外仍保留原始基线并去重；上述例子为 25 组、75 次运行，也会超过默认运行上限。
完整默认网格是 324 组、972 次运行，会超过默认 `--max-runs=64` 而被拒绝。
需先确认计划，再显式提高限制；脚本不会自动挑选“最优配置”。

3m 应只验证少量入选配置，并指定整批修改和更长超时，例如：

```bash
python3 scripts/perf_reports/sweep_mci_thresholds.py \
    --dataset /root/data/codefilter-3m-384-angular-f32.hdf5 \
    --only baseline,added_mct-6 --repeats 1 --build-threads 16 \
    --mutation-batch-size 324138 --timeout 7200 \
    --target-recalls 0.85,0.90,0.92 --output-dir /tmp/mci-threshold-3m
```

这只是复现入口，不表示已经完成该 3m 阈值对照。
`--only` 中的名称来自 `--dry-run`；默认模式为 FORCE_REMOVE，
`--remove-mode mark` 可改测逻辑删除。
`--flush-after-mutation` 应作为单独对照运行，不与未 Flush 的样本混为一组。

### 13.2 输出与判读

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 参数计划、数据文件身份、二进制及动态 VSAG 库指纹。 |
| `配置/repeat-N/attempt-N/` | 原始 CSV、完整命令、日志和成功/失败状态。 |
| `all.csv` | 所有成功运行的逐阶段/ef 原始值及相对各自初始状态的 recall 变化。 |
| `summary.csv` | 每配置、阶段、ef 的中位数，recall/QPS 最小–最大值及样本数。 |
| `target-recall.csv` | 达到指定 recall 下的最快实测 QPS；不插值、不外推。 |
| `failures.json`、`status.json` | 失败列表、成功数和计划总数。 |

目标 recall 的 QPS 只有在所有计划重复均成功且达到目标时才填写，否则留空。
单次运行没有达到目标，不能用其更低 recall 下的高 QPS 替代。
同 ef 的 `recall_delta_pp` 逐运行相对自己的 S0 计算，再汇总，单位为百分点。

新 CSV 还包含 `avg_dist_cmp`、`avg_hops`、`avg_seed_count`、平均团大小、
总覆盖关系及 delta extra 关系数。前面三个值来自计时前的 recall 查询；
MCI 的 hops 在这里表示访问团数，不是最短路径长度。
`memberships_per_live_node` 按存活点数归一化，避免 MARK 的物理槽位数混淆。
这些是路径工作量与结构摘要，不是连通性或真值可达性的直接证明。

脚本检查五阶段有效/物理点数、MCI 全覆盖、删除标记、100% 快速路径、
参数回显及测量完整性；不满足条件的运行不混入汇总。
中断后使用完全相同的参数加 `--resume` 续跑：

```bash
python3 scripts/perf_reports/sweep_mci_thresholds.py \
    --output-dir /tmp/mci-threshold-oat --resume
```

只跳过通过验证且有完成标记的运行；未完成尝试保留并在新 attempt 目录重试。
参数、数据身份或二进制指纹变化时拒绝续跑，防止混合不同实验。

脚本测试：

```bash
python3 -m unittest discover -s scripts/perf_reports -p 'test_sweep_mci_thresholds.py' -v
```

[src-sweep]: ../../../../../scripts/perf_reports/sweep_mci_thresholds.py

## 14. 80 万初始点的随机增删压测

[run_mci_stress.py][src-stress] 使用独立的 stress 模式，不沿用五阶段实验的真值保护规则。
完整 3m 数据集含 3,241,378 条向量，默认从中无放回随机选 800,000 条全量构建索引。
每轮再从完整数据池无放回抽取 `round(3,241,378 / 14) = 231,527` 个 ID：

- 轮次开始时已存在的 ID 执行 FORCE_REMOVE；不存在的 ID 执行 ADD。
- 同一轮抽样不重复，不同轮次可以再次抽到同一个 ID；不是把数据池切成 14 个固定分片。
- 每轮先处理抽中的存活点删除，再添加抽中的缺失点，分别检索和记录统计。
- 默认 14 轮；通常产生初始状态加 28 个修改检查点，共 116 个 ef 测量点。
- 初始点数为 80 万，不要求后续点数保持 80 万；混合随机切换会改变存活集合和点数。

默认删除团大小阈值仍为 3，ADD 的 join ratio=0.6、added_mct=3、clique max=50。
构建和检索各 16 线程，固定前 200 条查询，ef=40/80/160/320，
每个 ef 计时 10,000 次检索。增删批大小为每轮抽样量，不逐点调用公共接口。

```bash
cmake --build build-release --target mci_mutation_benchmark -j 8
python3 scripts/perf_reports/run_mci_stress.py \
    --dataset /root/data/codefilter-3m-384-angular-f32.hdf5 \
    --initial-count 800000 --step-count 231527 --rounds 14 --mode toggle \
    --output-dir /root/data/mci-stress-800k-20260908
```

程序将数据集加载到内存，预计算固定查询在完整数据池中的精确标签过滤距离排序；
每个检查点从该完整排序中选出仍存活的 top-k，等价于对当前集合重新计算精确真值。
不会保护或排除真值邻居，也不会使用全库原始 top-k 直接评价 80 万子集。
若固定查询在某个阶段不足 top-k 个存活匹配点，明确失败，不悄悄更换查询集合。
排序缓存和数据集属于基准程序开销，不计入索引内存；图中索引内存不是进程 RSS。
不同检查点的数据集合不同，因此 recall 变化不能只归因于索引结构退化。

输出包括：

- `statistics.png`：点数、recall、QPS、索引及分项内存、团数、增删耗时；每个检查点更新。
- `qps-recall.png`：初始状态及均匀抽取的最多四个后续检查点的 QPS–recall 曲线。
- `curve.csv`：逐检查点、逐 ef 的完整测量；`.events.csv` 附件记录操作、点数及真值开销。
- `curve.csv.ids.csv`：初始成员和每次操作 ID，支持审计无重复抽样与存在性。
- `curve.csv.truth.csv`：逐检查点、逐查询的精确真值 ID。
- `manifest.json`、`benchmark.log`、`status.json`：参数、日志和最终完成状态。

曲线只包含已经完成检索并通过索引点数/覆盖检查的检查点，运行中不是完整结果。
输出目录不可非空，不支持从中途索引状态续跑；失败时保留原始日志和已完成曲线。
仅重新生成已有测量的统计图：

```bash
python3 scripts/perf_reports/run_mci_stress.py \
    --output-dir /root/data/mci-stress-800k-20260908 --plot-only
python3 -m unittest discover -s scripts/perf_reports -p 'test_mci_stress.py' -v
```

`--mode alternate` 为可选对照：奇数轮从缺失集合抽样 ADD，偶数轮从存活集合抽样 DELETE，
每轮操作 231,527 条。它与默认混合切换负载不同，不能把二者结果混为同一条曲线。
单元/集成测试使用合成数据独立暴力验证动态真值、ID 存在性、每轮样本数和固定种子复现。
这些测试不代表 80 万实测已完成，完整实测以输出目录的 `status.json` 为准。

[src-stress]: ../../../../../scripts/perf_reports/run_mci_stress.py

## 15. 纯 FP32 五阶段复测与初始索引保存

2026-09-08 完成的测试使用 3,241,378 条向量、FP32、16 个构建/搜索线程，
每批 FORCE_REMOVE 或 ADD 324,138 条向量。ef_search=320 时结果如下：

| 阶段 | 存活向量数 | Recall@10 | QPS | 本阶段增删秒数 | 索引 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初始 | 3,241,378 | 94.50% | 2,691.89 | 0 | 6,323.55 |
| 删除 10% | 2,917,240 | 94.75% | 2,863.30 | 349.26 | 5,684.00 |
| 删除 20% | 2,593,102 | 95.80% | 3,255.91 | 337.23 | 5,232.58 |
| 加回 10% | 2,917,240 | 96.35% | 3,277.14 | 534.94 | 5,783.94 |
| 加回 20% | 3,241,378 | 94.65% | 3,346.91 | 528.11 | 6,432.85 |

初始构建耗时 553.47 秒。索引 MiB 是索引报告的分配量，**不是进程 RSS**；
本次未记录各阶段 RSS。本 PR 不包含原始结果文件。
这些结果来自适配新版 main 前的开发实现，不代表适配后 PR 版本的重新压测。
固定 ef 下 QPS 提高也不等于同等 recall 下性能一定提高。

删除修复改为 HGraph KNN 后，可用以下后台脚本复测：

```bash
python3 scripts/perf_reports/run_mci_fp32_cycle.py \
  --dataset /root/data/codefilter-3m-384-angular-f32.hdf5 \
  --output-dir /root/data/mci-fp32-cycle-new --threads 16
```

脚本使用已构建的 Release 二进制，并复制程序和 libvsag 到结果目录，避免后续编译
改变运行版本。阶段为初始 100%、删除到 90%、删除到 80%、添加回 90%、添加回 100%。
每批为初始总量的 10%，ef 为 40/80/160/320，使用 200 个查询测 recall、10,000 次查询测
多线程 QPS。底层构建与检索使用指定线程数；删除批次及 MCI 修复仍串行执行。
删除集合沿用旧五阶段测试的真值邻居保护规则，不等同于无保护随机增删压测。

`initial.index` 在首次删除前保存，配套 `initial.index.json` 记录数据集路径、大小、
修改时间、点数和完整构建参数。命令行工具直接支持 `--save-initial-index`，
但不支持 stress 模式。已有快照不会被覆盖，保存耗时不计入构建和增删耗时。
文件使用现有 `Serialize(std::ostream&)` 格式，文件与元数据需一起保留。

历史预检曾发现：从文件流或 BinarySet 加载后继续 FORCE_REMOVE，400 点 FP32
用例触发内存错误，因此上述性能测试从头构建。PR review 后已定位并修复：反序列化
未恢复图的反向边，尾点搬移遗漏入边更新，留下越界引用，最终在 GetStats 入度统计时破坏堆。
现在 flat 底图和 sparse 上层图均从解码后的有效正向边重建反向边，不改变存盘格式。
新增回归覆盖两种序列化接口、两种内存 IO，以及加载后两次删除、两次加回和搜索。
**增删基准仍不提供加载复用选项**；这些回归不代表已对保存的 3m 快照重新压测。

`status.json`、`benchmark.log` 和 `curve.csv` 可用于查看后台进度；完成后验证全部
20 个 stage/ef 测量点，生成 `qps-recall.png`。本脚本不自动排队，运行前应确认没有
其他性能测试争用资源。

```bash
python3 -m unittest discover -s scripts/perf_reports -p 'test_mci_initial_index.py' -v
```
