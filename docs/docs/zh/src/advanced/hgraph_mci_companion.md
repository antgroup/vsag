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

浮点快速路径和通用全量构建路径共用最终补覆盖流程。极大团枚举结束后，仍未被覆盖的点
会与图邻居组成兜底团；没有可用邻居时使用单点团。此流程放宽团内距离约束，按团大小上限
截断时保留当前点，并且只统计实际保存的成员关系。

| 参数 | 作用 |
| --- | --- |
| `use_mci` | 设为 `true` 时使用默认构建参数启用 MCI。 |
| `mci_mcs` | 构建团时使用的候选邻居数量。 |
| `mci_clique_max` | 全量构建时保留的极大团的最小大小。团始终完整保存，因此这是下限而非上限。 |
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

迁移提示：不再支持 `mci_incremental_added_mct`。请从 `index_param` 中删除这一旧的
团数量配置，改用 `mci_incremental_degree_min`、`mci_incremental_degree_n_divisor`
和 `mci_incremental_degree_mcs_divisor`。旧键会被忽略，不会自动转换；若保留旧键且未
显式设置度数参数，将使用新默认值。由于团之间存在重叠，不同团贡献的去重邻居数量不同，
不存在一一对应的转换公式。迁移后应重新评估 recall 和增删成本，并以新配置重建旧快照，
不能认为删除参数就能保证快照兼容。

MCS 除数默认 `2`，是为了在应用度数下限之前，将自适应项限制在 KNN 候选预算的一半，
并不是将最终目标限制为团大小的一半。MCS 默认 `200`，与 `mci_clique_max` 不同。
若 `mcs=32` 且下限为 `50`，对任意 N，目标都是 `min(N-1, 50)`，并非
`max(50, N/10000)`。这是一项调参启发式，不保证一半候选会成为邻居；下限可以主导目标，
加入完整团也可能超过目标。

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
      "search_no_lock": false,
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
`search_no_lock` 是布尔查询参数，默认 `false`，替代 `mci_seed_no_lock`（旧名称会报错）。
开启后，HGraph 和 MCI 搜索中的路由、标签/删除集合、向量、clique 和结果处理等读锁
使用空锁。访问表池改用独立的原子借还链表，不会无保护地读写原来的共享队列。
搜索算法、过滤和不存在/已删除 ID 的处理语义保持不变。

此选项仅用于只读内存索引：查询期间不能并发构建、添加、删除、更新向量/标签、调优、
扩容或反序列化。支持多个独立查询并发，也支持带锁与无锁查询混合并发。
`hgraph.parallelism` 必须为 1，因为单查询内部的工作线程池需要同步；
可用多个线程分别执行独立查询。该参数不关闭分配器、操作系统、外部过滤器或磁盘缓存内部的同步。
锁策略只在本次调用内生效，正常返回或抛出异常后恢复，不修改索引共享的锁对象。
查询统计字段为 `search_no_lock`。已有索引可直接复用，无须重建或重新保存。

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

## 代码入口与维护

下表中的文件路径均相对于仓库根目录。

| 文件 | 重点入口 / 职责 |
| --- | --- |
| `include/vsag/index.h`、`src/index/index_impl.h` | 公共 Build/Add/Remove/Search 接口及错误包装 |
| `src/algorithm/hgraph/hgraph_build.cpp` | `Add` → `add_impl`：先插入 HGraph，再维护成功插入点的 MCI |
| `src/algorithm/hgraph/hgraph_mci.cpp` | 全量构团；`search_mci_knn`、`incremental_update_mci_clique`、`repair_mci_clique`、`force_remove_with_mci`、`maybe_compact_mci` |
| `src/algorithm/hgraph/hgraph_modify.cpp` | Remove 模式分流、图边修补、尾部 ID 搬移、存储缩容 |
| `src/datacell/clique_datacell.{h,cpp}` | 两向 CSR、三种 delta、删除快照、团废弃、节点重映射和 Flush |
| `src/impl/searcher/mci_searcher.cpp` | `search_clique_view`：统一遍历 base CSR + delta + 删除标记 |
| `src/impl/label_table/label_table.{h,cpp}` | 外部标签映射、删除集合、一次查询固定删除集合的读视图 |
| `src/algorithm/hgraph/hgraph_parameter.{h,cpp}`、`hgraph_param_mapping.cpp` | 参数默认值、校验和外部 JSON 映射 |
| `src/algorithm/hgraph/hgraph_serialize.cpp` | MCI 格式版本及序列化恢复 |
| `src/analyzer/hgraph_analyzer.cpp` | 团覆盖、大小、成员数及内存统计 |

### 1.1 数据布局

基础层包含两向 CSR：`clique → nodes` 和 `node → cliques`。增量层与 ADD/删除修复共享：

| 字段 | 保存内容 |
| --- | --- |
| `delta_cliques_` | 新建团的完整成员 |
| `delta_clique_extra_` | 追加到基础团的成员 |
| `delta_node_to_cids_` | 增量产生的 node → clique 关系 |

节点删除标记、团废弃标记独立维护。Flush 清空 delta 并重建两向 CSR；它不会改变向量 inner ID，
但可能重新编号团。FORCE_REMOVE 会移动向量 inner ID，因此还必须同步重映射 MCI。

### 1.2 ADD 与删除修复的共用流程

1. ADD 先完成 HGraph 插入，再维护成功插入点的 MCI：块内并行规划，再按 inner ID
   升序单线程合并（每行仍按下面 2–3 步的规则处理）。候选来自 `search_mci_knn`，内部调用
   HGraph `KnnSearch`，显式设置 `use_mci=false`；不是通过 MCI 搜候选。
2. 目标邻居数为 `min(mci_mcs, visible_total - 1)`。内部检索 ef 为 `max(query_k, 100)`；
   剔除自身、删除点及可见范围外点，不足时可扩大请求数量。这里不使用性能测试的 ef=320。
3. 优先加入满足 `|KNN ∩ C| / |C| >= join_ratio` 且未达到增量大小上限的已有团，
   以去重邻居度数而非团数作为停止条件。度数不足时，使用尚未相邻的 KNN 候选，调用与全量 Build 共用的
   `MCILocalCliqueBuilder::Build`（`src/algorithm/mci/mci_local_builder.h`），使用增量团大小上限，
   把当前点视为需要覆盖的点，重复直到目标达到、候选耗尽或度数无进展。
   不再以团大小达到 2 作为 alpha 扩张的停止条件。没有候选时创建单点团。
4. 删除只废弃受影响且删除后大小 **小于** `delete_size` 的团；从这些团中收集预计有效覆盖团数
   **小于** `delete_mct` 的存活点。修复前再次检查覆盖数，避免已经恢复覆盖的点重复修复。
5. 纯 FP32 修复从存储中取得或解码该点向量，然后进入第 1–3 步的共用维护流程。
   不再次调用公共 ADD 插入已有向量，也不产生新的向量标签。ADD 的候选可见范围是插入前缀；
   修复已有点时可见范围是当前整张 HGraph。

ADD 的团数限制参数已移除；`delete_mct=3` 仍是修复候选触发阈值。
度数目标为 `max(mci_incremental_degree_min, min(N/10000, mcs/2))`，N≤1 时为 0，否则不超过 N-1；
N 是当前存活点数，包含图插入完成的批次，整数除法向下取整。下限默认 50，10k、mcs=200 时目标是 50。
加入完整团可以超过目标；无法继续增加邻居时允许低于目标停止。
非 FP32 修复仍保留成对距离候选路径，本指南不把 FP32 的结论推广到 RaBitQ。

### 1.3 三种维护操作

| 操作 | 向量槽位 | MCI 维护 | 是否落盘 |
| --- | --- | --- | --- |
| MARK_REMOVE（默认） | 保留，仅逻辑删除 | 小团筛选、低覆盖点修复，结果进入 delta | 否 |
| FORCE_REMOVE | 尾点搬移填洞，减少物理槽位并尝试缩容 | 批量快照、ID 重映射、修复、自动 Flush | 否 |
| 内部自动压缩 | 不删除或移动向量 | 合并 delta、移除废弃成员、压紧团编号、重建两向 CSR | 否 |

## 基准配置与 C++ 接入

### Codefilter FP32 构建配置

将下面 JSON 字符串传给 `Factory::CreateIndex("hgraph", config_json)`。
所有 MCI 构建参数直接放在 `index_param` 下，不使用嵌套 `mci` 对象。

```json
{
  "dtype": "float32",
  "metric_type": "cosine",
  "dim": 384,
  "index_param": {
    "base_quantization_type": "fp32",
    "base_io_type": "memory_io",
    "graph_type": "nsw",
    "max_degree": 32,
    "ef_construction": 200,
    "build_thread_count": 16,
    "support_force_remove": true,
    "use_mci": true,
    "mci_knng_source": "hgraph",
    "mci_mcs": 200,
    "mci_clique_max": 50,
    "mci_alpha": 1.2,
    "mci_incremental_join_ratio_threshold": 0.6,
    "mci_incremental_degree_min": 50,
    "mci_incremental_degree_n_divisor": 10000,
    "mci_incremental_degree_mcs_divisor": 2,
    "mci_incremental_clique_max": 50,
    "mci_delete_clique_size_threshold": 30,
    "mci_delete_node_mct_threshold": 3
  }
}
```

这是测试配置，不是所有库参数的默认值。`support_force_remove` 必须在创建时启用；
MCI FORCE_REMOVE 要求 flat 图存储，会自动启用反向边，并拒绝不兼容的去重、重复分组或属性存储配置。

| 参数 | 库默认值 | 基准程序 CLI / 说明 |
| --- | ---: | --- |
| `mci_mcs` | 200 | `--mci-mcs`；基准默认 50，限制候选邻居数 |
| `mci_clique_max` | 50 | `--mci-clique-max`；全量团大小上限 |
| `mci_alpha` | 1.2 | `--mci-alpha`；构团扩展系数 |
| `mci_incremental_join_ratio_threshold` | 0.6 | `--mci-incremental-join-ratio-threshold`；范围 [0,1] |
| `mci_incremental_degree_min` | 50 | 度数目标下限，正整数，随索引参数序列化 |
| `mci_incremental_degree_n_divisor` | 10000 | 度数目标中的存活点数除数，正整数 |
| `mci_incremental_degree_mcs_divisor` | 2 | 度数目标中的 MCS 除数，正整数 |
| `mci_incremental_clique_max` | 50 | `--mci-incremental-clique-max`；至少 2；基准不指定时跟随全量上限 |
| `mci_delete_clique_size_threshold` | 30 | 正整数；默认废弃剩余有效成员数小于 30 的受影响团，恰好为 30 时保留 |
| `mci_delete_node_mct_threshold` | 3 | `--mci-delete-node-mct-threshold`；正整数，严格小于才修复 |

例如 `delete_size=4` 会考虑删除后只剩 0–3 个成员的受影响团，并不废弃所有包含被删点的团。
没有小团被废弃时，单独提高 `delete_mct` 可能完全不触发额外修复。

### 搜索配置

```json
{
  "hgraph": {
    "ef_search": 320,
    "use_mci": true,
    "mci_seed_ratio": 0.1,
    "hgraph_valid_ratio_threshold": 1.0
  }
}
```

构建用 `index_param`，搜索用 `hgraph`。库的路由阈值默认是 **0.05**；上面和基准程序使用 **1.0**，
使选择率低于 1 的过滤查询优先尝试 MCI，仍不能保证每次一定走 MCI。
必须提供实际过滤器及合理的 `ValidRatio()`；不要伪报选择率来强制路由。
`use_mci=false` 可在同一个索引上对照普通 HGraph 搜索。

### C++ 调用片段

以下是接入片段，不是完整数据加载程序。`config_json`、`search_json` 为前面的 JSON 字符串；
`base`、`added`、`query` 是调用方准备好的 Dataset，`filter` 是过滤器，`removed_labels` 是外部标签数组。
使用 `Owner(false)` 时，调用方应保持向量及标签缓冲区在调用期间有效。

```cpp
auto created = vsag::Factory::CreateIndex("hgraph", config_json);
if (!created.has_value()) {
    throw std::runtime_error(created.error().message);
}
auto index = created.value();
auto check = [](const auto& result) {
    if (!result.has_value()) {
        throw std::runtime_error(result.error().message);
    }
};

auto built = index->Build(base);
check(built);
// Build/Add 返回未成功插入的标签；也要检查列表，而不只检查 expected。
if (!built.value().empty()) {
    throw std::runtime_error("some initial vectors were not inserted");
}
auto removed = index->Remove(removed_labels, vsag::RemoveMode::FORCE_REMOVE);
check(removed);
// 改为 MARK_REMOVE 即保留物理槽位；removed.value() 是实际删除数量。
auto appended = index->Add(added);
check(appended);
if (!appended.value().empty()) {
    throw std::runtime_error("some added vectors were not inserted");
}
// MCI 自动压缩，不需要也不再提供公共 Flush 调用。
auto result = index->KnnSearch(query, 10, search_json, filter);
check(result);
```

Remove 使用外部标签，不接受把内部槽位当成标签；物理删除后不要缓存 inner ID 或团 ID。
完整的 Dataset 与 Filter 示例见 `examples/cpp/324_feature_hgraph_mci_companion.cpp`。

## Add、删除、序列化和统计

通过扁平构建参数启用 MCI 后，`HGraph::Add()` 先完成 HGraph 插入批次，
再更新成功插入点的 MCI。它会先尝试加入合适的已有团；
如果去重邻居度数仍不足，则继续为新点创建增量团。

这一批次的 MCI 更新按块处理：块内先在冻结的只读视图上并行规划每一行的候选 KNN、
入团提案和局部构团，再由单线程按 inner ID 升序合并并写回。逐行的覆盖度和度数目标
不受规划顺序影响；但候选 KNN 来自并发图检索，因此与全量并行构建一样，最终团的
具体结构可能与线程数、线程调度有关。

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

MCI 按成功增删的向量数量自动压缩，ADD 与 MARK_REMOVE 共用计数，累计到 100 个触发。
ADD 每处理一个成功新增点检查一次，大批 ADD 内也会分段压缩；MARK_REMOVE 先投影并修复
完整删除批次，再检查阈值，因此大批删除在批末压缩一次，不拆分删除语义。
重复或不存在的删除 ID、失败的新增不计数，修复点不视为新增向量。
FORCE_REMOVE 保留每批末尾的必要压缩并清零计数。压缩成功、全量重建或加载后计数清零，
该计数仅存在于运行时。未达到阈值时，查询和序列化仍包含 delta。
不提供公共 Flush 接口，也不为压缩扩展 Index 的虚函数接口。

内部 flush 合并新建 delta 团和
基础团追加成员，移除废弃团及已删除成员，压紧团 ID 并重建 node → clique 关系。
flush 会清空 delta，但保留向量 inner ID 和删除标记；重复执行不改变有效成员关系。
它不重建 HGraph，也不写入磁盘，持久化仍需调用 `Serialize()`。

flush 与 Add/Remove 串行执行，构建和发布新 CSR 期间持有团存储独占锁，查询可能等待。
所有替换缓冲区分配成功后才发布，分配失败不会破坏原有团数据。
自动维护遇到分配失败会保留 delta，不让已完成的 ADD/MARK_REMOVE 因压缩失败而报错，
下次成功增删时重试。这不改变 FORCE_REMOVE 原有的错误语义。
临时内存需要同时容纳新旧 CSR；不要跨 flush 保存并复用团 ID。
加载已验证的紧凑 CSR 时，若没有 delta 成员、空团行或删除/废弃标记，会保留干净状态，
首次 flush 不再重建 CSR；其他快照仍保守地在 flush 时进行压缩。

MCI 查询只获取一次共享锁来固定 CSR、delta 和删除标记，直接遍历成员而不复制列表。
连续 FP32 向量在 Add/Remove 后、flush 前后均可直接计算距离；统计字段
`mci_raw_float_csr` 也包含直接遍历 delta 的快速查询。其他向量布局复用相同的遍历和
候选队列，通过其原有距离接口计算。删除集合在一次 MCI 查询期间固定，先检查用户
过滤条件，再检查删除集合，避免每访问一个点都获取删除集合锁。

Add/MARK_REMOVE 期间 MCI 可能暂时不可用，并发查询仍按已有语义回退到 HGraph，
不保证修改期间的完整召回；图入口点已删除时，回退结果可能为空。
自动压缩属于增删内部步骤，查询在必要时等待团存储锁。
回退查询在结果打包前，会根据同一删除集合视图重新检查候选 ID，丢弃遍历过程中
被删除的旧候选，包括同一 label 已在新槽位重新添加的旧槽位。
这并不意味着整个并发查询具有事务快照语义。
若候选在遍历期间被删除，即使索引仍有足够的存活向量，KNN 回退结果也可能不足 `k` 条。
并发修改期间结果为尽力返回，不执行第二轮补齐搜索，因为补搜同样可能遇到新的修改。

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

## 评测说明

开发阶段使用的 benchmark、脚本及原始结果不属于本 PR；下表字段说明历史测试程序产出的 CSV 格式。

### 如何读结果

五阶段保护评测查询的 top-k 真值点，五次使用同一真值；随机增删不保护真值点，
每个检查点从精确全库排序中取当前存活 top-k。两种实验不能混用 recall 结论。

| CSV 字段 | 解释 |
| --- | --- |
| `stage`、`active_vectors`、`index_elements` | 阶段、存活数量、索引对外元素计数；后者不能代替物理存储统计 |
| `ef_search`、`recall_at_k`、`qps` | 搜索宽度、recall、计时吞吐；固定 ef 不等于固定质量 |
| `build_seconds`、`mutation_seconds`、`flush_seconds` | 历史构建、增删和显式压缩耗时；当前自动压缩计入增删耗时，不再提供阶段后公共 Flush |
| `index_memory_bytes`、`vector_memory_bytes`、`graph_memory_bytes`、`mci_memory_bytes` | 索引及分项统计，不是进程 RSS |
| `mci_route_ratio`、`mci_raw_float_ratio` | MCI 路由与直接 FP32 路径命中比例；不能只看配置判断路由 |
| `mci_total_cliques`、`mci_delta_cliques`、`mci_total_memberships` | 团数、增量团数、成员关系总数 |
| `avg_dist_cmp`、`avg_hops`、`avg_seed_count` | 距离计算、遍历及 seed 开销，用来分析 QPS 变化 |

MiB = bytes / 1,048,576。相同阶段不同 ef 行会重复阶段内存/耗时，不能相加。
QPS 是多线程吞吐，`1000 / QPS` 不是单请求延迟；当前 CSV 不含延迟分位数或逐阶段 RSS。

本 PR 不包含原始结果文件；这是适配新版 main 前的开发版本测量，不是当前 PR HEAD 的重新压测。

## 回归测试与边界

```bash
make debug VSAG_ENABLE_TESTS=ON COMPILE_JOBS=12
build/tests/unittests '[mci],[LabelTable],RaBitQSplitDataCell serialize and methods'
```

文档编写前验证：C++ 定向测试 38 个用例、3,294 条断言通过；脚本测试 10+1+3 项通过，
其中两个集成测试当时使用 Debug 基准程序作正确性验证。未完成全量 lint、全量测试和 90% 覆盖率验证。
review 后另补充了反向边恢复及加载后 FORCE_REMOVE 回归。大数据集加载复测、ABI、序列化兼容
及更全面的并发修改仍需独立验证；不能把定向测试当作生产就绪证明。

### 静态 MCI 团遍历的 bitmap 预取

存在不可变、按 inner-id 索引的合法性 bitmap 时，静态 MCI 以 32 个成员为一块读取团成员，并预取前方最多 96 个位置的成员对应的 bitmap 缓存行。局部合法性掩码按原始顺序选出成员，随后沿用既有 epoch 节点判重和距离计算，减少每个不合法成员上的依赖分支。此过程不预处理完整合法点集合，不改变种子、团成员或搜索顺序。没有可直接读取的 bitmap 时，继续使用原始访问路径。

开发对照使用环境参数：`VSAG_MCI_MEMBER_MODE`（0 原始逐点访问、1 滚动预取、2 分块预取、3 分块合法性掩码、4 掩码加预取，默认 4），`VSAG_MCI_MEMBER_DISTANCE`（默认 96，上限 256）、`VSAG_MCI_MEMBER_BLOCK`（默认 32，上限 64）和 `VSAG_MCI_MEMBER_LOCALITY`（GCC 预取局部性提示，默认 1）。参数在每条查询开始读取一次，应在并发检索前设置，查询运行时不修改进程环境。这些参数不改变索引序列化格式，无需重建索引。

### 查询临时合法团 CSR

静态索引的 MCI 分流路径支持实验搜索参数 `hgraph.mci_filtered_cliques=true`（默认 false），要求 `search_no_lock=true`。过滤器提供完整合法 ID 列表时，查询先映射、排序并去重合法内部 ID，再枚举其点到团关联，通过计数、前缀和及填充建立紧凑合法团成员表。每个线程复用连续成员池、紧凑团元数据及 epoch 索引，没有逐团堆分配，也不逐查询清空全部团状态。仅包含一个合法点的团不保存成员。

临时表构造和 ID 准备均包含在查询时间内；搜索直接访问合法成员，跳过 bitmap 和过滤器回调，保留点及团的访问判重。成员按内部 ID 递增排列，可能改变原始团成员遍历顺序，因此有限搜索宽度下的返回结果可能变化。完整种子覆盖时跳过构造。没有完整合法 ID 列表或列表无法完整映射时，保留原访问路径。此功能只构造查询私有视图，不修改已保存索引，也不为过滤图新增连接。

查询统计包含 `mci_filtered_cliques_requested`、`mci_filtered_cliques_used`、`mci_filtered_id_prepare_ms`、`mci_filtered_clique_build_ms`、`mci_filtered_incidences`、`mci_filtered_nonempty_cliques`、`mci_filtered_kept_cliques`、`mci_filtered_members` 和 `mci_filtered_capacity_bytes`。`mci_member_checks` 记录实际扫描的合法成员槽，`mci_filtered_source_member_checks` 记录所访问原团对应的原始成员槽数。

### 动态候选团扫描

MCI 分流路径新增实验参数 `hgraph.mci_dynamic_clique_scan=true`，默认 false。要求 `search_no_lock=true`、不可变的基准 CSR、可完整映射的 `GetValidIds` 合法点列表，以及 `mci_filtered_cliques=false`。复用已保存索引；合法 ID 不完整时明确拒绝，不静默切换算法。

计算所有采样种子的距离，以其中最近的 `max(k, mci_ef_search)` 个点初始化候选最大堆及其团并集 X。线程私有、可复用的 epoch 数组记录各团被当前候选覆盖的次数。按内部 ID 递增扫描剩余合法点，检查其点到团 CSR；任一团计数为正时，该点只计算一次距离，并立即更新候选堆及 top-k 结果堆。用 B 替换 A 时，A 所属团计数减一，B 所属团计数加一。检查下一个合法点前提交完整更新；实现先加 B 再减 A，避免公共团短暂归零引起无效激活。候选距离相等时优先较小的内部 ID。

团计数从零变为正数时，安排下一轮检查尚未计算距离的合法点，包括本轮扫描位置之前的点。完整一轮没有新激活团后停止，此时最终 X 不再覆盖任何尚未评分的合法点。这仍是依赖扫描顺序和有限 ef 的近似搜索。种子覆盖全部合法点时跳过团并集构造及后续扫描。此路径不物化全部合法团成员，不扫描原团成员槽，也不逐查询清空全部状态。

动态替换可能沿多层传播，因此不再严格限定两跳。此模式拒绝 `mci_hops_limit`，统计中的 `mci_hops_limit=0` 表示不适用深度限制，扫描轮数不表示图跳数。原团遍历路径可用正 uint32 参数 `hgraph.mci_hops_limit`（默认 3）进行对照。原遍历的 idle 和 lower-bound 提前停止选项不控制动态扫描。

统计包括 `mci_dynamic_clique_scan_used`、`mci_dynamic_selected_seeds`、`mci_dynamic_scored_points`、`mci_dynamic_scan_passes`、`mci_dynamic_point_checks`、`mci_dynamic_clique_checks`、`mci_dynamic_count_updates`、`mci_dynamic_activations`、`mci_dynamic_replacements`、`mci_dynamic_active_cliques`，以及 `mci_dynamic_` 前缀的合法 ID 准备、初始团并集和扫描耗时。`mci_member_checks=0` 表示没有检查原团成员槽，点到团检查另行计数。所有准备工作均包含在原生查询时间内。

### 有序单遍与双遍团扫描

设置 `mci_dynamic_clique_scan=true` 后，参数 `hgraph.mci_clique_scan_passes=1` 或 `2` 启用最多一遍或两遍的有序扫描；省略时保留原来的收敛扫描。要求不可变点到团 CSR 每行团 ID 严格递增，旧索引若不满足此前提则不支持此实验模式。本次使用的去重 Cohere10M 索引已全量核验全部点及归属。索引序列化格式不变。

每个线程复用合法点元数据：内部 ID、CSR 起止位置、最小及最大团 ID。序列1按完整团 ID 行的字典序排列，完全相同行按内部 ID 打破平局；序列2按最大团 ID 排列序列1的位置。最小、最大主键使用稳定基数排序，仅最小团 ID 相同的小组需要比较剩余字典序。序列2的后缀最小位置表可直接找到序列1中最大团 ID 能达到下界的最早位置。每32个连续点还保存块内最大团 ID。

当前候选团并集 X 维护准确引用计数和带 epoch 的层次位图，缓存 X 的最小、最大团 ID，提供紧凑成员判断和后继查找，不逐查询清空全部位图。候选替换完整提交后才判断下一个点。点的团 ID 区间与 X 不相交时跳过，整块不相交时跳过32点；序列1最小团 ID 超过当前 X 上界时终止。通过活跃团后继跳过点的无效团前缀，团 ID 超过活跃上界时停止行内判断。查询中预取后方点的不可变 CSR。

第二遍同时使用第一遍中新激活团的 ID 范围和当前 X 边界，通过序列2跳到安全起点。初始终点设为第一遍最后一次激活的位置之前，因为此后覆盖只可能缩小；第二遍若产生新激活，则扩大激活范围并恢复完整可扫描后缀。每点最多评分一次。遍数有上限，因此不同于原收敛模式，不保证最终 X 已不再覆盖未评分点；扫描轮数和排序均不表示图跳数。

实验开关 `hgraph.mci_clique_scan_gallop` 默认 false，true 时在行内反复使用后继与二分跳跃，要求启用有界扫描；false 在初次前缀跳过后顺序检查紧凑位图。同一 ef 和遍数下，两种实现返回结果相同。排序、位图维护和检索均包含在原生查询计时内。`mci_bounded_` 统计记录遍数上限、相交实现、排序耗时、范围及空隙拒绝、前缀/块/尾部跳过、第二遍起点及初始终点、二分比较次数、初始及最终 X 边界。排序耗时同时包含在 `mci_dynamic_scan_ms` 内。

### 分流 MCI 分支的导航

非迭代 KNN 路径在遍历 HGraph 导航层之前，先尝试以过滤集合提供的种子执行 MCI。成功接管请求的 MCI 分支不使用 HGraph 入口点，导航层距离次数为零。MCI 不可用或不能接管请求时，仍执行普通 HGraph 的入口点导航。该优化不改变索引格式或 MCI 种子预算。

### 多团扫描候选使用 HGraph 邻域

实验查询参数 `hgraph.mci_hub_clique_threshold` 默认为 0（关闭）。正数阈值启用按点分流：团归属数量大于等于阈值的候选点仍可进入候选堆和结果堆，但不激活自己的全部团，改用已有底层 HGraph 邻居作为该点的扩展邻域。普通候选点仍精确维护其团并集引用计数。索引中的完整团归属保持不变，因此其他候选仍可通过公共团发现多团点。

`hgraph.mci_hub_hgraph_hops` 默认 1，仅接受 1 或 2。一跳对尚未评分的合法邻居计算距离；两跳另外访问一跳非法邻居的邻接表，桥接点本身不评分、不进入任何堆。新接受的合法多团候选进入待处理队列，处理时已经出堆的点跳过。图扩展和距离计算均通过 query epoch 标记判重。

要求旧分流选择动态扫描 MCI、`search_no_lock=true`、不可变 CSR 和已有底层 HGraph。支持原收敛扫描与有序一遍/两遍扫描。不另起一次全局 HGraph 检索，`hgraph_ef_search` 不限制这种局部替代邻域。该操作属于近似邻域替换，不保证召回不变。种子覆盖全部合法点时仍跳过全部团和图扩展。索引格式及序列化保持不变。

`mci_hub_` 统计包含阈值、跳数、接受的多团候选数、未激活归属数、实际扩展多团点数、非法桥接扩展数、邻居检查数和新评分邻居数。最后一项已经包含在总距离次数中，不应再加一次。所有相关工作均计入原生查询时间。

### 全局 MCI 相关性路由

实验 API 位于 include/vsag/global_mci_hybrid.h，复用一个现有 FP32/L2 HGraph 及其静态全局 MCI。先调用 GlobalMCIHybrid::Create(index)，再调用其 KnnSearch。不使用分区索引、聚类中心或负载的相关性标签。router 存活期间索引必须不可变。仅支持 search_no_lock=true、查询内部 parallelism=1；可并发执行独立查询。

先按保留率分流：保留比例大于等于 hgraph_valid_ratio_threshold（默认 0.05）时直接执行过滤 HGraph。显式设置 correlation_method="clique_probe" 的其余查询先执行完整未过滤 HGraph 搜索，包括底图，使用 correlation_navigation_ef（默认 80）得到近似 top-10 锚点。满足过滤条件的锚点优先加入 MCI 种子，仍受 mci_seed_max_count 约束。

预测只检查这些锚点覆盖的全局 MCI 团。按锚点轮流取至多 correlation_max_cliques（256）个不同团，再在这些团内分散检查至多 correlation_member_budget（8192）个成员槽。线程私有 epoch 标记对重复向量去重。局部合法点密度除以全局保留比例得到富集度。它是局部产出的启发式指标，并非整个重叠团并集的无偏估计。

富集度不高于 correlation_negative_ratio（0.5），且独立假设下样本中的期望合法点数达到 correlation_min_expected（1），则判断负相关，精确扫描全部合法点。富集度不低于 correlation_positive_ratio（2）且至少命中一个合法点，则判断正相关。其余情况继续全局 MCI；样本不足或为空不会强制扫描。导航与预测均不读取 ground truth。correlation_routing=false 提供原保留率分流；correlation_probe_only=true 保留导航、预测和种子注入，但不切换扫描。

查询参数示例（直接传入，或包在 global_mci_hybrid 下）：

    {
      "correlation_method": "clique_probe",
      "hgraph_valid_ratio_threshold": 0.05,
      "ef_search": 40,
      "mci_ef_search": 40,
      "hgraph_ef_search": 80,
      "mci_seed_ratio": 0.1,
      "mci_seed_coverage": 0,
      "mci_seed_max_count": 10000,
      "mci_hops_limit": 3,
      "search_no_lock": true,
      "parallelism": 1,
      "correlation_navigation_ef": 80,
      "correlation_max_cliques": 256,
      "correlation_member_budget": 8192
    }

过滤器应通过 GetValidIds 提供完整合法集合；不能枚举时扫描分支逐一检查索引标签。导航、团采样、ID 转换、扫描准备和最终检索全部位于查询 API 内，必须计入端到端时延。结果统计 global_mci_hybrid 包含实际分支、预测结果、样本计数、合法导航种子数、导航及检索距离次数和各阶段耗时。router 不重建、不重排、不剪枝索引，也不修改序列化格式。

### 两组距离采样的相关性预测

GlobalMCIHybrid 的 MCI 分支默认使用 correlation_method="distance_samples"。全部相关性参数均有默认值，平时通过 wrapper 检索时可以省略。此预测不执行 HGraph 导航，也不读取团成员关系；外层保留率分流仍先执行。复用保存好的索引，不改变序列化。

从全集不放回均匀采样 correlation_global_samples（512）个内部 ID，从完整 GetValidIds 列表不放回均匀采样 correlation_valid_samples（256）个位置。correlation_sample_seed（42）固定全集样本；合法样本另外混合 query 向量的位模式，保证相同 query/filter 可复现。提供方的 ID 顺序必须稳定，列表必须完整且无重复。采样、映射和距离计算全部计入查询时间；全集采样与谓词无关。

以全集距离的 correlation_radius_quantile（0.1）分位数确定半径，合法样本落入半径的比例除以全集样本对应比例，得到富集度。距离相等时采用实际的闭区间全集比例。使用 correlation_confidence_z（1.96）的近似 Wilson 区间构造富集度区间：上界不高于 correlation_negative_ratio（0.5）则转扫描，下界不低于 correlation_positive_ratio（2）则判断正相关，其余中性。这是不确定性启发式，不是自适应分位数下的严格联合置信保证，也不能保证代表更小的真实 top-k 邻域。

全部合法样本使用已经算好的准确距离作为 MCI 种子，替换普通 seed_ratio 采样列表；数量必须满足 mci_seed_max_count，0 表示不限。内部桥接使用 mci_seed_replace、mci_seed_ids 及对齐且有限的 mci_seed_distances。缓存种子以 hops=1 进入候选队列和结果队列，不再计算距离；转扫描时也复用这些距离。全集样本与合法样本重叠的点仅计算一次。索引向量和图结构保持不可变。

correlation_method="sample_seeds" 是对照：使用相同合法样本和距离复用，不采样全集、不预测、不切换扫描。显式设置 correlation_method="clique_probe" 可使用原 HGraph top-10 预测。统计区分 sample_distances、reused_sample_distances、search_distances、total_distances，复用次数不计作新的距离计算。采样、映射、参数序列化、预测和正式检索全部计时。


### 静态路由的兼容性和采样预算

普通 HGraph/MCI API 保留上游的 live/delta 遍历。仅设置 `search_no_lock=true` 不会选择实验静态算法。wrapper 通过已有静态路由参数接入；该路径中的 `mci_hops_limit` 以种子深度 1、子点深度为父点深度加 1 计数。达到深度上限的点可进入结果，但不继续展开。构建、更新和加载索引保留上游实现，本次修改不包含此前实验中的构建剪枝和团重排序。

| 两组距离采样参数 | 默认值 | 允许值 |
| --- | ---: | --- |
| `correlation_method` | `distance_samples` | `distance_samples`、`sample_seeds`、`clique_probe` |
| `correlation_routing` | `true` | 布尔值 |
| `correlation_probe_only` | `false` | 布尔值 |
| `correlation_global_samples` | 512 | 整数 1–65536 |
| `correlation_valid_samples` | 256 | 整数 1–1024，决定替换后的种子预算 |
| `correlation_sample_seed` | 42 | 整数 0–1000000000 |
| `correlation_radius_quantile` | 0.1 | (0,1] 内的有限值 |
| `correlation_confidence_z` | 1.96 | 有限正数 |
| `correlation_negative_ratio` | 0.5 | [0,1) 内的有限值 |
| `correlation_positive_ratio` | 2.0 | 大于 1 的有限值 |

`mci_seed_max_count` 必须为 0，或至少等于配置的 `correlation_valid_samples`，即使合法点总数更小也是如此；wrapper 会拒绝不足的上限。两种采样模式要求完整、唯一、顺序稳定的 `GetValidIds`，不会像旧团探测方式那样在缺失时退回全库枚举。大规模谓词优先提供 bitmap 接口。`correlation_probe_only=true` 保留预测并禁用预测触发的扫描；`correlation_method="sample_seeds"` 用相同合法种子对照，不做全集采样。MCI 分支不保证精确召回。

两组距离采样预测默认启用。例如只传 `{"ef_search":40}`，或传 `{}` 接受 wrapper 的全部默认配置，即可检索，无需填写上述相关性字段。仍可显式传参覆盖默认值。普通 `Index::KnnSearch` 不会因为传入这些参数而自动创建 `GlobalMCIHybrid`。
