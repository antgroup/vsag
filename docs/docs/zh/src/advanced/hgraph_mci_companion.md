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

1. ADD 先完成 HGraph 插入，再逐个维护成功插入点。候选来自 `search_mci_knn`，内部调用
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
