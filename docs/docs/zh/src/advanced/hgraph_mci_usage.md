# HGraph MCI：代码与配置指南

本文对应 MCI 增删开发分支，聚焦纯 FP32 的构建、ADD、MARK_REMOVE、FORCE_REMOVE 和 Flush。
索引类型始终是 `hgraph`；MCI 是共享 HGraph 向量存储的团索引，不是另一份独立 HNSW。
算法细节及历史测量见 [增删实现与测试报告](hgraph_mci_mutation.md)。

> 曾复现的“加载后 FORCE_REMOVE”内存错误已定位为缺少图反向边恢复，并补充修复及回归测试。
> 历史性能测试从全量 Build 开始，没有新增大数据集快照加载复测。
> 新增 `Index::Flush()` 虚接口涉及 ABI，MCI 序列化格式升级为 v2；旧二进制兼容性需单独验证。

## 1. 代码入口

下表中的文件路径均相对于仓库根目录。

| 文件 | 重点入口 / 职责 |
| --- | --- |
| `include/vsag/index.h`、`src/index/index_impl.h` | 公共 Build/Add/Remove/Flush/Search 接口及错误包装 |
| `src/algorithm/hgraph/hgraph_build.cpp` | `Add` → `add_impl`：先插入 HGraph，再维护成功插入点的 MCI |
| `src/algorithm/hgraph/hgraph_mci.cpp` | 全量构团；`search_mci_knn`、`incremental_update_mci_clique`、`repair_mci_clique`、`force_remove_with_mci`、`Flush` |
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
| Flush | 不删除或移动向量 | 合并 delta、移除废弃成员、压紧团编号、重建两向 CSR | 否 |

增删和 Flush 共用 MCI mutation mutex 串行化。物理 ID 搬移及最终缩容需要独占保护；
修复阶段会释放 force-remove 锁，以便内部 HGraph 检索自行获取读锁，此时 MCI 尚未发布，
外部查询可能回退到 HGraph。不要把它理解成完整操作期间一直阻塞查询，也不要假设增删具备事务回滚。

## 2. 配置

### 2.1 Codefilter FP32 构建配置

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
    "mci_mcs": 50,
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

### 2.2 搜索配置

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

### 2.3 C++ 调用片段

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
check(index->Flush());  // 可选；不是持久化，FORCE_REMOVE 内部已经自动 Flush。
auto result = index->KnnSearch(query, 10, search_json, filter);
check(result);
```

Remove 使用外部标签，不接受把内部槽位当成标签；物理删除后不要缓存 inner ID 或团 ID。
完整的 Dataset 与 Filter 示例见 `examples/cpp/324_feature_hgraph_mci_companion.cpp`。

## 3. 测试工具范围

开发阶段使用的 benchmark 和脚本已移出本 PR。历史结果仍保留在测试报告中；
当前代码树不提供这些测试程序、构建目标或脚本。

## 4. 如何读结果

五阶段保护评测查询的 top-k 真值点，五次使用同一真值；随机增删不保护真值点，
每个检查点从精确全库排序中取当前存活 top-k。两种实验不能混用 recall 结论。

| CSV 字段 | 解释 |
| --- | --- |
| `stage`、`active_vectors`、`index_elements` | 阶段、存活数量、索引对外元素计数；后者不能代替物理存储统计 |
| `ef_search`、`recall_at_k`、`qps` | 搜索宽度、recall、计时吞吐；固定 ef 不等于固定质量 |
| `build_seconds`、`mutation_seconds`、`flush_seconds` | 构建、阶段增删、阶段后显式 Flush；FORCE_REMOVE 内部 Flush 已计入增删时间 |
| `index_memory_bytes`、`vector_memory_bytes`、`graph_memory_bytes`、`mci_memory_bytes` | 索引及分项统计，不是进程 RSS |
| `mci_route_ratio`、`mci_raw_float_ratio` | MCI 路由与直接 FP32 路径命中比例；不能只看配置判断路由 |
| `mci_total_cliques`、`mci_delta_cliques`、`mci_total_memberships` | 团数、增量团数、成员关系总数 |
| `avg_dist_cmp`、`avg_hops`、`avg_seed_count` | 距离计算、遍历及 seed 开销，用来分析 QPS 变化 |

MiB = bytes / 1,048,576。相同阶段不同 ef 行会重复阶段内存/耗时，不能相加。
QPS 是多线程吞吐，`1000 / QPS` 不是单请求延迟；当前 CSV 不含延迟分位数或逐阶段 RSS。

已完成五阶段结果及 ef=320 汇总见 [测试报告第 15 节](hgraph_mci_mutation.md#15-纯-fp32-五阶段复测与初始索引保存)。
本 PR 不包含原始结果文件；
这是适配新版 main 前的开发版本测量，不是当前 PR HEAD 的重新压测。

## 5. 回归测试与边界

```bash
make debug VSAG_ENABLE_TESTS=ON COMPILE_JOBS=12
build/tests/unittests '[mci],[LabelTable],RaBitQSplitDataCell serialize and methods'
```


文档编写前验证：C++ 定向测试 38 个用例、3,294 条断言通过；脚本测试 10+1+3 项通过，
其中两个集成测试当时使用 Debug 基准程序作正确性验证。未完成全量 lint、全量测试和 90% 覆盖率验证。
review 后另补充了反向边恢复及加载后 FORCE_REMOVE 回归。大数据集加载复测、ABI、序列化兼容
及更全面的并发修改仍需独立验证；不能把定向测试当作生产就绪证明。
