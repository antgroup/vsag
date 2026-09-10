# HGraph 与 Pyramid 自适应邻居剪枝

HGraph 提供默认关闭的实验性自适应选择器，用于 NSW 底层构图。它改变 `Build` 和 `Add` 产生的图，包括 RaBitQ split/fused 优化构建路径；查询时不执行此策略。公开参数与默认值见 [HGraph 配置](hgraph.md#实验性自适应剪枝)。

Pyramid 的 NSW 底图复用同一个选择器：策略全局共享，基准 alpha 使用所属 hierarchy 的有效值。详见 [Pyramid 配置与范围](pyramid.md#实验性-nsw-自适应剪枝)。Pyramid 路由层不启用自适应；此扩展也不会将 HGraph 的 RaBitQ 临时构图存储引入 Pyramid。

## 作用范围与兼容性

- `enabled=false` 保留原选择器，包括候选数量小于度数上限时的直接返回路径。
- `enabled=true` 用于新节点正向选边。额外设置 `apply_to_reverse=true`，才对已满的底层反向邻居表使用自适应剪枝；未满的邻居表仍直接追加。
- 反向剪枝以已有邻居为中心，候选为它的旧邻居与新插入节点。选择及写回仍在该已有邻居的原节点锁内完成。
- 上层图和更新／细化操作保留原选择器。ODescent、导入缓存构建、非 L2 度量，以及启用状态下的 `apply_to_upper=true` 均被拒绝。
- 序列化记录策略。加载时 enabled 必须一致；启用时基础 alpha、步长、补边及作用范围也必须一致。旧参数对象缺少此字段时按关闭处理；这不保证旧版 VSAG 二进制能读取新版索引。

## 距离判定

设 `u` 是选边中心，`c` 是候选，`s` 是已选邻居，`K` 是当前图的目标度数。距离来自原有构图 distance provider；开启此策略不切换量化器，也不额外保留原始向量。支持的 L2 路径使用平方 L2 距离，可能由构图量化器估计。

若存在已选邻居 `s` 满足下式，则拒绝 `c`：

$$
\alpha\,d(s,c) < d(u,c).
$$

等号不拒绝。增大 alpha 放宽单次判定，减小 alpha 收紧单次判定。但选边是贪心过程，先前选择会影响后续候选，因此不能据此推断不同 alpha 的最终邻居集合总是包含关系。

候选按 `(距离, 内部 ID)` 排序，去除自环和重复 ID；同一 ID 保留最近的一次。距离必须有限且非负。不同 ID 的相同向量不会被此选择器去重。

## 自适应规则

`alpha0` 为 HGraph 原有 `alpha` 参数，`delta` 为 `adjust_step`。一次扫描依次接受通过判定的候选，直到达到 `K` 或输入耗尽。拒绝的候选单独记录；未扫描的尾部不计入拒绝数。

1. 在 `alpha0` 下扫描规范化候选，得到接受列表 `A` 和拒绝列表 `B`。
2. 如果 `delta=0`，按补边开关从拒绝列表补至 `K`，然后返回。这不等价于关闭策略。
3. 如果 `0 < |A| < K`，保留 `A`，用放宽后的 alpha 重扫 `B`：

| 比例 `K / |A|` | 放宽后的 alpha |
| --- | --- |
| `<= 1.5` | `alpha0 + delta` |
| `> 1.5` 且 `<= 3` | `alpha0 + 2*delta` |
| `> 3` | `alpha0 + 3*delta` |

若 `fill_rejected=true`，按扫描顺序追加仍被拒绝的候选，直到达到 `K` 或候选耗尽。此补边不再检查几何判定式。

4. 如果 `|A| = K`，根据首轮拒绝数选择收紧后的 alpha：

| 比例 `|B| / K` | 收紧后的 alpha |
| --- | --- |
| `>= 5` | `alpha0` |
| `>= 2.5` 且 `< 5` | `alpha0 - delta` |
| `< 2.5` | `alpha0 - 2*delta` |

清空 `A`，使用收紧后的 alpha 重扫**完整的原始规范化列表**，包括首轮未扫描的尾部。若不足 `K`，保留当前接受项，在 `alpha0` 下重扫新拒绝列表。即使 `fill_rejected=true`，此分支也不无条件补边。

返回列表按 `(距离, 内部 ID)` 排序。空候选、仅自环或 `K=0` 返回空邻居。算法可能返回不足 `K` 个邻居，不凭空生成候选。

配置要求 alpha0、delta 有限，`delta >= 0`、`alpha0 - 2*delta > 0` 且 `alpha0 + 3*delta` 有限。GIST 参考配置 `alpha0=1.06, delta=0.06` 对应放宽值 1.12/1.18/1.24，收紧值 1.06/1.00/0.94。这是显式实验参数，不是按维度设置的默认值。

## 几何示例

使用平方 L2，中心 `u=(0,0)`，候选 `a=(1,0)`、`b=(0.5,1)`、`c=(-2,0)`，`K=2`、`alpha0=1.06`、`delta=0.06`。

首轮接受 `a,b`：`d(a,b)=d(u,b)=1.25`，随后达到上限，尚未扫描 `c`。拒绝数为 0，因此收紧至 0.94。第二轮先接受 `a`，因 `0.94*1.25 < 1.25` 拒绝 `b`，再因 `0.94*9 >= 4` 接受原先未扫描的 `c`。最终邻居为 `a,c`，说明收紧时必须保留首轮未扫描尾部。

## 开销与性能解释

对 `C` 个候选，规范化排序耗时为 `O(C log C)`；常数轮扫描进行 `O(CK)` 次两点距离比较，每次成本取决于构图 distance provider。临时空间局限于一次选边，包括候选／拒绝列表、ID 集合和按源节点缓存的 distance computer。没有全局自适应状态，也没有查询阶段的策略开销。当前辅助类缓存的是 distance computer，不是所有点对距离结果。

额外扫描增加构建成本，频繁裁剪已满反向邻居表时尤其明显。查询阶段，相同 `efSearch` 不保证展开节点数、扫描边数或 RaBitQ 精化次数相同。改变图结构可能在同一 efSearch 下提高召回、降低 QPS，应分别比较 recall–QPS 曲线、构建耗时及内存。

## 相关研究与实现来源

上述按计数双向调整的具体规则，参考了从 Descartes `ClassicSelector` 二进制恢复的行为。VSAG 的接入范围、配置、校验和确定性候选规范化属于本次适配。这不是完整复现 Descartes 索引，也不声称首创自适应邻居选择。

- [Descartes](https://github.com/01-ai/Descartes) 公开介绍了自适应邻居选择，但 README 没有给出上述数值分段规则。
- [HNSW Algorithm 4](https://arxiv.org/pdf/1603.09320) 提供从拒绝候选补边的 `keepPrunedConnections`，不包含这套 alpha 调度。
- [Vamana 构图](https://intel.github.io/ScalableVectorSearch/advanced/graph_search.html) 使用 alpha 剪枝和两轮全图构建，与逐节点反馈不同。
- [SymphonyQG §3.2.2](https://arxiv.org/html/2411.12229v1#S3.SS2.SSS2) 按节点调整角度阈值补边，使出度对齐 FastScan 批大小。
- [Alpha-CNG §4.2、Algorithm 4](https://arxiv.org/html/2510.05975v1#S4.SS2) 逐节点增大 alpha，直到达到度数阈值或耗尽配置的 alpha 范围；其剪枝公式额外包含 tau 项，公开算法不使用本实现的接受／拒绝比例收紧规则。

这些是已公开的相关技术，不构成对精确常数的来源链或首创归属的证明。

## 测试覆盖

`[adaptive_pruning]` 测试包括比例边界、严格不等式、补边、零步长、自环／重复候选规范化、非法输入、真实 L2 几何与候选排列稳定性、正向／反向范围、缓存拒绝、参数兼容，以及 1/4 线程下 FP32、RaBitQ split/fused 的构建—序列化—加载—追加流程。周边回归标签为 `[pruning_strategy]`、`[HGraphParameter]`、`[hgraph]` 和 `[build_cache]`。

Pyramid 接入测试另覆盖平面节点升级、单层/多层根图、根图与路径查询、FP32/RaBitQ + SQ8 在 1/4 线程下构建—加载—继续 Add、导入缓存拒绝、hierarchy alpha 校验及序列化策略兼容性。
