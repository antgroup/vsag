# 搜索归因分析（召回诊断）

搜索归因是一种逐查询的诊断机制，用于解释每个期望的标签是否被召回以及原因。通过填充
`SearchRequest::expected_labels_` 并调用 `SearchWithRequest` 来启用；结果数据集会携带一
份 JSON 报告，可通过 `Dataset::Reasoning()` 获取。

## 快速开始

```cpp
// 填充关注的标签（最多 256 个）
SearchRequest request;
request.expected_labels_ = {100, 200, 300};
request.query_ = query_dataset;
request.topk_ = 10;
request.params_str_ = R"({"sindi_v2": {...}})";

auto results = index->SearchWithRequest(request);
std::cout << results->Reasoning() << std::endl;
```

## 报告格式 (v1)

报告是一个包含两个顶级分组的 JSON 对象：

### `expected_analysis`

| 字段 | 类型 | 描述 |
|------|------|------|
| `summary` | string | "M/N expected labels found, P missed" |
| `missed_targets` | array | 每个缺失标签一个条目 |

每个 `missed_targets` 条目：

| 字段 | 类型 | 描述 |
|------|------|------|
| `label` | int | 原始标签 |
| `inner_id` | int | 内部 ID |
| `diagnosis` | string | 下列诊断值之一 |
| `true_distance` | float | 真实距离 |
| `quantized_distance` | float | 搜索过程中观察到的近似距离 |
| `was_visited` | bool | 该 ID 是否被访问过 |
| `visited_at_hop` | int | 首次访问时的 hop 编号（-1 表示从未访问） |
| `was_evicted` | bool | 是否从候选集中被淘汰 |
| `filter_rejected` | bool | 是否被过滤器拒绝 |
| `reorder_evicted` | bool | 是否被重排序淘汰 |

### `meta`

| 字段 | 类型 | 描述 |
|------|------|------|
| `schema_version` | int | 始终为 1 |
| `status` | string | ok, skipped_range_search, unsupported_by_index, empty_index |
| `index_type` | string | HGraph, IVF, SINDI 等 |
| `search_mode` | string | knn 或 range |
| `topk` | int | 请求的 top k |
| `use_reorder` | bool | 是否启用了重排序 |
| `filter_active` | bool | 是否启用了过滤器 |
| `termination_reason` | string | none, lower_bound_reached, hops_limit_reached, timeout |
| `total_hops` | int | 总搜索跳数 |
| `total_distance_computations` | int | 总距离计算次数 |
| `available_diagnoses` | [string] | 引擎可产生的所有诊断类型 |
| `available_events` | [string] | 本次搜索中索引支持的事件 |
| `supports_range` | bool | 该索引是否支持范围搜索归因 |

## 诊断类型

| 值 | 含义 |
|----|------|
| `success` | 标签出现在结果集中 |
| `not_reachable` | 期望的目标从未被访问 |
| `filter_rejected` | 过滤器阻止了该条目 |
| `quantization_error` | 近似距离与真实距离偏差过大 |
| `ef_too_small` | 被访问但被淘汰（尝试增大 ef） |
| `reorder_evicted` | 重排序阶段被淘汰 |
| `unknown` | 无法确定原因 |

对于未命中的目标，诊断按首个匹配规则返回：未访问、过滤拒绝、量化误差、
候选淘汰、重排淘汰，最后为 unknown。因此同时满足条件时，`quantization_error`
优先于 `ef_too_small`。量化检查沿用 `近似距离 > 1.5 * 真实距离` 的启发式，
且仅在真实距离为正时适用；它不是与距离度量无关的误差界，也不能证明量化导致了
未命中，尤其应谨慎解释可能为负的 IP 距离。

`supports_range` 是布尔值，表示已注册的范围搜索归因支持，不代表索引是否提供
RangeSearch API；未注册的索引类型返回 false。

## 各索引事件支持

| 索引 | KNN | Range | Visit | Eviction | Filter Reject | Reorder | Reorder Eviction | Bucket Selection |
|------|-----|-------|-------|----------|---------------|---------|------------------|------------------|
| HGraph | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| IVF | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SINDI | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| SINDI_V2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BruteForce | ✓ | ✓ | ✓ | — | ✓ | — | — | — |
| WARP | ✓ | ✓ | ✓ | — | ✓ | — | — | — |
| Pyramid | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |

## 扩展指南

1. 新增诊断类型：在 `reasoning_types.h` 中添加 `ReasoningDiagnosis` 枚举值，在 `ToString`
   中映射，在 `reasoning_context.cpp` 中添加诊断规则，添加测试并更新文档。
2. 新增事件类型：在 `reasoning_types.h` 中添加 `ReasoningEvent` 枚举值，在
   `ReasoningContext` 中实现 `RecordXxx` 方法，用 `// [reasoning]` 标记埋点位置，
   更新能力矩阵并补充文档。
3. 新索引接入：在 `reasoning_capability.cpp` 中添加能力条目，实现 `SearchWithRequest`，
   插入归因记录钩子。