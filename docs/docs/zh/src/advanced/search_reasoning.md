# 搜索推理（Search Reasoning）

VSAG 的搜索推理机制用于解释**为什么特定的期望向量出现在（或没有出现在）搜索结果中**。当你在搜索
请求中指定一组期望 label 时，索引会记录每个期望向量在搜索过程中经历了什么——是被访问过、被过滤器
拒绝、被候选堆淘汰，还是根本没有被扫描到——并把一份结构化的 JSON 报告附着在搜索结果上。

## 什么时候用

用于召回问题排查、召回率回归工具建设，或回答"这个向量应该是 top-1，为什么没搜到"这类问题。

## 如何触发

1. 构造 `SearchRequest`，把期望命中的 label 放进 `expected_labels_`。
2. 调用索引的 `SearchWithRequest(request)`。
3. 通过返回 Dataset 的 `GetReasoning()` 读取报告。

```cpp
vsag::SearchRequest request;
request.query_ = query_dataset;              // 单条查询向量
request.topk_ = 10;
request.params_str_ = R"({"hgraph": {"ef_search": 100}})";
request.expected_labels_ = {42, 1024};       // 期望命中的 label

auto result = index->SearchWithRequest(request);
if (result.has_value()) {
    const std::string& reasoning = result.value()->GetReasoning();  // JSON 报告
}
```

`expected_labels_` 为空时搜索行为与结果完全不变——只有给定期望 label 时才会产生报告。

## 报告格式

JSON 报告包含两个部分：

```json
{
  "expected_analysis": {
    "summary": "1/2 expected labels found, 1 missed",
    "missed_targets": [
      {
        "label": 1024,
        "inner_id": 1007,
        "diagnosis": "quantization_error",
        "true_distance": 0.11,
        "quantized_distance": 0.24,
        "was_visited": true,
        "visited_at_hop": 2,
        "was_evicted": false,
        "filter_rejected": false,
        "reorder_evicted": false
      }
    ]
  },
  "meta": {
    "schema_version": 1,
    "status": "ok",
    "index_type": "HGraph",
    "search_mode": "knn",
    "topk": 10,
    "use_reorder": true,
    "filter_active": false,
    "termination_reason": "lower_bound_reached",
    "total_hops": 35,
    "total_distance_computations": 1840,
    "available_diagnoses": ["success", "not_reachable", "..."],
    "available_events": ["visit", "eviction", "..."],
    "supports_range": false
  }
}
```

- `expected_analysis.summary` — 期望 label 的整体命中情况。
- `expected_analysis.missed_targets` — 每个未命中的期望 label 一条记录，包含诊断结论与证据链
  （访问 / 淘汰 / 过滤 / 重排标记及距离）。
- `meta` — 本次搜索的上下文（索引、模式、参数）与终止统计。

## 诊断结论

针对每个未命中的期望 label 报告（按优先级，首个命中即返回）：

| 诊断 | 含义 |
|---|---|
| `not_reachable` | 该向量所在的窗口/桶在搜索中从未被扫描。 |
| `filter_rejected` | 被激活的过滤器排除。 |
| `quantization_error` | 近似距离高估了真实距离（约 true × 1.5），输给了分数更好的候选。 |
| `ef_too_small` | 被访问过但在候选堆竞争中落败；可尝试调大 `ef_search` / `n_candidate`。 |
| `reorder_evicted` | 近似排序存活，但在精确重排阶段被丢弃。 |
| `success` | 不适用——用于实际命中的期望 label。 |
| `unknown` | 以上均无法判定。 |

## 各索引能力矩阵

并非所有索引记录同样的证据。`src/impl/reasoning/reasoning_context.cpp` 中的注册表
（`kReasoningCapabilities`）是唯一事实来源：

| 索引 | KNN | Range | 事件 |
|---|---|---|---|
| HGraph | ✓ | –（跳过） | visit, eviction, filter_reject, reorder, reorder_eviction |
| IVF | ✓ | ✓ | visit, eviction, filter_reject, reorder, reorder_eviction, bucket_selection |
| SINDI | ✓ | ✓ | visit, filter_reject, reorder, reorder_eviction, bucket_selection |
| SINDI_V2 | ✓ | ✓ | visit, eviction, filter_reject, reorder, reorder_eviction, bucket_selection |
| BruteForce | ✓ | ✓ | visit, filter_reject |
| WARP | ✓ | ✓ | visit, filter_reject |
| Pyramid | ✓ | –（跳过） | visit, eviction, filter_reject, reorder, reorder_eviction |

说明与限制：

- **HGraph 的 range 搜索**带期望 label 时只返回最小状态报告
  （`meta.status = "skipped_range_search"`），不做逐目标分析——该索引的推理仅支持 KNN。
- **SINDI_V2** 通过用搜索时完全相同的剪枝后查询词对期望目标重新打分来重建访问情况；得分为 0
  表示没有共享词（`not_reachable`）。
- **SINDI_V2 KNN 搜索**会在被访问的期望目标未进入最终候选池时记录近似淘汰，并将精确重排的
  准入失败/堆淘汰单独记录。
- 对空 SINDI_V2 索引发起带期望 label 的请求时，返回 `meta.status = "empty_index"`。
- `termination_reason` 可为 `none`、`lower_bound_reached`、`hops_limit_reached` 或 `timeout`，
  取决于索引支持情况。

## 扩展推理（面向维护者）

全部枚举定义、能力注册表，以及"如何新增诊断/事件"的操作指南位于
`src/impl/reasoning/reasoning_context.h` / `.cpp`。代码库中每个插桩点都带有 `// [reasoning]`
注释，因此 `grep -rn "\[reasoning\]" src/` 可以枚举全部采集点。
