---
title: "打通 SearchWithRequest iterator search（P0：HGraph 打通主路径 + 验证 + 文档）"
status: in_progress
priority: high
created_by: "lead"
assigned_to: "lead"
created_at: "2026-11-10"
updated_at: "2026-11-10"
target_repo: "antgroup/vsag"
related_files:
  - "src/algorithm/hgraph/hgraph_search.cpp"
  - "src/algorithm/hgraph/hgraph.h"
  - "src/index/index_impl.h"
  - "include/vsag/search_request.h"
  - "include/vsag/index.h"
  - "include/vsag/iterator_context.h"
  - "tests/test_hgraph.cpp"
pr_url: ""
---

# 打通 SearchWithRequest iterator search

## 背景

VSAG 当前有 iterator search（迭代式搜索/翻页会话），允许通过多次 `KnnSearch` 调用逐页遍历
搜索结果。但实现仅通过旧版 `Index::KnnSearch(query, k, params, filter, iter_ctx, is_last_search)`
重载暴露，且只被 HGraph 实现。

`SearchRequest` 中已经定义了对应字段：
- `enable_iterator_search_` (bool)
- `p_iter_ctx_` (IteratorContext**)
- `is_last_search_` (bool)

但从 `IndexImpl::SearchWithRequest` 到各索引的 `SearchWithRequest` 实现，**没有任何地方消费这
三个字段**。它们是 dead code。

分析依据：`read-only-vsag/` 的 `github/main` (commit `a62661b3`)。

目标：让用户能通过统一的 `SearchWithRequest` API 使用 iterator search（仅限 HGraph），并为
未来扩展到其他索引打好基础。

## 问题描述

1. **SearchWithRequest 路径全空**：
   - `IndexImpl::SearchWithRequest` (`src/index/index_impl.h:519-661`) 不检查 iterator 字段
   - `HGraph::SearchWithRequest` (`src/algorithm/hgraph/hgraph_search.cpp`) 不处理 iterator
   - `SearchRequest` 三个字段 (`enable_iterator_search_`, `p_iter_ctx_`, `is_last_search_`)
     在整个代码库中从未被任何 `SearchWithRequest` 实现消费（只有 BruteForce 测试中构造了
     SearchRequest 并断言过这些字段，但没经过搜索路径）
   - Pyramid 的 `search_one_graph` 中有一次 `not request.enable_iterator_search_` 检查，
     但那是在非 iterator 路径上做排除，不是支持

2. **k>1 分页语义未验证**：
   - 现有测试 (`tests/test_hgraph.cpp`, `tests/test_hgraph_rabitq_split.cpp`) 全部 k=1
   - `tests/test_hgraph.cpp:1051-1069` 用 `kResultSize=10` 测了 iterator 路径，但只验证
     了「能取到结果」，没有验证分页语义（翻页不重复不遗漏、多页并集等于一次全量搜索）

3. **`is_last_search=true` drain 路径未经直接测试**：
   - 所有测试用 `is_last_search=false` + `delete iter_ctx` 来清理，destructor 有 drain 逻辑，
     但没有测试直接以 `is_last_search=true` 调用来验证 drain 行为

4. **无文档/示例说明如何用 SearchWithRequest 做 iterator search**

## 改进方案

### 1. HGraph::SearchWithRequest 消费 iterator 字段

在 `src/algorithm/hgraph/hgraph_search.cpp` 的 `HGraph::SearchWithRequest` 中：
- 检测 `request.enable_iterator_search_` 为 true 时的分流逻辑
- 从 `request.p_iter_ctx_` 获取/创建 `IteratorFilterContext`
- 复用现有 `KnnSearch(query, k, params, filter, allocator, iter_ctx, is_last_filter)` 的翻页逻辑
- iterator search 强制单 query（与现有 overload 一致），多 query + iterator 组合直接报错

### 2. IndexImpl::SearchWithRequest 透传 iterator 字段

在 `src/index/index_impl.h` 的 `SearchWithRequest` 中：
- `enable_iterator_search_` 为 true 时验证 `p_iter_ctx_` 不为 null
- 将 iterator 字段正确透传到 inner index 的 `SearchWithRequest`
- 非 HGraph 索引返回明确的 UNSUPPORTED 错误（当前基类默认抛异常）

### 3. 补充 k>1 分页语义测试

在 `tests/test_hgraph.cpp` 中新增测试：
- 小数据集 BruteForce baseline 验证（全部结果已知）
- k=5/10 多次 iterator search，每次取 k 条
- 验证：多页之间无重复、无遗漏、多页并集等于一次 `k=N` 搜索的全量结果

### 4. 补充 `is_last_search=true` drain 路径直接测试

- 测试 `is_last_search=true` 调用时 drain 剩余候选结果
- 验证 drain 后结果 + 之前翻页结果 = 全量
- 验证 drain 后 context 可安全销毁

### 5. 补 usage example / 文档

在 `SearchRequest` 字段注释附近或 `include/vsag/index.h` 增加 Iterator Search 使用示例。

## 预期效果

- 用户可通过 `SearchRequest.enable_iterator_search_` + `p_iter_ctx_` + `is_last_search_`
  使用 iterator search（HGraph）
- k>1 分页语义正确、有测试保证
- 不影响非 iterator 搜索的现有行为和性能
- 为后续 BruteForce/IVF 扩展建立统一的 API 入口

## 参考资料

- `read-only-vsag/` github/main `a62661b3`
- `src/algorithm/hgraph/hgraph_search.cpp:100-306` — 现有 iterator KnnSearch 实现
- `src/index/index_impl.h:519-661` — SearchWithRequest 主调度
- `include/vsag/search_request.h:188-212` — iterator 字段定义
- `tests/test_hgraph.cpp` — 现有 iterator 测试（k=1）
