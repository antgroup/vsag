---
title: "为 HGraph 透出 skip_ratio / skip_strategy 搜索参数"
status: in_progress
priority: high
created_by: "opencode"
assigned_to: "opencode"
created_at: "2026-06-29"
updated_at: "2026-06-29"
target_repo: "antgroup/vsag"
related_files:
  - "src/algorithm/hgraph/hgraph_parameter.h"
  - "src/algorithm/hgraph/hgraph_parameter.cpp"
  - "src/algorithm/hgraph/hgraph_search.cpp"
  - "src/algorithm/hgraph/hgraph_parameter_test.cpp"
  - "src/constants.cpp"
  - "docs/docs/zh/src/resources/index_parameters.md"
  - "docs/docs/en/src/resources/index_parameters.md"
pr_url: ""
---

# 为 HGraph 透出 skip_ratio / skip_strategy 搜索参数

## 背景

PR #2314 修复了 HGraph KNN 搜索中 filter 重复检查与图连通性问题，引入了 `FilterSearchSkipStrategy` 和 `ShouldVisit()` 机制。该机制在底层 `BasicSearcher`/`ParallelSearcher` 中已完全可用，HGraph 的搜索路径也确实在使用它——但 `skip_ratio` 和 `skip_strategy` 从未被 `HGraphSearchParameters` 解析，用户无法通过公开 JSON 参数覆盖默认值。

当前 HGraph 一直使用 `InnerSearchParam` 的硬编码默认值 `skip_ratio = 0.8F`，而 HNSW 索引从 v0.17.0 起就已经透出了 `skip_ratio`（默认 0.9）和 `skip_strategy`。

## 问题描述

### 1. 参数未透出

`HGraphSearchParameters`（`src/algorithm/hgraph/hgraph_parameter.h:76-97`）没有 `skip_ratio` 和 `skip_strategy_type` 字段，`FromJson()` 也不解析这两个参数。

### 2. 底层已支持

搜索路径 `HGraph::SearchWithRequest` → `search_one_graph` → `BasicSearcher::Search` / `ParallelSearcher::Search` 会将 `InnerSearchParam` 完整传递给 searcher，searcher 内部使用 `inner_search_param.skip_ratio` 创建 `FilterSearchSkipStrategy` 并调用 `ShouldVisit()`。唯一的缺口是 `HGraphSearchParameters` 没有将外部参数赋值给 `search_param.skip_ratio`。

### 3. 默认值不一致

| 索引 | skip_ratio 默认值 |
|------|-------------------|
| HNSW | 0.9（`hnsw_zparameters.h:64`） |
| HGraph | 0.8（`inner_search_param.h:46`） |

### 4. 历史版本情况

| 版本 | HNSW skip_ratio | HGraph skip_ratio |
|------|:---:|:---:|
| v0.17.0 | 已透出 | 未透出（底层已支持，默认 0.8F） |
| v0.18.0 | 已透出 | 未透出（同上） |
| v0.18.6 | 已透出 | 未透出（同上） |
| main | 已透出 | 未透出（同上） |

HGraph 从 v0.17.0 起就通过 `BasicSearcher` 内部使用了 skip_ratio 机制（PR #499, 2025-03-13），但始终没有对外暴露。

## 改进方案

### 1. 修改 `HGraphSearchParameters`

在 `src/algorithm/hgraph/hgraph_parameter.h` 中添加字段：

```cpp
class HGraphSearchParameters : public IndexSearchParameter {
public:
    // ... 现有字段 ...
    float skip_ratio{0.8F};
    FilterSearchSkipStrategyType skip_strategy_type{
        FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE};
};
```

### 2. 解析参数

在 `src/algorithm/hgraph/hgraph_parameter.cpp` 的 `FromJson()` 中添加解析：

```cpp
if (params[INDEX_TYPE_HGRAPH].Contains(HNSW_PARAMETER_SKIP_RATIO)) {
    obj.skip_ratio = params[INDEX_TYPE_HGRAPH][HNSW_PARAMETER_SKIP_RATIO].GetFloat();
}
if (params[INDEX_TYPE_HGRAPH].Contains(HNSW_PARAMETER_SKIP_STRATEGY)) {
    obj.skip_strategy_type = parse_filter_search_skip_strategy_type(
        params[INDEX_TYPE_HGRAPH][HNSW_PARAMETER_SKIP_STRATEGY].GetString());
}
```

### 3. 传递到 InnerSearchParam

在 `src/algorithm/hgraph/hgraph_search.cpp` 中，将参数赋值到 `search_param`：

```cpp
search_param.skip_ratio = params.skip_ratio;
search_param.skip_strategy_type = params.skip_strategy_type;
```

注意：需要在所有构建 `InnerSearchParam` 的路径中都赋值（`KnnSearch` 迭代器路径和 `SearchWithRequest` 路径）。

### 4. 添加测试

在 `src/algorithm/hgraph/hgraph_parameter_test.cpp` 中添加测试用例，验证 JSON 参数解析正确。

### 5. 更新文档

在 `docs/docs/zh/src/resources/index_parameters.md` 和英文版中补充 `skip_ratio` 和 `skip_strategy` 参数说明。

### 6. 多版本支持

**此改动需要在以下版本分支上都实施：**

- `main`（当前开发分支）
- `v0.18` 分支（cherry-pick）
- `v0.17` 分支（cherry-pick，如果仍维护）

HGraph 从 v0.17.0 起就内部支持了 skip_ratio，但从未透出。用户可能在任何版本上都需要调整这个参数来优化 filter 场景下的搜索质量。

## 预期效果

1. 用户可以通过 JSON 搜索参数 `{"hgraph": {"skip_ratio": 0.7}}` 控制 filter 跳过比例
2. HGraph 和 HNSW 在 filter 场景下的行为可统一调优
3. 与 HNSW 的参数体系保持一致，降低用户认知成本
4. 保持向后兼容（不传参时行为与当前一致，默认 0.8F）

## 参考资料

- HNSW 参数透出实现：`src/index/hnsw_zparameters.cpp:117-127`
- HNSW 常量定义：`src/constants.cpp:112-113`
- InnerSearchParam 默认值：`src/impl/inner_search_param.h:46-47`
- BasicSearcher 使用 skip_ratio：`src/impl/searcher/basic_searcher.cpp:163`
- PR #2314（filter 连通性修复）：https://github.com/antgroup/vsag/pull/2314
- PR #499（searcher 层引入 skip_ratio）：2025-03-13
- Iterator search 文档中提及 skip_ratio：`docs/docs/zh/src/advanced/iterator_search.md`
