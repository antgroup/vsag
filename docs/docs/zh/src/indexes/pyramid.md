# Pyramid

Pyramid 是 VSAG 的 **层级路径分区** 图索引。每条向量都附带一个路径字符串（例如
`"a/d/f"`），Pyramid 会按路径树为每个节点构建一个子图；查询时提供一个路径前缀，
检索即被限定在相应的子树内。

这种设计非常适合多租户部署、标签分区的物料库，或者任何“一个逻辑索引服务多个群体、
群体之间不允许结果交叉”的场景。

- 源码：`src/algorithm/pyramid.{h,cpp}`、`src/algorithm/pyramid_zparameters.{h,cpp}`
- 示例：[`examples/cpp/107_index_pyramid.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/107_index_pyramid.cpp)

## 工作原理

1. **路径树。** 每条向量在 ID 之外还携带一个 `path`，分隔符为 `/`
   （例如 `"tenant_a/lang_en/topic_news"`）。Pyramid 会为构建期间出现过的每个路径前缀
   维护一个子索引。
2. **按层构建子图。** 默认情况下每一层都会独立构建一张近邻图。可以用 `no_build_levels`
   跳过那些太小或太粗、不适合构图的层级——这些层级仍作为透传容器存在，但检索会退化为
   线性扫描。
3. **图的构建。** 每个子图与 HGraph 采用同一套机制：`nsw` 插入或 `odescent`，并通过
   `graph_iter_turn`、`neighbor_sample_rate`、`alpha` 控制构图剪枝。底层向量按
   `base_quantization_type` 存储；启用精排时另外保留一份高精度副本。
4. **检索。** 查询向量同样要附带路径。搜索会顺路径树向下走到最具体匹配查询路径的子图，
   然后在该子图内执行图检索（`ef_search`；中间层由 `subindex_ef_search` 控制）。

## 快速开始

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "alpha": 1.2,
        "graph_type": "odescent",
        "graph_iter_turn": 15,
        "neighbor_sample_rate": 0.2,
        "no_build_levels": [0, 1],
        "use_reorder": true,
        "build_thread_count": 16
    }
})";
auto index = vsag::Factory::CreateIndex("pyramid", params).value();

// 构建时为每条向量提供路径。
auto base = vsag::Dataset::Make();
base->NumElements(n)
    ->Dim(128)
    ->Ids(ids)
    ->Paths(paths)          // std::string* 长度为 n，例如 "a/d/f"
    ->Float32Vectors(data)
    ->Owner(false);
index->Build(base);

// 按路径前缀执行检索。
std::string query_path = "a/d";
auto query = vsag::Dataset::Make();
query->NumElements(1)
    ->Dim(128)
    ->Float32Vectors(q)
    ->Paths(&query_path)
    ->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10,
    R"({"pyramid": {"ef_search": 100}})").value();
```

## 构建参数

构建参数放在 `index_param` 下。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_quantization_type` | string | — | 底层量化类型（`fp32`、`fp16`、`bf16`、`sq8`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq`） |
| `max_degree` | int | `64` | 子图内节点的最大出度 |
| `graph_type` | string | `"nsw"` | `nsw` 或 `odescent` |
| `ef_construction` | int | `400` | `nsw` 构图时的候选集大小 |
| `alpha` | float | `1.2` | 构图剪枝系数 |
| `graph_iter_turn` | int | — | ODescent 迭代轮数（`graph_type: "odescent"` 时生效） |
| `neighbor_sample_rate` | float | — | ODescent 的邻居采样比率 |
| `no_build_levels` | int[] | `[]` | 跳过构图的层级（从根节点开始的 0-based 下标） |
| `use_reorder` | bool | `false` | 是否保留高精度副本用于精排 |
| `precise_quantization_type` | string | `"fp32"` | 精排使用的量化类型 |
| `index_min_size` | int | `0` | 子索引的最小规模；小于该值的分区会退化为线性扫描 |
| `support_duplicate` | bool | `false` | 是否允许重复 ID |
| `build_thread_count` | int | `1` | 构建阶段并发线程数 |

## 检索参数

检索参数放在 `pyramid` 子对象下：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ef_search` | int | `100` | 叶子层子图检索的候选集大小 |
| `subindex_ef_search` | int | `50` | 沿路径向下遍历中间子图时的候选集大小 |

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"pyramid": {"ef_search": 200, "subindex_ef_search": 80}})").value();
```

## 何时选择 Pyramid

- 多租户服务：每个租户只能看到自己分区的结果，且希望避免为每个租户单独维护一份索引。
- 带有层级标签的物料库（语言 / 地域 / 品类），查询永远限定在某个已知的前缀下。
- 小分区非常多的负载：可以用 `no_build_levels` 与 `index_min_size` 跳过那些小到不值得
  构图的分区。

如果不需要按路径限定查询范围，[HGraph](hgraph.md) 更简洁，性能通常也更高。

## 相关文档

- [创建索引](../guide/create_index.md)
- [索引参数](../resources/index_parameters.md)
- [图索引增强](../advanced/enhance_graph.md)
- [HGraph](hgraph.md)
