# LazyHGraph

LazyHGraph 是一个自适应的稠密向量索引：数据量较小时先使用精确的 BruteForce
索引，达到可配置的 `transition_threshold` 后自动转换为 HGraph。它适合“初始规模较小、
后续持续增长”的集合：早期查询保持精确并避免构图开销，规模变大后获得 HGraph 的近似检索
延迟与量化能力。

- 源码：`src/algorithm/lazy_hgraph.{h,cpp}`
- 示例：
  [`examples/cpp/111_index_lazy_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/111_index_lazy_hgraph.cpp)

## 工作方式

1. **Flat 阶段。** 达到阈值前，数据存放在内部 BruteForce 索引中，使用 FP32 向量，
   搜索结果是精确的。
2. **转换。** 当 `Build` 收到不少于 `transition_threshold` 条向量，或 `Add` 让 flat
   阶段增长到该规模时，LazyHGraph 会用 flat 数据构建内部 HGraph。
3. **Graph 阶段。** 转换完成后，新增数据与查询都交给内部 HGraph 处理。搜索参数仍使用
   `hgraph` 子对象。

## 快速开始

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "lazy_hgraph": {
        "transition_threshold": 1000,
        "hgraph": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100,
            "build_thread_count": 4
        }
    }
})";
auto index = vsag::Factory::CreateIndex("lazy_hgraph", params).value();

auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Add(base);

auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10, R"({"hgraph": {"ef_search": 100}})").value();
```

## 构建参数

LazyHGraph 的构建参数放在顶层 `lazy_hgraph` 对象中。为了兼容通用工厂参数形态，
同一个对象也可以放在 `index_param` 中。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `transition_threshold` | uint64 | `1000` | 从 flat 阶段转换到 HGraph 的向量数量阈值，必须为正数。 |
| `hgraph` | object | `{}` | 转换后内部 HGraph 使用的构建参数。见 [HGraph](hgraph.md#构建参数)。 |

LazyHGraph 只支持顶层 `dtype: "float32"`。flat 阶段固定使用 FP32 BruteForce
存储，不接受单独的 flat 量化参数。

## 搜索参数

搜索参数与 HGraph 一样，使用 `hgraph` 子对象：

```json
{"hgraph": {"ef_search": 100}}
```

在 flat 阶段，搜索是精确的；进入 graph 阶段后，内部 HGraph 会使用传入的 HGraph
搜索参数，例如 `ef_search`。

## 生命周期说明

- `Build` 会根据输入规模选择初始阶段：小于 `transition_threshold` 保持 flat；
  大于等于阈值则直接构建 HGraph。
- `Add` 可能触发从 flat 到 graph 的单向转换。
- flat 阶段的 `Remove` 始终执行物理删除，即使调用方传入 `RemoveMode::MARK_REMOVE`，
  这样后续图转换不会携带 tombstone。
- `GetExtraInfoByIds`、`UpdateExtraInfo` 与基于 extra_info 的过滤在两个阶段都支持。
  见 [Extra Info](../advanced/extra_info.md)。

## 何时使用 LazyHGraph

- 稠密 FP32 集合初始较小，并会持续增长。
- 集合较小时希望获得精确结果。
- 希望同一个索引在规模变大后自动切换到 HGraph。

如果数据在构建时已经很大、需要非 FP32 输入类型，或希望从第一条插入开始就使用图索引行为，
请直接使用 [HGraph](hgraph.md)。

## 相关文档

- [HGraph](hgraph.md)
- [Extra Info](../advanced/extra_info.md)
- [创建索引](../guide/create_index.md)
