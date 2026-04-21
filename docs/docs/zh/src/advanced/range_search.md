# 范围搜索

除了 k-近邻搜索（`KnnSearch`），VSAG 还支持**范围搜索**（`RangeSearch`）：返回所有与查询向量距离
小于或等于指定半径的结果。该接口适用于阈值过滤、去重、近似召回等场景。

## 基本用法

```cpp
#include <vsag/vsag.h>

// 1. 构造索引（以 HNSW 为例）
auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_params).value();
index->Build(dataset);

// 2. 准备查询
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(dim)->Float32Vectors(query_vec)->Owner(false);

// 3. 范围搜索
float radius = 0.5f;
auto result = index->RangeSearch(query, radius, search_params);
if (result.has_value()) {
    auto ids = result.value()->GetIds();
    auto dists = result.value()->GetDistances();
    int64_t n = result.value()->GetDim();
    // ...
}
```

> 完整示例参见 `examples/cpp/302_feature_range_search.cpp`。

## `limited_size` 参数

`RangeSearch` 支持通过 `limited_size` 限制返回结果的最大数量：

```cpp
// 返回最多 100 条满足半径条件的结果
auto result = index->RangeSearch(query, radius, search_params, /*limited_size=*/100);
```

- `limited_size = -1`（默认）：返回所有满足条件的结果（不限）。
- `limited_size > 0`：在满足半径条件的候选中返回最多这么多条。
- `limited_size = 0`：非法取值，实现中会显式拒绝
  （`CHECK_ARGUMENT(limited_size != 0, ...)`）。

## 与 Filter 组合

`RangeSearch` 的签名与 `KnnSearch` 一致，同样支持传入过滤器（见 `examples/cpp/301_feature_filter.cpp`）。
过滤器在搜索过程中即时生效，而不是事后过滤，效率更高。

## 支持情况

| 索引类型 | 支持 RangeSearch |
|---------|-----------------|
| hnsw | 是 |
| hgraph | 是 |
| diskann | 是 |
| ivf | 是 |
| brute_force | 是 |
| sindi | 稀疏向量场景支持 |

## 注意事项

- 距离度量（内积 / L2 / 余弦）会影响 `radius` 的语义。请与索引创建时的 `metric_type` 保持一致。
- 当 `radius` 过大时结果集可能巨大，建议配合 `limited_size` 使用。
- HNSW / HGraph 类图索引下，`RangeSearch` 的 `ef` 等运行期参数与 `KnnSearch` 共享含义。
