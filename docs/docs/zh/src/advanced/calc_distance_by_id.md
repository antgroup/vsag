# 按 ID 计算距离

除了 `KnnSearch` 和 `RangeSearch`，VSAG 还提供了在**已建好索引的向量上按 ID 计算距离**的
接口，可用于对外部候选集进行重排、召回核验，或在 VSAG 之上构建自定义检索流水线。

接口分为两种形式：

- `CalcDistanceById`  — 单个 ID，返回单个距离值。
- `CalDistanceById`   — 一批 ID，返回一个包含距离数组的 `DatasetPtr`。

每种形式都有两个重载：一个接收 `const float*`（稠密向量），另一个接收 `DatasetPtr`
（稠密或稀疏均可）。

> **关于命名的说明。** 批量接口目前拼作 `CalDistanceById`（`Calc` 少了一个 `c`）。
> 这是批量重载最初加入时遗留的拼写笔误，两个名字**并不表示语义差异**，区别仅在于
> *单个 vs. 批量*。出于向后兼容当前仍保留这一拼写，预计未来某个版本会将其
> **标记为弃用（deprecated）**，并改用拼写正确的新名（建议为
> `CalcDistancesById`）。建议新代码通过一层薄封装来调用，方便后续迁移。跟踪请见
> [issue #2068](https://github.com/antgroup/vsag/issues/2068)。

## 接口概览

```cpp
// 单个 ID，稠密浮点指针
tl::expected<float, Error>
CalcDistanceById(const float* vector,
                 int64_t id,
                 bool calculate_precise_distance = true) const;

// 单个 ID，DatasetPtr（稠密或稀疏）
tl::expected<float, Error>
CalcDistanceById(const DatasetPtr& vector,
                 int64_t id,
                 bool calculate_precise_distance = true) const;

// 批量 ID，稠密浮点指针
tl::expected<DatasetPtr, Error>
CalDistanceById(const float* query,
                const int64_t* ids,
                int64_t count,
                bool calculate_precise_distance = true,
                int64_t topk = -1) const;

// 批量 ID，DatasetPtr（稠密或稀疏）
tl::expected<DatasetPtr, Error>
CalDistanceById(const DatasetPtr& query,
                const int64_t* ids,
                int64_t count,
                bool calculate_precise_distance = true,
                int64_t topk = -1) const;
```
对于 `NumElements() > 1` 的 `DatasetPtr` 查询，请先检查
`SUPPORT_BATCH_CALC_DISTANCE_BY_ID`。`count` 表示每个 query 的 ID 数，`ids` 需要包含
`NumElements() * count` 个 row-major ID，返回距离使用相同布局。当 `topk > 0` 时，
每个 query 返回按距离升序排列的 `min(topk, count)` 个结果，并同时返回对应 ID。


声明位于
[`include/vsag/index.h`](https://github.com/antgroup/vsag/blob/main/include/vsag/index.h)。

### `calculate_precise_distance`

- `true`（默认）：尽量使用**高精度**向量表示（如完整 float32）来计算距离。当索引仅保留
  量化编码时，获取精确值可能开销更大。
- `false`：可以使用索引内存中已有的**量化 / 近似**表示，速度更快但距离是近似值。

### 返回值含义

- 单 ID 重载返回 `float` 距离值。
- `topk == -1` 时，裸指针批量重载返回一行 `count` 个距离；`DatasetPtr` 重载返回
  `NumElements()` 行、每行 `count` 个距离。两者都保持输入顺序，且不返回 ID。
- `topk > 0` 时，批量重载每个 query 返回按距离升序排列的最小 `min(topk, count)` 个
  距离，`GetIds()` 包含对应 ID。无效 ID（距离为 `-1`）排在有效距离之后，仅在有效 ID
  不足时才出现在结果中。
- 距离的语义由建索引时设置的 `metric_type`（IP / L2 / cosine）决定，参见
  [度量语义](../resources/metric_semantics.md)。

## 基本用法

```cpp
#include <vsag/vsag.h>

// 1. 构建 HGraph 索引
auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();
index->Build(base);

// 2. 单 ID 距离
auto d = index->CalcDistanceById(query_vector.data(), /*id=*/42);
if (d.has_value()) {
    std::cout << "distance to id 42 = " << d.value() << std::endl;
}

// 3. 批量 ID 距离
std::vector<int64_t> ids = { 1, 2, 3, 4, 5 };
auto result = index->CalDistanceById(query_vector.data(), ids.data(), ids.size());
if (result.has_value()) {
    const float* dists = result.value()->GetDistances();
    for (size_t i = 0; i < ids.size(); ++i) {
        if (dists[i] == -1.0f) {
            std::cout << ids[i] << " -> 无效 ID" << std::endl;
        } else {
            std::cout << ids[i] << " -> " << dists[i] << std::endl;
        }
    }
}
```

## 多查询

当索引公布 `SUPPORT_BATCH_CALC_DISTANCE_BY_ID` 能力时，`DatasetPtr` 批量重载可以一次接收
多个 query。候选 ID 和输出都采用 row-major 布局。对于 LazyHGraph，调用会委托给当前生效的
内部索引；即使 wrapper 公布了该能力，实际支持也可能在 phase 切换后改变，因此仍需处理返回错误：

```cpp
// 两个稠密 query，每个 query 三个候选 ID。
auto queries = vsag::Dataset::Make()
                   ->NumElements(2)
                   ->Dim(dim)
                   ->Float32Vectors(query_vectors.data())
                   ->Owner(false);
std::vector<int64_t> candidate_ids = {
    10, 11, 12,  // query 0 的候选
    20, 21, 22   // query 1 的候选
};

if (index->CheckFeature(vsag::SUPPORT_BATCH_CALC_DISTANCE_BY_ID)) {
    auto result = index->CalDistanceById(queries, candidate_ids.data(), 3, true, /*topk=*/2);
    if (result.has_value()) {
        auto batch = result.value();
        // batch 的 NumElements() == 2，Dim() == 2。
        for (int64_t q = 0; q < batch->GetNumElements(); ++q) {
            for (int64_t j = 0; j < batch->GetDim(); ++j) {
                const int64_t offset = q * batch->GetDim() + j;
                std::cout << batch->GetIds()[offset] << ": "
                          << batch->GetDistances()[offset] << '\n';
            }
        }
    }
}
```

`topk == -1` 时，`GetDim()` 等于 `count`，位置 `q * count + j` 对应输入 ID
`ids[q * count + j]`。`topk` 为正数时，行跨度为 `GetDim()`；每行先放最近的有效候选，
只有有效 ID 少于 `topk` 时才会在行尾保留无效 ID。

可运行的完整示例见
[`examples/cpp/306_feature_calculate_distance_by_id.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/306_feature_calculate_distance_by_id.cpp)。

## 稀疏向量

对于 SINDI 等稀疏向量索引，`const float*` 重载不适用。需要通过 `SparseVectors(...)` 把查询
封装为 `DatasetPtr`，并调用 `DatasetPtr` 重载：

```cpp
auto query = vsag::Dataset::Make();
query->NumElements(1)->SparseVectors(&sparse_query)->Owner(false);

auto d = index->CalcDistanceById(query, /*id=*/42);
```

## 支持矩阵

| 索引类型 | 单 ID 稠密重载（`const float*`） | 单 ID DatasetPtr | 多查询 DatasetPtr 批量重载 | 说明 |
|----------|---------------------------------|------------------|---------------------------|------|
| hgraph | 支持 | 支持 | 公布能力时支持 | 遵循 `calculate_precise_distance`。 |
| ivf | 支持 | 支持 | 公布能力时支持 | 可用性取决于是否保留精确存储。 |
| brute_force | 支持 | 支持 | 公布能力时支持 | 仅稠密单向量索引。 |
| pyramid | 支持 | 支持 | 支持 | |
| lazy_hgraph | 支持 | 不支持 | 取决于当前生效的内部索引 | DatasetPtr 批量调用会委托给当前 BruteForce/HGraph；可用性可能在 phase 切换后改变。 |
| sindi | 不支持 | 支持 | 支持 | 仅稀疏向量。 |
| hnsw | 支持 | 不支持 | 不支持 | 稠密批量接口仅支持 `topk == -1`。 |
| diskann | 支持 | 不支持 | 不支持 | 稠密批量接口仅支持 `topk == -1`。 |

对于未实现某重载的索引，调用会返回 `UNSUPPORTED_INDEX_OPERATION` 错误。

## 注意事项

- 稠密重载中，查询向量的维度必须与索引维度一致。
- 批量重载存在默认实现：循环调用单 ID 接口；部分索引会重写以做批量优化。
- 与 VSAG 其他只读接口一样，这些方法可以与 `KnnSearch` 等只读操作并发调用。
