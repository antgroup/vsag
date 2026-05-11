# BruteForce

BruteForce 是 VSAG 提供的**精确扁平索引**。查询时直接对语料中的每条向量计算距离并返回真实的
top-k —— 没有图遍历、没有倒排表、不做近似。它的主要用途是为 HGraph、IVF 等近似索引提供
**ground truth 基准**，也适合用于小规模语料或对召回率有严格要求的生产场景。

- 源码：`src/algorithm/brute_force.{h,cpp}`
- 示例：[`examples/cpp/105_index_brute_force.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/105_index_brute_force.cpp)

## 工作原理

1. **Build。** 向量按照 `base_quantization_type`（默认 `fp32`，即原始精度）编码后保存到一个
   扁平数据单元中。对于不压缩的量化器，不需要训练；当使用 PQ/SQ_uniform 等需要训练的量化器
   时，`Build` 会先跑一遍训练。
2. **Add。** 新向量直接追加到扁平存储中，没有再平衡或重建成本。
3. **Search。** 针对每条查询，按照配置的 `metric_type`（`l2`、`ip` 或 `cosine`）逐条计算
   距离，再用 top-k 小顶堆得到最近邻 id。距离计算使用 SIMD 内核，并支持**单查询内并行**：
   通过 `parallelism` 搜索参数可以把同一条查询的扫描拆分到多个线程上（实现见
   `BruteForce::SearchWithRequest`，`src/algorithm/brute_force.cpp`）。

由于索引保留了每一条向量（除非选择了有损量化器），当 `base_quantization_type = fp32` 时
结果是**完全精确的**，因此 `eval_performance` 工具默认用 BruteForce 作为生成 ground truth 的
参考索引。

## 快速开始

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128
})";
auto index = vsag::Factory::CreateIndex("brute_force", params).value();

// 构建。
auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Build(base);

// 搜索 —— 没有索引特有的旋钮，传空 JSON 即可（也可以设置 `parallelism`）。
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(query, /*topk=*/10, "{}").value();
```

完整可运行示例见
[`examples/cpp/105_index_brute_force.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/105_index_brute_force.cpp)。

## 构建参数

最简配置只需要三个顶层字段（`dtype`、`metric_type`、`dim`）。大多数场景下不需要
`index_param`，这也是
[示例 105](https://github.com/antgroup/vsag/blob/main/examples/cpp/105_index_brute_force.cpp)
所采用的形式。进阶用法可通过 `index_param` 启用量化或存储相关的开关：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_quantization_type` | string | `"fp32"` | `fp32`、`fp16`、`bf16`、`sq8`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq` |
| `use_attribute_filter` | bool | `false` | 启用属性过滤（参见 [属性过滤](../advanced/attribute_filter.md)） |

> **关于 `store_raw_vector` 的说明。** `store_raw_vector` 字段会被共用的
> `InnerIndexParameter` 解析，但 BruteForce **不会**根据它决定是否启用
> `GetRawVectorByIds`。在 BruteForce 上，原始向量读取能力仅在
> `base_quantization_type = fp32`、并且度量不是 `cosine` 或量化器配置了
> 持有向量范数（`hold_molds`）时开启。在 BruteForce 上设置
> `store_raw_vector: true` 目前不会改变任何能力标志 —— 如果需要量化索引同时
> 支持 `GetRawVectorByIds`，请使用 HGraph 或 IVF。

下面是一个使用 `sq8` 量化以节省内存、同时保持线性扫描语义的示例：

```json
{
    "dtype": "float32",
    "metric_type": "ip",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8"
    }
}
```

当 `base_quantization_type` 选择了需要训练的量化器（`sq8`、`sq8_uniform`、`sq4_uniform`、
`pq`、`pqfs`、`rabitq`）时，`Build` 会先用传入的数据集训练量化器，此时召回率不再保证
100%。只有 `fp32`、`fp16`、`bf16` 不需要训练，能保持精确距离（仅受浮点数值精度影响）。

## 搜索参数

BruteForce 没有任何索引特有的搜索旋钮（不存在 `ef`、`nprobe` 之类的参数），但通用的
`IndexSearchParameter` 字段都生效：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `parallelism` | int | `1` | 把单条查询的线性扫描拆分到索引内部线程池中的若干线程上。该值越大，大语料下的单查询延迟越低，代价是占用更多 CPU 核。`<= 0` 的取值会被钳制到 `1`。 |

```cpp
// 默认单线程扫描。
auto r1 = index->KnnSearch(query, topk, "{}").value();

// 用 8 个线程并行扫描同一条查询。
auto r2 = index->KnnSearch(query, topk, R"({"parallelism": 8})").value();
```

范围查询参见 [范围搜索](../advanced/range_search.md)。BruteForce 的 `RangeSearch` 不论
`parallelism` 设置如何，始终单线程执行。

## 索引能力

BruteForce 声明的能力标志如下（参见 `BruteForce::InitFeatures`，
`src/algorithm/brute_force.cpp`）：

| 能力                                | 说明 |
|-------------------------------------|------|
| `SUPPORT_BUILD` / `SUPPORT_ADD_AFTER_BUILD` / `SUPPORT_ADD_CONCURRENT` | 支持一次构建、增量追加，以及并发插入。 |
| `SUPPORT_ADD_FROM_EMPTY` | 仅在非训练型量化器（`fp32`、`fp16`、`bf16`）下可用。 |
| `SUPPORT_KNN_SEARCH` / `SUPPORT_KNN_SEARCH_WITH_ID_FILTER` / `SUPPORT_SEARCH_CONCURRENT` | 标准 top-k 查询、id 列表过滤，以及并发搜索。 |
| `SUPPORT_RANGE_SEARCH` / `SUPPORT_RANGE_SEARCH_WITH_ID_FILTER` | 仅在非训练型量化器（`fp32`、`fp16`、`bf16`）下可用。 |
| `SUPPORT_DELETE_BY_ID` / `SUPPORT_DELETE_CONCURRENT` | 支持按 id 删除，且并发安全。 |
| `SUPPORT_CAL_DISTANCE_BY_ID` | 与已存储向量计算距离（仅非训练型量化器）。 |
| `SUPPORT_GET_RAW_VECTOR_BY_IDS` | 仅当 `base_quantization_type = fp32`，且度量不是 `cosine` 或底层量化器持有向量范数（`hold_molds`）时才声明。量化的 BruteForce 索引**不会**声明该能力。 |
| `SUPPORT_CHECK_ID_EXIST` / `SUPPORT_CLONE` / `SUPPORT_ESTIMATE_MEMORY` / `SUPPORT_GET_MEMORY_USAGE` | 标准的内省与生命周期接口。 |
| `SUPPORT_SERIALIZE_BINARY_SET` / `SUPPORT_SERIALIZE_FILE` / `SUPPORT_SERIALIZE_WRITE_FUNC` | 完整的保存能力。 |
| `SUPPORT_DESERIALIZE_BINARY_SET` / `SUPPORT_DESERIALIZE_FILE` / `SUPPORT_DESERIALIZE_READER_SET` | 完整的加载能力。（没有对应的 `DESERIALIZE_WRITE_FUNC`，读路径使用 `READER_SET` 形式。） |
| `NEED_TRAIN` | 当 `base_quantization_type` 是 `sq8`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq` 之一时声明。 |

BruteForce **不支持** 的能力包括：`SUPPORT_UPDATE_VECTOR_CONCURRENT`、
`SUPPORT_UPDATE_ID_CONCURRENT`、`SUPPORT_EXPORT_MODEL`。

## 适用场景

- **召回基准。** 为近似索引计算 ground truth（`eval_performance` 工具就是这么做的）。
- **小规模语料。** 几百到几十万条向量，全量扫描成本可接受，且无需做参数调优。
- **强召回需求。** 完全不能容忍近似误差的业务。
- **小规模量化实验。** 在同一条线性扫描流水线上对比不同 `base_quantization_type` 的效果，
  排除图结构 / 倒排表带来的干扰。

如果数据规模更大，请优先选择 [HGraph](hgraph.md)（延迟敏感、高召回）或
[IVF](ivf.md)（吞吐量优先、内存友好）。

## 参考

- [创建索引](../guide/create_index.md)
- [k-近邻搜索](../guide/knn_search.md)
- [范围搜索](../advanced/range_search.md)
- [属性过滤（混合搜索）](../advanced/attribute_filter.md)
- [性能评估工具](../resources/eval.md)
