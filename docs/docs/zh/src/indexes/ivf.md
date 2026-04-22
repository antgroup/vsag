# IVF

IVF（Inverted File，倒排索引）是 VSAG 的 **分桶式** 索引。它在构建时将语料聚类成若干桶，
查询时只扫描与查询距离最近的若干个桶的中心对应的倒排列表，把 O(N) 的线性扫描降为
O(N · `scan_buckets_count` / `buckets_count`)，并通过这两个参数在召回与延迟之间进行权衡。

与图索引相比，IVF 在召回上略有损失，但换来了更低的内存开销、更高的批量吞吐以及更简单的
切片方式——因此在语料非常大（数亿及以上）、内存紧张、或查询可天然并行化的场景中，
IVF 通常是一个更合适的默认选择。

- 源码：`src/algorithm/ivf.{h,cpp}`、`src/algorithm/ivf_parameter.{h,cpp}`
- 示例：[`examples/cpp/106_index_ivf.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/106_index_ivf.cpp)

## 工作原理

1. **聚类。** 在数据集的采样上运行 k-means（或随机采样，`ivf_train_type: "random"`）
   得到 `buckets_count` 个中心（centroid）。
2. **分配。** 每条向量被写入距离最近的中心对应的倒排列表，以配置的粗量化
   （`base_quantization_type`）存储；可选地再保留一份高精度副本（`use_reorder: true`）
   用于精排。
3. **检索。** 查询时先计算查询向量与所有中心的距离，选出最近的 `scan_buckets_count`
   个桶；然后只在这些桶内对向量打分。启用精排时，`factor` 控制从粗排阶段多取多少候选
   再送入精排器重打分。

此外还有一种 **GNO-IMI** 策略（`partition_strategy_type: "gno_imi"`），它把空间按两组
正交中心划分（`first_order_buckets_count` × `second_order_buckets_count`），在超大规模
语料上能得到更精细的分区。

## 快速开始

```cpp
#include <vsag/vsag.h>

std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "buckets_count": 256,
        "base_quantization_type": "sq8",
        "partition_strategy_type": "ivf",
        "ivf_train_type": "kmeans"
    }
})";
auto index = vsag::Factory::CreateIndex("ivf", params).value();

// 构建索引。
auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Build(base);

// 执行检索。
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10,
    R"({"ivf": {"scan_buckets_count": 16}})").value();
```

## 构建参数

构建参数放在 `index_param` 下。完整列表请见 [索引参数](../resources/index_parameters.md)
及仓库中的 `docs/ivf.md`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `partition_strategy_type` | string | `"ivf"` | 分桶策略：`ivf`（单层）或 `gno_imi`（双层正交） |
| `buckets_count` | int | `10` | 倒排列表数量（`ivf` 策略下生效） |
| `first_order_buckets_count` | int | `10` | 第一级桶数（`gno_imi` 策略下生效） |
| `second_order_buckets_count` | int | `10` | 第二级桶数（`gno_imi` 策略下生效） |
| `ivf_train_type` | string | `"kmeans"` | 中心训练方式：`kmeans` 或 `random` |
| `base_quantization_type` | string | `"fp32"` | `fp32`、`fp16`、`bf16`、`sq8`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq` |
| `base_pq_dim` | int | `1` | PQ 子空间数（`pq` / `pqfs` 时必填） |
| `use_reorder` | bool | `false` | 是否保留高精度副本用于精排 |
| `precise_quantization_type` | string | `"fp32"` | 精排量化类型（`use_reorder: true` 时使用） |
| `base_io_type` | string | `"memory_io"` | 粗排向量的存储后端 |
| `precise_io_type` | string | `"block_memory_io"` | 精排向量的存储后端（`memory_io`、`block_memory_io`、`mmap_io`、`buffer_io`、`async_io`、`reader_io`） |
| `precise_file_path` | string | `""` | 当精排 IO 为磁盘后端时的文件路径 |

`buckets_count` 的经验值一般为 `sqrt(N)` ~ `4 * sqrt(N)`，其中 `N` 是语料规模。

## 检索参数

检索参数放在 `ivf` 子对象下：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scan_buckets_count` | int | —（必填） | 每次查询扫描的桶数，须 ≤ `buckets_count` |
| `factor` | float | `2.0` | 启用精排时，粗排阶段会预取 `factor * topk` 个候选再重打分 |
| `parallelism` | int | `1` | 单次查询内扫描桶时使用的线程数 |
| `timeout_ms` | double | `+∞` | 单次查询最长耗时（毫秒），超时会返回当前的部分结果 |

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"ivf": {"scan_buckets_count": 32, "factor": 2.0, "parallelism": 4}})").value();
```

## 何时选择 IVF

- 超大规模语料（数亿及以上），工作集无法完全放入内存。
- 对每秒查询数（QPS）敏感、对单次延迟相对宽松的批量或高吞吐场景。
- 内存紧张的部署，可使用激进的量化方案（`sq8`、`sq4_uniform`、`pq`、`pqfs`）配合
  `use_reorder` 恢复召回。
- 对切片友好的部署：桶天然映射到分片或磁盘块。

对于延迟敏感、要求高召回的稠密 embedding 场景，请优先比较 [HGraph](hgraph.md)。

## 相关文档

- [创建索引](../guide/create_index.md)
- [索引参数](../resources/index_parameters.md)
- [内存-磁盘混合索引](../advanced/hybrid_index.md)
- [序列化格式](../advanced/serialization.md)
