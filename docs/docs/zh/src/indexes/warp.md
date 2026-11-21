# WARP

WARP 是 VSAG 的**多向量精确检索**索引。它把每篇文档存成一组 token 级向量，而不是单个
embedding，对应 ColBERT 这类 late-interaction 模型的表示方式；相关性用 **MaxSim** 打分
（对每个查询 token 取最相似的文档 token，再对查询 token 求和）。

WARP 没有独立的算法实现：它就是运行在多向量模式下的 [BruteForce](brute_force.md)。
`Factory::CreateIndex("warp", ...)` 会创建 `IndexImpl<BruteForce>`，并写入一个隐藏的内部
标记（`_warp_mode`）把底层 datacell 切换成多向量版本。由于是全量扫描，在配置的量化器下
结果是精确的。

- Source: `src/algorithm/bruteforce/bruteforce.{h,cpp}`、`src/datacell/multi_vector_datacell.{h,inl}`、`src/quantization/multi_vector_computer.cpp`
- Example: [`examples/cpp/110_index_warp.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/110_index_warp.cpp)

## 工作原理

1. **构建。** 文档被追加进一个扁平的多向量存储，每条记录是 `uint32` 的 token 数，紧跟
   相应数量的编码后 token 向量。没有图结构、没有聚类；需要训练的量化器（`fp16`、
   `sq8_uniform`、`int8`）在第一批插入时对所有 token 跑一次训练，`fp32` / `bf16` 不训练。
2. **追加。** 新文档直接追加，没有重平衡成本；重复的 label 会通过 `failed_ids` 返回。
3. **检索。** 每篇文档都会被解码并与查询多向量比较：对每个查询 token 选出最近的文档
   token，再把各 token 的距离累加。top-k 堆按序填满，可用 `parallelism` 检索参数把单次
   查询切分到多个线程。

MaxSim 需要一个“距离”，所以单 token 项在 `ip` 下是 `1 - <q, d>`，在 `l2` 下是
`||q - d||^2`。因此 `ip` 下内积越大越接近，而两种 metric 返回的都是距离（越小越好）。

## 快速开始

```cpp
#include <vsag/vsag.h>

std::string build_params = R"({
    "dtype": "float32",
    "metric_type": "ip",
    "dim": 128
})";

auto index = vsag::Factory::CreateIndex("warp", build_params).value();

// 每篇文档都是维度为 `dim` 的一组 token 向量。
std::vector<vsag::MultiVector> multi_vectors(num_docs);
for (int64_t i = 0; i < num_docs; ++i) {
    multi_vectors[i].len_ = token_count_of_doc_i;          // token 数
    multi_vectors[i].vectors_ = token_vectors_of_doc_i;    // len_ * dim 个 float
}

auto base = vsag::Dataset::Make();
base->NumElements(num_docs)
    ->Dim(dim)
    ->Ids(ids)
    ->MultiVectors(multi_vectors.data())
    ->MultiVectorDim(dim)
    ->Owner(true);
index->Build(base);

auto query = vsag::Dataset::Make();
query->NumElements(1)
    ->Dim(dim)
    ->MultiVectors(&query_multi_vector)   // 一个 vsag::MultiVector
    ->MultiVectorDim(dim)
    ->Owner(false);

auto result = index->KnnSearch(query, topk, R"({"parallelism": 4})").value();
```

检索 JSON 很宽松，但只有 `brute_force` 和 `hgraph` 这两个子对象名会被识别；由于 WARP 复用
BruteForce 的检索参数解析器，`{"warp": {...}}` 会被当作根对象处理，写在 `warp` 里的键全部被忽略。
要让检索参数生效，请直接写在根层（例如 `{"parallelism": 4}`）。

## 构建参数

顶层字段是 `dim`、`dtype`、`metric_type`：

| 参数 | 取值 | 说明 |
|------|------|------|
| `dim` | 正整数 | token 维度，必填 |
| `dtype` | `float32` | WARP 拒绝 `int8`（`WARP not support int8 datatype`）；其他取值未被验证 |
| `metric_type` | `ip` / `l2` | `cosine` 会在检索阶段被 `MultiVectorComputer` 拒绝 |

`index_param` 只接受下面这些键，其他键会抛 `invalid config param: <key>`：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_quantization_type` | string | `"fp32"` | 存储 token 向量的量化器：`fp32`、`fp16`、`bf16`、`sq8_uniform`、`int8`。`fp16`、`sq8_uniform`、`int8` 需要显式训练。 |
| `base_io_type` | string | `"block_memory_io"` | token 存储的 IO 后端：`block_memory_io`、`memory_io`、`buffer_io`、`async_io`、`uring_io`、`mmap_io`；加载序列化索引时使用 `reader_io`。 |
| `base_file_path` | string | `"./default_file_path"` | 磁盘型 `base_io_type` 的文件路径。两个索引共用同一路径并同时写入会损坏存储，请为每次构建指定不同路径。 |
| `resize_increase_count_bit` | int | `10` | 存储扩容时增量批次的 `log2`，取值需在 `[1, 31]` 内。 |

WARP 比 [BruteForce](brute_force.md) 更窄。以下 BruteForce 构建键**不被** WARP 接受，会抛
`invalid config param`：`precise_quantization_type`、`precise_io_type`、`precise_file_path`、
`base_pq_dim`、`thread_count`、`store_raw_vector`、`use_attribute_filter`、`use_residual`。
WARP 只构建一份多向量底座存储，没有精排编码，也没有 extra info 存储。

## 检索参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `parallelism` | int | `1` | 把单次查询的扫描切分到该数量的线程；`<= 0` 按 `1` 处理。 |
| `use_extra_info_filter` | bool | `false` | 扫描时按 extra info 谓词过滤候选 id。只有创建索引时给了顶层 `extra_info_size` 才有意义。 |
| `threshold` | float | —（不过滤） | 根层键，由 `ParseSearchThreshold` 读取；距离高于阈值的结果会从 top-k 中剔除，必须是有限数值。 |

`timeout_ms`、`factor`、`enable_reorder` 会被共享的检索解析器接受，但 WARP **不使用**它们：
全量扫描没有时间上限，`factor` 不会改变精确检索的候选集，也不存在重排阶段。

## 运维说明

- **删除。** 只支持 `MARK_REMOVE`；`FORCE_REMOVE` 会报错
  （`multi-vector mode only supports MARK_REMOVE`）。标记不存在或已删除的 id 是空操作。
- **不支持属性过滤。** WARP 的 `use_attribute_filter` 恒为 false，因此带属性表达式的
  `SearchWithRequest` 会报 `attribute filter is not available`。请改用基于 id 的过滤
  （见 [带过滤的搜索](../advanced/filtered_search.md)）。
- **不支持向量更新。** `UpdateVector` 需要单条稠密 FP32 向量，多向量索引下会被拒绝；请
  重建索引或先删后加。
- **内存估算。** `EstimateMemory` 假定每篇文档平均 10 个 token，并按每个 token
  `dim * sizeof(float)` 计算，因此对真实多向量语料只是近似值。

## 何时选择 WARP

当文档是 token 集合、并且 MaxSim 质量比延迟更重要时使用 WARP：它在中等规模语料上给出
精确的 late-interaction 分数。若需要图加速的多向量检索和两阶段粗排/精排流程，请使用
[SIMQ](simq.md)。

## 相关文档

- [SIMQ](simq.md) — 近似多向量检索，带 MaxSim 精排
- [BruteForce](brute_force.md) — WARP 所基于的单向量精确基线
- [带过滤的搜索](../advanced/filtered_search.md) — id 与函数过滤
- [索引参数](../resources/index_parameters.md) — 跨索引参数总览