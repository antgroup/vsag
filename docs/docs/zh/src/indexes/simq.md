# SIMQ

SIMQ 是 VSAG 面向 **多向量（multi-vector）** 检索的索引——适用于每篇文档
由一组 token 级向量（而非单个 embedding）表示的数据场景。这种模式常见于
ColBERT 等 late-interaction 模型，其中文档由每个 token 对应一个向量表示，
相关性通过 **MaxSim**（各查询 token 的最大相似度之和）计算。


- 源码：`src/algorithm/simq/`

## 工作原理

1. **Token 向量动态聚类。** 构建阶段，所有文档的全部 token 向量被抽取到一个
   扁平池中，使用基于 HGraph 的动态聚类算法进行聚类。初始聚类中心按
   `init_cluster_ratio` 控制的比例采样；超过 `max_cluster_size` 的簇会被增量切分。
2. **代表性图用于粗排。** 在簇中心上构建一个代表性 HGraph。检索时，每个查询
   token 在该图上搜索最近的若干个簇（由 `coarse_k` 控制），跨查询 token
   累加簇得分，得到候选集。
3. **精确 MaxSim 精排。** 对得分最高的 `rerank_k` 个候选文档，从磁盘（或内存）
   读回原始 token 向量，计算查询 token 与文档 token 之间的精确 MaxSim 相似度。

聚类粗排与精确精排的两阶段结合，
为多向量检索提供了可调的召回率/延迟权衡。

## 快速开始

```cpp
#include <vsag/vsag.h>

std::string build_params = R"({
    "dtype": "float32",
    "metric_type": "ip",
    "dim": 256,
    "index_param": {
        "base_io_type": "async_io",
        "base_file_path": "/path/to/simq_base_codes.bin",
        "init_cluster_ratio": 0.1,
        "max_cluster_size": 160,
        "split_start_idx": 80,
        "random_seed": 42,
        "coarse_k": 50,
        "rerank_k": 1000
    }
})";
auto index = vsag::Factory::CreateIndex("simq", build_params).value();

// 使用 MultiVector 构建数据集。
// 每篇文档包含可变数量的 token 向量，每个向量维度为 dim。
std::vector<vsag::MultiVector> base_mvs(num_docs);
std::vector<int64_t> ids(num_docs);
for (int64_t i = 0; i < num_docs; ++i) {
    base_mvs[i].len_ = doc_token_counts[i];             // 文档 i 的 token 数量
    base_mvs[i].vectors_ = doc_token_vectors[i];        // 扁平数组：len_ * dim 个 float
    ids[i] = i;
}
auto base = vsag::Dataset::Make();
base->NumElements(num_docs)
    ->Dim(dim)
    ->Ids(ids.data())
    ->MultiVectors(base_mvs.data())
    ->MultiVectorDim(dim)
    ->Owner(false);
index->Build(base);

// 使用多向量查询进行检索。
vsag::MultiVector query_mv;
query_mv.len_ = query_token_count;
query_mv.vectors_ = query_token_vectors;
auto query = vsag::Dataset::Make();
query->NumElements(1)
    ->Dim(dim)
    ->MultiVectors(&query_mv)
    ->MultiVectorDim(dim)
    ->Owner(false);

std::string search_params = R"({
    "simq": {
        "coarse_k": 600,
        "rerank_k": 5000
    }
})";
auto result = index->KnnSearch(query, /*topk=*/100, search_params).value();

// 读取结果。
const int64_t* result_ids = result->GetIds();
const float* result_dists = result->GetDistances();
int64_t result_count = result->GetDim();
for (int64_t i = 0; i < result_count; ++i) {
    int64_t id = result_ids[i];
    float dist = result_dists[i];
}
```

## 构建参数

SIMQ 专属构建参数放在 `index_param` 下。`dim`、`dtype`、`metric_type`
是顶层字段。`dtype` **必须** 为 `"float32"`，`metric_type` **必须** 为 `"ip"`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dim` | int | —（必填） | 每个 token 向量的维度 |
| `base_io_type` | string | `"async_io"` | 精排阶段使用的原始多向量数据存储后端 |
| `base_file_path` | string | `"./default_file_path"` | 磁盘 IO 类型使用的文件路径 |
| `init_cluster_ratio` | float | `0.2` | 初始聚类中心的 token 向量采样比例 |
| `max_cluster_size` | int | `64` | 单个簇允许的最大 token 向量数 |
| `split_start_idx` | int | `32` | 簇切分时新簇的起始位置 |
| `random_seed` | int | `42` | 聚类打乱的随机种子 |
| `coarse_k` | int | `8` | 构建时每个查询 token 搜索的最近簇数量 |
| `rerank_k` | int | `100` | 构建时进入精排的候选文档数量上限 |

- **`dim`** — 所有文档和查询中的所有 token 共享同一维度
- **`base_io_type`** — 可选值：`async_io`、`memory_io`、
  `block_memory_io`、`buffer_io`、`mmap_io`、`reader_io`
- **`base_file_path`** — 默认值为占位符，使用磁盘类型（`async_io`、
  `buffer_io`、`mmap_io`）时需提供真实路径
- **`init_cluster_ratio`** — 取值范围 `(0, 1]`。值越小簇越少越大，
  值越大簇越多越细
- **`max_cluster_size`** — 必须 > 1
- **`split_start_idx`** — 通常设为 `max_cluster_size` 的一半，
  取值范围 `(1, max_cluster_size)`
- **`coarse_k`**、**`rerank_k`** — 必须 > 0

> **聚类参数的选择。** `init_cluster_ratio` 与 `max_cluster_size` 共同控制
> 簇的数量与大小。较小的 `init_cluster_ratio` 搭配较大的 `max_cluster_size`
> 会产生更少的簇，粗排更快但召回降低。建议以 `init_cluster_ratio = 0.1`–`0.2`、
> `max_cluster_size = 2 × split_start_idx` 为起点，再通过检索参数调优。

## 检索参数

检索参数放在 `simq` 子对象下：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `coarse_k` | int | *（构建时默认值）* | 每个查询 token 搜索的最近簇数量 |
| `rerank_k` | int | *（构建时默认值）* | 进入精排的候选文档数量上限 |

- **`coarse_k`** — 覆盖构建时的值。值越大候选范围越广，
  召回越高但延迟也越大
- **`rerank_k`** — 覆盖构建时的值。值越大召回越高，
  但磁盘读取和计算开销也越大
- 不设置时使用构建时的默认值。显式设置时两个值都必须 > 0

```cpp
auto result = index->KnnSearch(
    query, topk,
    R"({"simq": {"coarse_k": 600, "rerank_k": 5000}})").value();
```

## 何时选择 SIMQ

- **Late-interaction 检索**：使用 ColBERT 等模型，每篇文档是一组 token 向量，
  相关性通过 MaxSim 计算。
- **多向量粒度匹配**：单 embedding 丢失过多信息，需要 token 级细粒度匹配。
- **大规模多向量语料**：暴力 MaxSim 检索过慢，需要粗排 + 精排的两阶段管线
  来平衡召回与延迟。

SIMQ 仅接受 `float32` 多向量数据，仅支持内积相似度。
**不支持** 单稠密向量或稀疏向量（请使用 HGraph 或 SINDI）。

## 实践建议

- **调整 `coarse_k` 与 `rerank_k`。** 增大 `coarse_k` 扩大簇级候选范围；
  增大 `rerank_k` 让更多文档进入精确打分。实践中 `rerank_k` 对召回的影响
  更大，但每个额外候选都需要一次磁盘读取和完整的 MaxSim 计算，延迟也会增加。
- **IO 类型选择。** 语料规模超出内存时使用 `async_io`；多向量数据可以放入
  内存时，使用 `memory_io` 或 `block_memory_io` 获得最低的精排延迟。
- **簇大小配置。** 将 `max_cluster_size` 设为 `split_start_idx` 的约两倍。切分位置
  决定了簇溢出时 token 向量的分配方式，居中设置可使两半保持平衡。

## MultiVector 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `len_` | `uint32_t` | 当前文档或查询包含的 token 向量数量 |
| `vectors_` | `float*` | `len_ * dim` 个 float 的连续数组 |

## 相关文档

- [创建索引](../guide/create_index.md)
- [索引参数](../resources/index_parameters.md)
- [k-近邻搜索](../guide/knn_search.md)
