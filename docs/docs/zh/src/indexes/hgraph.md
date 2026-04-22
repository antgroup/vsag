# HGraph

HGraph 是 VSAG 的旗舰 **图索引**。它构建的是与 HNSW 思路类似的多层近邻图，但在此基础上
提供了更丰富的量化方案、统一的构建参数 schema（`index_param`），并原生支持精排（reorder）、
增量更新、删除、以及基于 ELP 的运行时自动调优。

对于大多数稠密向量场景（文本 / 图像 / 多模态 embedding，维度 64–4096，规模从数千到数亿），
HGraph 都是推荐的默认索引。

- 源码：`src/algorithm/hgraph.{h,cpp}`
- 示例：[`examples/cpp/103_index_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/103_index_hgraph.cpp)

## 工作原理

1. **构图。** 向量被组织成层级近邻图：上层作为导航入口，底层连接每个数据点到在
   `max_degree` 预算内的最近邻。构图算法可以是 NSW 风格插入（`graph_type: "nsw"`，默认）
   或 ODescent（`graph_type: "odescent"`）。
2. **量化。** 底层存储使用可配置的量化器进行压缩（`base_quantization_type` —
   `fp32`、`fp16`、`bf16`、`sq8`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq`）。
   可选地再保留一份高精度副本（`use_reorder: true` 搭配 `precise_quantization_type`），
   用于对粗排结果进行重打分。
3. **搜索。** 自顶向下在图上做贪心 beam search，扩展候选集到 `ef_search` 个节点；如启用精排，
   最终结果会在高精度表示上重新打分。

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
        "ef_construction": 400
    }
})";
auto index = vsag::Factory::CreateIndex("hgraph", params).value();

// 构建索引。
auto base = vsag::Dataset::Make();
base->NumElements(n)->Dim(128)->Ids(ids)->Float32Vectors(data)->Owner(false);
index->Build(base);

// 执行检索。
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(128)->Float32Vectors(q)->Owner(false);
auto result = index->KnnSearch(
    query, /*topk=*/10, R"({"hgraph": {"ef_search": 100}})").value();
```

## 构建参数

构建参数放在 `index_param` 下。下表列出最常用的配置项；完整列表请见
[索引参数](../resources/index_parameters.md) 以及仓库中的 `docs/hgraph.md`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_quantization_type` | string | —（必填） | `fp32`、`fp16`、`bf16`、`sq8`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq` |
| `max_degree` | int | `64` | 图节点最大出度 |
| `ef_construction` | int | `400` | 构建阶段的候选集大小（越大召回越高，构建越慢） |
| `graph_type` | string | `"nsw"` | 构图算法：`nsw` 或 `odescent` |
| `use_reorder` | bool | `false` | 是否额外保留一份高精度副本用于精排 |
| `precise_quantization_type` | string | `"fp32"` | 精排使用的量化类型（仅在 `use_reorder: true` 时生效） |
| `base_pq_dim` | int | `128` | PQ 子空间数（`pq` / `pqfs` 时必填） |
| `build_thread_count` | int | `100` | 构建阶段并发线程数 |
| `support_duplicate` | bool | `false` | 是否在插入时做重复 ID 检测 |
| `support_remove` | bool | `false` | 是否支持 `Remove()` |
| `store_raw_vector` | bool | `false` | 除量化副本外再保留原始向量（`cosine` 场景有用） |
| `use_elp_optimizer` | bool | `false` | 构建完成后自动调优检索参数 |
| `base_io_type` / `precise_io_type` | string | `"block_memory_io"` | 存储后端（`memory_io`、`block_memory_io`、`buffer_io`、`async_io`、`mmap_io`） |
| `base_file_path` / `precise_file_path` | string | `"./default_file_path"` | 当使用磁盘后端时的文件路径 |
| `hgraph_init_capacity` | int | `100` | 初始容量提示（不会限制最终规模） |

## 检索参数

检索参数放在 `hgraph` 子对象下：

| 参数 | 类型 | 说明 |
|------|------|------|
| `ef_search` | int | 搜索前沿候选集的大小，越大召回越高、查询越慢 |

```cpp
auto result = index->KnnSearch(
    query, topk, R"({"hgraph": {"ef_search": 200}})").value();
```

## 何时选择 HGraph

- 维度大约在 64–4096 的稠密 float 向量。
- 对延迟敏感且要求高召回的场景。
- 需要增量插入（可选通过 `support_remove` 打开删除）的混合负载。
- 内存受限环境，可用 `sq8` / `sq4_uniform` / `pq` 压缩，再配合 `use_reorder` 弥补召回。

如果你的业务偏向粗粒度分桶（每次查询只扫部分桶）或严重受 SSD I/O 制约，建议先对比
[IVF](ivf.md) 再决定是否选择 HGraph。

## 相关文档

- [创建索引](../guide/create_index.md)
- [图索引增强](../advanced/enhance_graph.md)
- [优化器](../advanced/optimizer.md)
- [序列化格式](../advanced/serialization.md)
