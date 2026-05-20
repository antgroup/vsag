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
   `fp32`、`fp16`、`bf16`、`sq8`、`sq4`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq`、`tq`）。
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
| `base_quantization_type` | string | —（必填） | `fp32`、`fp16`、`bf16`、`sq8`、`sq4`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、`rabitq`、`tq` —— 各量化器细节见[量化章节](../quantization/README.md) |
| `max_degree` | int | `64` | 图节点最大出度 |
| `ef_construction` | int | `400` | 构建阶段的候选集大小（越大召回越高，构建越慢） |
| `graph_type` | string | `"nsw"` | 构图算法：`nsw` 或 `odescent` |
| `use_reorder` | bool | `false` | 是否额外保留一份高精度副本用于精排 |
| `precise_quantization_type` | string | `"fp32"` | 精排使用的量化类型（仅在 `use_reorder: true` 时生效） |
| `base_pq_dim` | int | `1` | PQ 子空间数（`pq` / `pqfs` 时必填） |
| `build_thread_count` | int | `100` | 构建阶段并发线程数 |
| `support_duplicate` | bool | `false` | 是否在插入时做重复 ID 检测 |
| `duplicate_distance_threshold` | float | `0.0` | 重复判定距离阈值。大于 `0` 时按最近候选的距离判重；等于 `0` 时退化为当前编码 `memcmp` 判重 |
| `support_remove` | bool | `false` | 是否支持 `Remove()` |
| `store_raw_vector` | bool | `false` | 除量化副本外再保留原始向量（`cosine` 场景有用） |
| `use_elp_optimizer` | bool | `false` | 构建完成后自动调优检索参数 |
| `base_io_type` / `precise_io_type` | string | `"block_memory_io"` | 存储后端（`memory_io`、`block_memory_io`、`buffer_io`、`async_io`、`mmap_io`） |
| `base_file_path` / `precise_file_path` | string | — | 磁盘后端时的文件路径（使用 `mmap_io` / `async_io` / `buffer_io` 时必填） |
| `hgraph_init_capacity` | int | `100` | 初始容量提示（不会限制最终规模） |

## 支持的输入数据类型

顶层构建配置中的 `dtype` 字段决定 `Dataset` 如何解释原始向量字节。HGraph 支持四种输入类型，
`dtype` 值、对应的 `Dataset` setter 与演示示例见下表。

| `dtype`     | 元素类型 | `Dataset` setter         | 示例                                                                                                  |
|-------------|----------|--------------------------|-------------------------------------------------------------------------------------------------------|
| `float32`   | `float`  | `Float32Vectors`         | [`103_index_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/103_index_hgraph.cpp) |
| `int8`      | `int8_t` | `Int8Vectors`            | [`316_index_int8_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/316_index_int8_hgraph.cpp) |
| `float16`   | `uint16_t`（按 IEEE 754 binary16 位模式打包） | `Float16Vectors` | [`321_index_fp16_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/321_index_fp16_hgraph.cpp) |
| `bfloat16`  | `uint16_t`（按 Brain Float 位模式打包） | `Float16Vectors`（与 FP16 共用） | 在 `321_index_fp16_hgraph.cpp` 基础上按下文调整                                                       |

`dim` 始终表示逻辑维度（元素数量），与字节长度无关，因此四种数据类型下 `dim` 取值相同。

### `int8` 输入

量化好的 `int8` 向量直接通过 `Int8Vectors` 传入：

```cpp
std::vector<int8_t> data(num_vectors * dim);  // 填入 int8 元素
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids)
    ->Int8Vectors(data.data())->Owner(false);
```

对应构建配置（注意 `dtype: "int8"`）：

```json
{
    "dtype": "int8",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pq",
        "max_degree": 26,
        "ef_construction": 100,
        "alpha": 1.2
    }
}
```

查询时同样使用 `Int8Vectors` 和相同的 `dtype`。可运行示例：
[`316_index_int8_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/316_index_int8_hgraph.cpp)。

### `float16` / `bfloat16` 输入

FP16 与 BF16 都通过 `Float16Vectors` 传入，参数类型为 `const uint16_t*`，指向各元素的 16 位
存储。从 `float` 到 16 位格式的转换由调用方负责。VSAG 源码树内提供了便捷辅助函数
（`vsag::generic::FloatToFP16` 位于
[`src/simd/fp16_simd.h`](https://github.com/antgroup/vsag/blob/main/src/simd/fp16_simd.h)，
`vsag::generic::FloatToBF16` 位于
[`src/simd/bf16_simd.h`](https://github.com/antgroup/vsag/blob/main/src/simd/bf16_simd.h)），
但它们是**内部头文件**，并未通过 `include/vsag/` 对外安装。链接已安装版 VSAG 库的应用需要自行
完成转换（例如复制这段小工具函数、使用 `_cvtss_sh` / F16C 内置指令，或调用任意 FP16 库）。下面
的示例代码为了简洁直接使用了源码树内的辅助函数：

```cpp
// 下面的 fp16/bf16 辅助函数位于 src/simd/，并未随 VSAG 一并安装。
// 链接已安装版 VSAG 时，请替换为自行实现的 float -> uint16_t 转换。
#include "simd/fp16_simd.h"  // FloatToFP16（BF16 场景改为 simd/bf16_simd.h / FloatToBF16）

std::vector<uint16_t> data(num_vectors * dim);
for (size_t i = 0; i < data.size(); ++i) {
    data[i] = vsag::generic::FloatToFP16(some_float_source());
}
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)->Dim(dim)->Ids(ids)
    ->Float16Vectors(data.data())->Owner(false);
```

构建配置：

```json
{
    "dtype": "float16",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "pq",
        "max_degree": 26,
        "ef_construction": 100,
        "alpha": 1.2
    }
}
```

切换到 BF16 时，将 `dtype` 改为 `"bfloat16"`、把 `FloatToFP16` 替换为 `FloatToBF16` 即可；
`Float16Vectors` setter 与构建/检索流程不变。可运行 FP16 示例：
[`321_index_fp16_hgraph.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/321_index_fp16_hgraph.cpp)。

> **注意。** `321_index_fp16_hgraph.cpp` 文件头注释提到 `BFloat16Vectors()`，但该 setter 并不
> 存在 —— FP16 与 BF16 都通过 `Float16Vectors` 传入。无论 `dtype` 是 `"float16"` 还是
> `"bfloat16"`，都使用同一个 setter。

### 输入类型选择建议

- 对精度要求最高、且内存预算充裕时，选 `float32`（默认）。
- 想把输入存储减半，选 `float16` / `bfloat16`。FP16 指数范围更小，BF16 尾数位更少但指数范围
  与 FP32 一致，对 embedding 类向量通常更友好。
- 数据本身已是整数量化结果（来自上游量化器或 int8 输出的模型）时，选 `int8`。此时通常仍配合
  `pq` / `sq8` 之类的索引内量化器使用。

`dtype` 仅约束**输入**表示；索引内的实际存储仍由 `base_quantization_type`（以及
`use_reorder: true` 下的 `precise_quantization_type`）决定，因此
`dtype: "float16"` + `base_quantization_type: "sq8"` 这样的组合是允许的。

## 检索参数

检索参数放在 `hgraph` 子对象下：

| 参数 | 类型 | 说明 |
|------|------|------|
| `ef_search` | int | 搜索前沿候选集的大小，越大召回越高、查询越慢 |
| `enable_reorder` | bool | 默认值为 `true`。当索引构建时启用了 reorder，也可以在单次请求里设为 `false` 跳过最终精排；这也会一并关闭 RaBitQ 的 one-bit reorder 路径。 |

```cpp
auto result = index->KnnSearch(
    query, topk, R"({"hgraph": {"ef_search": 200}})").value();
```

```cpp
auto fast_result = index->KnnSearch(
    query, topk,
    R"({"hgraph": {"ef_search": 200, "enable_reorder": false}})").value();
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
