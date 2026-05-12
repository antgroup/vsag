# 量化变换（Quantization Transform）

**变换量化器**（`base_quantization_type: "tq"`）在最终量化器之前串联一个或多个向量变换。
变换会重塑向量分布，让后续量化器能更准确、更紧凑地编码 —— 例如把向量旋转一下，让能量分散
到各个维度（RaBitQ / SQ 受益最大），或者先用 PCA 降维再存储。

> 可运行示例：`examples/cpp/501_quantization_transform.cpp`。

## 为什么需要变换层

纯量化器直接压缩向量。对低比特量化器（如 `sq4`、`sq*_uniform`、`rabitq`），编码精度严重
依赖向量坐标的**分布**：长尾或各向异性的维度会浪费 code bit。变换层可以缓解这个问题：

- **随机旋转**（`rom`、`fht`）让坐标去相关，均匀/标量量化器在每个轴上工作得更好。
- **PCA**（`pca`）在保留主要方差的同时降低维度 —— code 大小按比例缩小。
- **MRLE**（`mrle`）是为 L2/IP 搜索设计的距离可恢复低秩编码。

变换后的输出再喂给一个标准量化器（`fp32`、`sq8`、`sq8_uniform`、`rabitq` ……），由后者
真正存储 code。整条链被称为 **`tq`（Transform Quantizer）**。

## 快速上手

`tq` 目前作为**对外可配置**的量化类型，只有 **HGraph** 真正暴露了它。HGraph 通过外部参数映射把
顶层键 `tq_chain` 和 `rabitq_pca_dim` 写到嵌套的 `base_codes.quantization_params`
（`src/algorithm/hgraph.cpp:370-385`）。IVF、BruteForce、Pyramid、WARP 虽然在内部 JSON 模板中
也会渲染 `tq_chain` 字段，但它们的外部参数映射里**都没有** `tq_chain`（或其它 TQ 参数）。
`CheckAndMappingExternalParam` 遇到未映射的外部键会直接抛 `invalid config param`
（`src/utils/util_functions.cpp:50-53`），因此在这些索引的 `index_param` JSON 中传 `tq_chain`
会在构建时报错。在非 HGraph 索引上启用 TQ 目前需要在代码侧补一条外部映射。

```cpp
std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "tq",
        "tq_chain": "pca, rom, sq8_uniform",
        "rabitq_pca_dim": 64,
        "max_degree": 32,
        "ef_construction": 300,
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
})";

vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
vsag::Engine engine(&resource);
auto index = engine.CreateIndex("hgraph", params).value();
index->Build(base);
auto result = index->KnnSearch(query, topk, search_params).value();
```

上面的例子里，base 向量先从 128 维降到 64 维（`pca`），随后做随机旋转（`rom`），最后用
`sq8_uniform` 量化。开启了 reorder，HGraph 同时保留一份 `fp32` 精确副本，对图搜索返回的
top 候选做精排（`include/vsag/index.h`；存储影响见 [内存管理](memory.md)）。

## `tq_chain` 语法

`tq_chain` 是一个**以逗号分隔的字符串**：一个或多个变换名，最后跟一个**唯一的**量化器名。
token 两侧的空白会被自动 trim
（`src/quantization/transform_quantization/transform_quantizer_parameter.cpp:53-74`）。

```
"<变换1>, <变换2>, ..., <量化器>"
```

示例：

| 链 | 作用 |
|---|---|
| `"rom, fp32"` | 随机旋转后以 fp32 存储（多用于基线/sanity）。 |
| `"fht, sq8_uniform"` | 快速 Hadamard 旋转 + 8 位均匀标量量化。 |
| `"pca, rom, sq8_uniform"` | 先 PCA 降维，再随机旋转，再 8 位均匀量化 —— 即示例 501。 |
| `"pca, rom, rabitq"` | PCA + 旋转后喂给 RaBitQ 二值量化器。 |
| `"mrle, fp32"` | MRLE 投影再以 fp32 存储（MRLE 必须放在最前）。 |

约束（`transform_quantizer_parameter.cpp:33-45`）：

- 链至少包含 **1 个变换 + 1 个量化器**（长度 ≥ 2）。空串或单 token 会抛
  `INVALID_ARGUMENT`。
- **最后一个 token 必须是 TQ flatten 路径能够 dispatch 的量化器** —— `fp32`、`sq8`、
  `sq8_uniform`、`sq4`、`sq4_uniform`、`bf16`、`fp16`、`pq`、`pqfs`、`rabitq` 之一
  （`src/datacell/flatten_interface.cpp:126-164`）。`TransformQuantizerParameter` 解析层会
  额外接受 `sparse`、`int8`、`tq`，但 flatten 工厂没有针对 `int8`/`tq` 的分发分支，并且当
  `is_transform_quantizer=true` 时显式拒绝 `sparse`
  （`src/datacell/flatten_interface.cpp:166`），因此这三个不能用作 TQ 末端，否则会在构建索引时
  以 "unsupported quantization type" 失败。
- 未识别的变换名会抛 `INVALID_ARGUMENT: invalid transformer name`
  （`transform_quantizer.h:225-227`）。

## 支持的变换

`src/quantization/transform_quantization/transform_quantizer.h:192-227` 的工厂当前识别
4 个变换名：

| 名称 | 输出维度 | 描述 | 实现 |
|---|---|---|---|
| `pca` | 设置了 `pca_dim` 则取该值，否则同输入 | 主成分分析投影；在保留方差的前提下降维。 | `src/impl/transform/pca_transformer.h` |
| `rom` | 同输入 | 随机正交矩阵；旋转向量以让各维去相关。 | `src/impl/transform/random_orthogonal_transformer.h` |
| `fht` | 同输入 | 快速 Hadamard / KAC 随机旋转；`rom` 的低开销变体。 | `src/impl/transform/fht_kac_rotate_transformer.h` |
| `mrle` | `mrle_dim`（≤ 输入维） | 距离可恢复低秩编码；**必须是链中第一个变换**。 | `src/impl/transform/mrle_transformer.h` |

说明：

- `mrle` 必须位于首位由 `transform_quantizer.h:155-159` 强制；`mrle_dim ≤ input_dim`
  由 `transform_quantizer.h:217-220` 强制。
- header 中声明的其它字符串（`residual`、`normalize`）**未**接入工厂，会被拒绝。

## 变换参数

变换 JSON 由 `VectorTransformerParameter::FromJson` 解析
（`src/impl/transform/vector_transformer_parameter.cpp:22-35`）：

| 键 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `pca_dim` | int | `0`（= 输入维） | `pca` 变换的输出维。 |
| `mrle_dim` | int | `0`（= 输入维） | `mrle` 变换的输出维。 |
| `input_dim` | int | 自动 | 由链自动填充 —— 不要手动设置。 |

### HGraph 顶层映射

使用 HGraph 时，两个顶层快捷键会被映射到嵌套的量化器参数中
（`src/algorithm/hgraph.cpp:370-385`）：

- `tq_chain` → `base_codes.quantization_params.tq_chain`
- `rabitq_pca_dim` → `base_codes.quantization_params.pca_dim`

`rabitq_pca_dim` 这个名字早于 Transform Quantizer 引入；当链中包含 `pca` 时，它实际
驱动的是 **`pca` 变换的输出维**（与 RaBitQ 无关）。如果链以 `rabitq` 结尾且未使用
`pca`，则同一个键会配置 RaBitQ 自身的 PCA 预处理
（`src/quantization/rabitq_quantization/rabitq_quantizer_parameter.cpp:30`）。

## Reorder 与精确码存储

变换链在设计上一定有信息损失（旋转无损，但 `pca` / `sq*_uniform` / `rabitq` 有损）。
把 `tq` 与 **reorder** 组合使用 —— 即额外保留一份精确（通常是 `fp32`）副本，对 top 候选
做精排 —— 可以以较小的内存成本恢复精度：

- `use_reorder: true` 会让 HGraph 额外维护一份 flatten 存储，称为**精确码存储**
  （`src/algorithm/hgraph.cpp:76-79`）。
- `precise_quantization_type` 决定精确码使用的量化器（默认 `fp32`；若想用内存换精度，
  也可以设为 `fp16` / `bf16` / `sq8`）。
- 搜索时先用低成本的 `tq` base codes 走图，得到的 top-K 候选再用精确码重新打分
  （`hgraph.cpp:978-981` 及附近调用）。

`use_reorder` 与 `precise_quantization_type` 并非 `tq` 专属 —— 当
`base_quantization_type` 是 `sq8`、`pq`、`rabitq` 等时同样适用。完整的逐索引参数表见
[HGraph 索引](../indexes/hgraph.md)。

## 链该怎么选

经验法则：

| 目标 | 建议链 | 备注 |
|---|---|---|
| 激进压缩 + 精度恢复 | `"pca, rom, sq8_uniform"` + `use_reorder: true`、`precise_quantization_type: "fp32"` | 示例 501 的基线。 |
| 最大压缩 | `"pca, rom, rabitq"` + reorder | 1 bit 量化 + 旋转校正；不开 reorder 精度损失明显。 |
| 各向异性数据、不降维 | `"rom, sq8_uniform"` 或 `"fht, sq8_uniform"` | 高维下用 `fht` 构建成本更低。 |
| 距离保持的低秩 | `"mrle, fp32"` | 度量感知降维，不再量化。 |

请在自有数据上 benchmark —— `tq` 的激进程度与 `use_reorder` 的取舍最终取决于数据分布、
目标召回率以及内存预算。

## 兼容性与合并

两个 `tq` 配置只有在链长度、每一步变换名、最终量化器都完全一致时才被视为兼容
（`src/quantization/transform_quantization/transform_quantizer_parameter.cpp:99-117`）。
这一点对序列化往返以及未来的合并/克隆操作至关重要 —— 准备合在一起的索引，应保持 chain
字符串稳定。

> **chain 字符串一致只是必要条件，并不充分。** `tq_chain` token 列表并不编码变换器参数
> （例如 `pca_dim` / `mrle_dim`，它们作为兄弟 JSON 键单独读取，见
> `src/quantization/transform_quantization/transform_quantizer.h:200-216`），也不编码末端量化器
> 的内部参数（例如 `pq` 子空间数、`rabitq` 旋转种子等）。这些参数会改变实际 code 的维度与
> 布局，因此两个构建要真正可合并/可克隆，必须保持**整套** transform + quantizer 参数一致，
> 不能只对齐 chain 字符串。

## 相关页面

- [HGraph 索引](../indexes/hgraph.md) —— `base_quantization_type`、`use_reorder`、
  `precise_quantization_type` 等参数说明。
- [内存管理](memory.md) —— base + precise 存储的内存开销。
