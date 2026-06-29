# FAQ 常见问题

本页整理 VSAG 用户在选型、调参和接入过程中最常遇到的问题。
更多细节可继续阅读各主题页面。

## 我应该选择哪个索引？

VSAG 中常用索引面向不同场景。
建议按数据类型、规模、召回和延迟目标来选。

`hgraph` 是默认推荐的稠密向量索引。
它适合文本、图像、多模态 embedding 等在线检索场景，
通常用于高召回、低延迟查询。
它支持多种量化、增量插入、删除、重排和自动调优。

`ivf` 适合超大规模数据、高吞吐批量查询、内存较紧张的场景。
它通过分桶减少扫描范围，通常比图索引更省内存，
但同等召回下可能需要更多调参。

`sindi` 用于稀疏向量检索，例如 BM25、SPLADE、BGE-M3 sparse 输出。
它只接受 `dtype: "sparse"`，并且当前主要使用 `metric_type: "ip"`。

`pyramid` 适合多租户、分区、标签路径类场景。
它在一个索引内部组织多棵子图，便于按 tag 或路径分区检索。

`brute_force` 是暴力搜索，适合小数据集、功能验证、构造召回率 baseline。
它结果精确，但大规模下延迟和吞吐通常不可接受。

经验建议：

- 不确定选什么，稠密向量先从 `hgraph` 开始。
- 稀疏向量选 `sindi`。
- 小数据集或验证算法效果选 `brute_force`。
- 超大规模、吞吐优先、可接受分桶召回折中时，对比 `ivf`。
- 有明显分区、租户或路径结构时考虑 `pyramid`。

相关页面：[索引总览](../indexes/README.md)、[最佳实践](best_practices.md)。

## 为什么同一套参数在不同数据集上的性能差很多？

这是向量检索里很常见的问题。
即使数据量、维度、索引参数完全相同，
不同数据集的搜索难度也可能差很多。

原因是数据分布不同：

- 有些数据集近邻结构清晰，查询很容易沿图走到正确区域。
- 有些数据集近邻边界模糊，需要扩展更多候选才能达到同样召回。
- embedding 的归一化方式、聚类程度、维度分布和噪声水平，
  都会影响搜索难度。

对 HGraph 这类图索引，`ef_search` 是影响召回和延迟的核心搜索参数。
它控制搜索时保留和扩展的候选规模：

- `ef_search` 越大，召回率通常越高。
- `ef_search` 越大，单次查询延迟通常也越高。
- 在其他条件接近时，`ef_search` 和查询延迟通常近似线性相关。

因此，比较不同数据集性能时，
不建议只看“同一个 `ef_search` 下的 QPS”。
更合理的方式是：

1. 先在每个数据集上分别调 `ef_search`。
2. 让它们达到相同目标召回率，例如 95% recall 或 98% recall。
3. 再比较 P50 / P95 / P99 延迟和 QPS。

如果数据集 A 达到 95% recall 只需要 `ef_search = 80`，
而数据集 B 需要 `ef_search = 300`，
那么 B 的延迟显著高于 A 是正常现象。
这说明 B 的检索难度更高，不一定是索引退化。

建议在性能报告中同时记录：

- 数据集名称和规模。
- 维度。
- 索引参数。
- 目标 recall 和实际 recall。
- `ef_search`。
- QPS。
- P50 / P95 / P99 latency。

相关页面：[HGraph](../indexes/hgraph.md)、[性能评估工具](eval.md)。

## `sq8_uniform` 为什么通常比 `sq8` 更快？什么时候该开 `use_reorder`？

`sq8` 和 `sq8_uniform` 都是 8-bit 标量量化，
但它们的缩放方式不同。

`sq8` 是逐维量化：

- 每个维度都有自己的 `min_i` / `max_i` / `scale_i`。
- 好处是每个维度都能适应自己的数值范围。
- 坏处是距离计算时需要处理逐维 scale，热路径更复杂。

`sq8_uniform` 是全局 uniform 量化：

- 所有维度共享同一套 `min` / `max` / `scale`。
- query 和 base code 更容易直接在整数域计算。
- SIMD、AVX-512、AMX、NEON 等向量化路径更友好。
- 距离计算可以减少逐元素反量化和逐维 scale 操作。

所以在数据分布适合时，`sq8_uniform` 往往比 `sq8` 更快。

适合用 `sq8_uniform` 的场景：

- 向量已经归一化，尤其是 `cosine` 场景。
- 各维度数值范围比较接近。
- 查询瓶颈主要在距离计算。
- 吞吐和延迟比极致召回更重要。
- 可以配合 `use_reorder` 修正粗排误差。

不太适合的场景：

- 不同维度数值范围差异很大。
- 向量由多个异构特征块拼接而成。
- 存在明显重尾维度或离群值。
- 不打算开启 reorder，且对召回非常敏感。

`use_reorder` 的作用是：
先用压缩后的 base quantizer 做粗排，
再用更高精度的 precise quantizer 对候选结果重打分。

常见配置：

```json
{
    "index_param": {
        "base_quantization_type": "sq8_uniform",
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

建议开启 `use_reorder` 的情况：

- 使用 `sq4`、`sq4_uniform`、`pq`、`pqfs`、`rabitq` 等有损程度较高的量化方式。
- 使用 `sq8` 或 `sq8_uniform` 后召回不够稳定。
- `topk` 较小，但最终排序质量要求高。
- 内存允许多保存一份更高精度表示。
- 线上更关注召回稳定性，而不是极限内存压缩。

可以不开 `use_reorder` 的情况：

- `fp32` 或 `fp16` 已经满足召回。
- `sq8_uniform` 不开 reorder 时召回已经达标。
- 内存预算非常紧。
- 延迟极敏感，不能接受重排开销。

简单建议：

- 高吞吐优先：先试 `sq8_uniform`，不开 reorder 测召回。
- 稳妥配置：`sq8_uniform` + `use_reorder: true`，
  并设置 `precise_quantization_type: "fp32"`。
- 强压缩配置：`sq4_uniform` / `pq` / `rabitq` 通常建议配 reorder。

相关页面：[Uniform 标量量化](../quantization/sq_uniform.md)、
[标量量化](../quantization/sq.md)。

## `l2`、`ip`、`cosine` 的距离语义是什么？

VSAG 的搜索结果统一按“距离越小越相似”排序。
即使底层是 inner product 或 cosine similarity，
返回值也会转换成距离语义。

具体语义：

- `l2` 返回 `L2Sqr`，也就是平方 L2 距离。
- `ip` 返回 `1 - inner_product`。
- `cosine` 返回 `1 - cosine_similarity`。

为什么 `l2` 返回平方距离？
平方 L2 距离和 L2 距离的排序完全一致，
省掉开方可以提升性能。
因此 VSAG 内部和返回值通常使用 `L2Sqr`。

这会影响 `RangeSearch` 的 radius 设置：

- 如果你希望 L2 距离小于 `2.0`，传入的 radius 应该是 `4.0`。
- 如果使用 `ip`，半径对应的是 `1 - inner_product`。
- 如果使用 `cosine`，半径对应的是 `1 - cosine_similarity`。

例如 cosine 相似度希望大于等于 `0.8`：

```text
distance = 1 - cosine_similarity
radius = 1 - 0.8 = 0.2
```

注意：

- 不同系统可能返回 similarity，也可能返回 distance。
- 和其他库或 ground truth 对比时，要先确认距离语义是否一致。
- 索引创建后，`metric_type` 不能在搜索时切换。

相关页面：[度量语义](metric_semantics.md)、[范围搜索](../advanced/range_search.md)。

## `base_quantization_type` 和 `precise_quantization_type` 有什么区别？应该怎么设置？

这两个参数分别控制粗排存储和重排存储。

`base_quantization_type` 是主存储量化方式：

- 用于索引内主要向量存储。
- 用于图搜索或倒排扫描阶段的粗排距离计算。
- 直接影响内存占用、搜索速度和粗排召回。
- 常见值包括 `fp32`、`fp16`、`bf16`、`sq8`、`sq8_uniform`、`pq` 等。

`precise_quantization_type` 是重排用的高精度量化方式：

- 只有在 `use_reorder: true` 时生效。
- 用于对粗排候选进行二次精排。
- 目的是修正有损量化带来的距离误差。
- 常见值是 `fp32`，也可以根据内存预算选择 `fp16`、`bf16`、`sq8` 等。

可以理解为：

```text
base_quantization_type    = 用什么格式快速找候选
precise_quantization_type = 用什么格式重新计算候选距离
```

高召回基线：

```json
{
    "index_param": {
        "base_quantization_type": "fp32"
    }
}
```

内存和召回折中：

```json
{
    "index_param": {
        "base_quantization_type": "sq8_uniform",
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

更省内存：

```json
{
    "index_param": {
        "base_quantization_type": "sq4_uniform",
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

更激进压缩：

```json
{
    "index_param": {
        "base_quantization_type": "pq",
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

设置建议：

- 召回优先、内存充足：`base_quantization_type: "fp32"`。
- 通用线上推荐：`base_quantization_type: "sq8_uniform"`，必要时开启 reorder。
- 数据分布不适合 uniform：尝试 `sq8`。
- 内存紧张：尝试 `sq4_uniform`、`pq`、`rabitq`，并开启 reorder。
- 如果开启 `use_reorder`，`precise_quantization_type` 默认优先考虑 `"fp32"`。

注意：`dtype` 是输入数据类型。
`base_quantization_type` 是索引内部存储和计算方式。
两者不是一回事。
例如输入可以是 `dtype: "float32"`，
但内部用 `base_quantization_type: "sq8_uniform"` 存储。

相关页面：[量化总览](../quantization/README.md)、[HGraph](../indexes/hgraph.md)、
[IVF](../indexes/ivf.md)。

## 过滤搜索应该用 Bitset、lambda、`Filter`、属性过滤还是 `extra_info`？

VSAG 提供多种过滤方式，适合不同使用场景。

`Bitset` 过滤适合已有一批需要排除的 `id`，
例如删除集合、黑名单集合、权限不可见集合。
`Bitset::Test(id) == true` 表示这个 `id` 被过滤掉。

`lambda` 或 `std::function<bool(int64_t)>` 适合简单过滤逻辑。
回调返回 `true` 表示该 `id` 被过滤掉。

`Filter` 对象适合更复杂的过滤逻辑，
也适合需要向搜索算法提供 `ValidRatio()` 等提示信息的场景。
`Filter::CheckValid(id) == true` 表示保留该 `id`。

属性过滤适合结构化字段过滤，
例如 `category = "book" AND price <= 100`。
它通过 `SearchRequest` 使用，适合“向量 + 结构化条件”的混合搜索。

`extra_info` 过滤适合每条向量附带一段固定长度字节数据的场景。
HGraph 可以在图遍历过程中基于这段字节做过滤。
`Filter::CheckValid(const char*) == true` 表示保留对应向量。

如何选择：

- 只想排除一批 `id`：用 `Bitset`。
- 过滤逻辑简单：用 `lambda`。
- 过滤逻辑复杂，且能估计通过率：用 `Filter` 对象。
- 过滤条件是结构化字段：用属性过滤。
- 元数据是固定长度字节，并希望和向量一起存储：用 `extra_info`。

最容易混淆的是 true / false 语义：

- `Bitset::Test(id)` 返回 `true` 表示过滤掉该 `id`。
- `lambda` 返回 `true` 表示过滤掉该 `id`。
- `Filter::CheckValid(id)` 返回 `true` 表示保留该 `id`。
- `Filter::CheckValid(const char*)` 返回 `true` 表示保留对应向量。

使用位图过滤时，`id` 最好控制在 `[0, 2^32)` 范围内，
避免低 32 位冲突。
如果过滤谓词非常严格，图搜索可能需要扩展更多候选才能凑够结果。
对 HGraph 可以考虑设置 `brute_force_threshold`，
让高选择性过滤自动走暴搜回退。

相关页面：[带过滤的搜索](../advanced/filtered_search.md)、
[属性过滤](../advanced/attribute_filter.md)、
[Extra Info（附加信息）](../advanced/extra_info.md)。
