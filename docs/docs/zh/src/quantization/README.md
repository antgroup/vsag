# 量化

向量量化是 VSAG 中权衡内存与召回的核心手段。每种索引类型都通过一个
**基础量化器**（由 `base_quantization_type` 配置）存储向量，并可以额外保留
一个**精确量化器**用于重排（`precise_quantization_type` + `use_reorder:
true`）。本章介绍每一种受支持的量化器：它做什么、接受哪些 JSON 参数、是否
需要训练、支持哪些度量、以及何时选用它。

![量化器选择决策树：按内存预算挑选量化器](../figures/quantization/quantization-overview.svg)

## 存储与搜索流水线

```
                  +---------------------+
   原始向量 ----->|  可选变换           |   (TQ 链：pca / rom / fht / mrle)
                  +----------+----------+
                             |
                             v
                  +---------------------+
                  |   基础量化器        |   fp32 / fp16 / bf16 /
                  |                     |   sq8 / sq4 / sq8_uniform /
                  |                     |   sq4_uniform / pq / pqfs /
                  |                     |   rabitq
                  +----------+----------+
                             |
                             v
                   +-------------------+
                   |   索引存储        |   (HGraph / IVF / Pyramid /
                   |                   |    BruteForce / SINDI)
                   +---------+---------+
                             |
                             v
                    图 / 倒排链路游走
                             |
             +---------------+-----------------+
             |                                 |
    use_reorder: false                use_reorder: true
             |                                 |
             v                                 v
        top-K 结果                  +---------------------+
                                    | 精确量化器          |  重排
                                    | (默认 fp32；        |
                                    |  fp16/bf16/sq8 可)  |
                                    +----------+----------+
                                               |
                                               v
                                          top-K 结果
```

`use_reorder` 与 `precise_quantization_type` 并非某一具体量化器专属——只要
索引支持重排，它们就生效（见
[HGraph](../indexes/hgraph.md)、[IVF](../indexes/ivf.md)、
[Pyramid](../indexes/pyramid.md)）。

## 支持的量化器一览

`src/datacell/flatten_interface.cpp` 的工厂会根据 JSON 中的 `type`
字段分派到具体量化器。

| `base_quantization_type` | 每维位数（约） | 需要训练 | 是否无损 | 典型场景 |
| --- | --- | --- | --- | --- |
| `fp32` | 32 | 否 | 是 | 参考基线 / 精确重排存储 |
| `fp16` | 16 | 否 | 近似无损 | 半精度存储；高维 float 向量的良好默认 |
| `bf16` | 16 | 否 | 近似无损 | 与 `fp16` 同样大小，动态范围更宽 |
| `sq8` | 8 | **是** | 否 | 通用的省内存基线 |
| `sq4` | 4 | **是** | 否 | 激进压缩，不重排时召回会下降 |
| `sq8_uniform` | 8 | **是** | 否 | 全局 min/max，SIMD 友好的 SQ8 |
| `sq4_uniform` | 4 | **是** | 否 | SIMD 友好的 SQ4；支持 `sq4_uniform_trunc_rate` |
| `pq` | ~`pq_bits` × `pq_dim` / `dim` | **是** | 否 | 基于码本，非常紧凑 |
| `pqfs` | 4 × `pq_dim` / `dim` | **是** | 否 | PQ FastScan——SIMD 加速版 PQ |
| `rabitq` | 1 或 HGraph x+y | **是** | 否 | 1 比特 / 低比特 split 二值量化，最强压缩 |
| `tq` | 取决于链路 | 取决于末端量化器 | 否 | [量化变换](../advanced/quantization_transform.md)：在另一个量化器之前串接旋转 / PCA |

`int8` 与 `sparse` 不作为通用的 `base_quantization_type` 暴露：

- `int8` 在使用 `dtype: "int8"` 时被自动选用，并非一种压缩模式。
- `sparse` 为 [SINDI](../indexes/sindi.md) 的倒排链表服务，密集索引不可
  直接选择。

## 训练需求

上表中标记为**是**的量化器实现了 `NEED_TRAIN` 标志，必须先调用 `Build`
（在输入向量上内部完成训练）或显式 `Train` 之后再 `Add`。完整生命周期见
[索引构建与训练](../advanced/build_and_train.md)。

对 HGraph 而言，训练数据就是传给 `Build` 的基础向量；对 IVF 而言，先训练
聚类中心，再把残差喂给所配置的基础量化器。

## 度量兼容性

本章所列量化器全部支持三种稠密度量（`l2` / `ip` / `cosine`）。对 `cosine`，
索引会在量化前对向量做归一化，因此底层量化器看不到原始模长。一些实践
要点：

- `pq` / `pqfs` 在每个子空间上做距离查表；当 `pq_dim` 非常小（≤ 4），
  在 `ip` / `cosine` 上比 `l2` 更容易受各向异性影响。
- `rabitq` 在输入向量去相关的情况下效果最好——要么开启
  `rabitq_use_fht` / `rabitq_pca_dim`，要么用 `tq` 链路如
  `"pca, rom, rabitq"` 包一层。

## 量化器选择

一份实用的决策树：

1. **需要精确距离或精确重排存储？** 用 `fp32`。
2. **只想内存减半且召回基本无损？** 用 `fp16`（若数据动态范围大，例如未
   归一化的嵌入，则用 `bf16`）。
3. **想要约 4× 的内存节省并愿意启用重排？** 用 `sq8`（在 `l2` / `ip` 上
   想要更高 SIMD 吞吐可用 `sq8_uniform`）。
4. **内存紧张、可在重排前承受更大召回损失？** 用 `sq4_uniform`。
5. **高维向量，希望基于码本做强压缩？** 用 `pq`，平台支持 SIMD 路径时用
   `pqfs`。
6. **追求最强压缩（1 比特）并能承担重排代价？** 用 `rabitq`，最好搭配
   `rabitq_use_fht: true` 或 `tq` 链路。

对上述任何一种有损量化器，将 `use_reorder: true` 配合
`precise_quantization_type: "fp32"` 是恢复召回的标准做法，代价是额外内存；
具体行为参考 [HGraph 参数表](../indexes/hgraph.md#parameters)。

## 量化在何处暴露

并非每种索引都将所有参数都暴露为外部 key。当前情况：

- **HGraph** 暴露最完整的集合：`base_quantization_type`、
  `precise_quantization_type`、`use_reorder`、`base_pq_dim`、
  `rabitq_pca_dim`、`rabitq_bits_per_dim_query`、
  `rabitq_bits_per_dim_base`、`rabitq_bits_per_dim_precise`、
  `rabitq_error_rate`、`rabitq_use_fht`、`sq4_uniform_trunc_rate`、`tq_chain`
  （见 `src/algorithm/hgraph.cpp`）。
- **IVF** 暴露 `base_quantization_type`、`base_pq_dim`、通用重排相关 key，
  以及 `rabitq_pca_dim`、`rabitq_bits_per_dim_query`、
  `rabitq_bits_per_dim_base`、`rabitq_version`、`rabitq_error_rate`、
  `rabitq_use_fht` 这些 RabitQ 调参 key。
- **Pyramid** 暴露 `base_quantization_type`、`base_pq_dim`、通用重排相关 key，
  以及 RabitQ 的 PCA、底库/查询位数和 FHT 相关 key。
- **BruteForce** 暴露 `base_quantization_type` 与通用重排相关 key；部分可调项
  （如 `tq_chain`）目前在内部接好但未作为外部 key 暴露。

每种索引的完整参数列表见对应索引页。

## 本章内容

- [FP32（基线）](fp32.md)
- [半精度浮点（FP16 / BF16）](fp16_bf16.md)
- [标量量化（SQ4 / SQ8）](sq.md)
- [Uniform 标量量化（SQ4 / SQ8 Uniform）](sq_uniform.md)
- [乘积量化（PQ）](pq.md)
- [PQ FastScan](pqfs.md)
- [RaBitQ](rabitq.md)
- [RaBitQ x+y Split](rabitq_split.md)
- [量化变换（TQ）](../advanced/quantization_transform.md)
