# SINDI DMQ 压缩原理说明

本文简要说明 VSAG SINDI 中 `rerank_type="dmq"` 压缩的对象、方式和大致收益。

## 结论

SINDI 中有两套容易混淆的压缩：

| 开关 | 压缩对象 | 方法 | 用途 |
| --- | --- | --- | --- |
| `use_quantization: true` | 倒排索引里的 value | 全局 8-bit 标量量化 | 降低召回层内存 |
| `use_reorder: true` + `rerank_type: "dmq"` | 精排用的原始稀疏向量副本 | term id bit packing + value 8-bit DMQ | 降低 rerank 副本内存 |

所以，DMQ 主要压缩的是 **SINDI 为精排保存的原始稀疏向量副本**，不是倒排索引本身。

## DMQ 保存什么

开启 `rerank_type: "dmq"` 后，每个非零项保存：

```text
term_id: bit-packed term id
value:   8-bit DMQ code
```

额外还有两类辅助数据：

- 每条文档向量保存 `mean` 和 `alpha` 两个 float；
- 每个出现过的 term 保存一个 256 档码本。

也就是说，term id 仍然要保留，用来判断查询和文档的 term 交集；value 不再保存原始 float，而是保存一个 `0..255` 的 code，解码时结合 term 码本以及文档自己的 `mean`、`alpha`。

## term id 压缩

term id 使用无损的 bit packing。普通 `uint32_t` 每个 term id 固定占 32 bit；如果配置了
`term_id_limit`，SINDI 可以按实际范围决定需要多少 bit。

| `term_id_limit` | 需要 bit 数 | 每个 term id 大小 |
| ---: | ---: | ---: |
| 30,000 | 15 bits | 约 1.875 bytes |
| 1,000,000 | 20 bits | 2.5 bytes |
| 10,000,000 | 24 bits | 3 bytes |
| `uint32_t` 全范围 | 32 bits | 4 bytes |

解码后的 term id 仍是原始整数，精排时仍能准确计算查询和文档的 term 交集。

注意：当前实现中，如果开启 `remap_term_ids: true`，DMQ backend 会按 32 bit 估计 rerank term id。
因此 `remap_term_ids` 主要解决词表 ID 稀疏、空洞大的结构管理问题，不一定带来 DMQ term id bit
packing 的额外收益。

## value 压缩

value 使用 direct 8-bit DMQ。当前参数虽然叫 `dmq_bits`，但实现只接受 `dmq_bits: 8`，即每个
value 主体保存 1 byte。

DMQ 编码流程可以概括为：

1. 对每条文档向量计算 value 均值 `mean_i`。
2. 将每个 value 转成 residual：`residual = value - mean_i`。
3. 为每个 term 单独训练一个 256 档码本 `C_term`。
4. 编码时在对应 term 的码本中选择一个 code：`code = encode(residual, C_term)`。
5. 每条文档保存缩放因子 `alpha_i`，用于校正码本近似值。

解码公式是：

```text
decoded_value = mean_i + alpha_i * C_term[code]
```

这种方式比全局线性量化更适合稀疏向量：不同 term 的权重分布可能差别很大，按 term 建码本可以更贴近各自分布；按文档保存 `mean` 和 `alpha` 则能修正文档内部的整体偏移和尺度。

## 精排打分

稀疏向量内积只处理查询和文档共有的 term。假设查询包含 `term 7`，某篇文档压缩后也包含
`term 7`，精排时会解码该位置并累加：

```text
decoded_x_7 = mean_doc + alpha_doc * C_7[code]
score += q_7 * decoded_x_7
```

最后返回的距离仍是：

```text
distance = 1 - approximate_inner_product
```

为了加速交集计算，文档 term id 会排序保存；查询也会排序。查询 term id 范围较小时，backend
还会建立临时查找表，减少排序 merge 的开销。

## 压缩比例估算

不压缩的 FP32 rerank 副本约为：

```text
term_id: 4 bytes
value:   4 bytes
合计:    8 bytes / nonzero
```

DMQ rerank 副本的非零项主体约为：

```text
term_id: id_bits / 8 bytes
value:   1 byte
合计:    id_bits / 8 + 1 bytes / nonzero
```

| `term_id_limit` | `id_bits` | DMQ 每非零项主体 | 相对 8 bytes | 压缩倍数 |
| ---: | ---: | ---: | ---: | ---: |
| 30,000 | 15 | 2.875 bytes | 约 36% | 约 2.78x |
| 1,000,000 | 20 | 3.5 bytes | 约 44% | 约 2.29x |
| 10,000,000 | 24 | 4 bytes | 50% | 2.00x |
| 32-bit 全范围 | 32 | 5 bytes | 约 63% | 1.60x |

真实内存还要加上：

- 每条文档一个 `EncodedVector` 元信息；
- 每条文档两个 float：`mean` 和 `alpha`；
- 每个出现过的 term 一个 256 档 DMQ 码本；
- term id 到码本下标的映射结构；
- vector 容器 capacity 预留空间。

当前 direct 8-bit DMQ 每个 term 的码本包含：

```text
255 个 threshold + 256 个 decode value = 511 个 float
约 2044 bytes / term
```

因此，term 出现越频繁、文档越长，码本和 per-vector 元信息越容易被摊薄；如果词表很大且大量
term 只出现一两次，码本开销会削弱压缩收益。

## 参数建议

常见配置：

```json
{
  "dtype": "sparse",
  "metric_type": "ip",
  "dim": 1024,
  "index_param": {
    "term_id_limit": 1000000,
    "window_size": 50000,
    "use_reorder": true,
    "rerank_type": "dmq",
    "dmq_bits": 8
  }
}
```

实践上：

- 压缩倒排召回层：开启 `use_quantization`；
- 压缩精排原始向量副本：开启 `use_reorder` 并设置 `rerank_type: "dmq"`；
- 想让 DMQ term id 更省内存：合理设置 `term_id_limit`，不要远大于真实词表范围；
- 词表 ID 极度稀疏或空洞很大时，可以考虑 `remap_term_ids`，但它不等价于 DMQ term id 压缩收益。
