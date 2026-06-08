# 标量量化（SQ4 / SQ8）

`sq8` 与 `sq4` 是**逐维标量量化器**：每个坐标按训练得到的逐维 `[min, max]`
范围，从 `float32` 映射到 8 位（`sq8`）或 4 位（`sq4`）整数。它们共享同
一份实现，仅以位宽参数化，位于
`src/quantization/scalar_quantization/scalar_quantizer.cpp` 与
`scalar_quantizer_parameter.h`。

如果想要 SIMD 更友好、使用**全局** `[min, max]` 的变种，见
[Uniform 标量量化](sq_uniform.md)。

![标量量化：按逐维 [min, max] 范围将坐标映射到 2^b 个 bin 之一](../figures/quantization/sq-axis.svg)

## SQ4 与 SQ8 一览

| 类型 | 每维位数 | 相对 fp32 内存 | 典型精度 | 备注 |
| --- | --- | --- | --- | --- |
| `sq8` | 8 | ~1/4 | 轻微召回下降 | 通用省内存基线 |
| `sq4` | 4 | ~1/8 | 不重排时下降明显 | 激进压缩；配合 `use_reorder: true` |

训练得到的是逐维 `min`/`max`，重尾分布的坐标可能浪费码位。如果数据各
向异性强，可考虑改用 [Uniform 标量量化](sq_uniform.md) 或先旋转的
[量化变换](../advanced/quantization_transform.md) 链路，例如
`"rom, sq8_uniform"`。

## 内存代价（仅码）

- `sq8`：每向量 `dim` 字节。
- `sq4`：每向量 `ceil(dim / 2)` 字节。

此外还有一份小型逐维范围表（`8 × dim` 字节，所有向量摊销）。

## 参数

目前 `sq8` 与 `sq4` 均无量化器专属 JSON 参数
（`scalar_quantizer_parameter.h:36-58`）。位宽仅由 `type` 字符串决定。

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "ef_construction": 300,
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

将 `"sq8"` 替换为 `"sq4"` 即得到 4 位码。

## 训练

设置了 `NEED_TRAIN`。训练从输入向量样本中收集逐维 `min` / `max`。
`Build(base)` 会内部完成训练；对需要显式 `Train` 的索引（如部分 IVF 流程），
请在 `Add` 之前调用。详见
[索引构建与训练](../advanced/build_and_train.md)。

## 度量兼容性

`l2`、`ip`、`cosine`——全部支持。距离通过把整数码解码回逐维缩放浮点后
计算。

## `sq8` 与 `sq4` 如何选

- **`sq8`**：图索引（HGraph、Pyramid）追求约 4× 内存压缩时的默认选择。
  召回损失通常很小，`use_reorder` 经常可选，但搭配
  `precise_quantization_type: "fp32"` 启用重排是最稳妥的配置。
- **`sq4`**：内存紧张且能承担精确重排存储时选用。几乎总是要配合
  `use_reorder: true`。
- 如果数据大致维度同质，改选 `sq*_uniform`；uniform 变种具有更高的 SIMD
  吞吐。
- 对重尾 / 各向异性数据，更推荐前置旋转的
  [量化变换](../advanced/quantization_transform.md) 链路。

## 相关页面

- [Uniform 标量量化（SQ4 / SQ8 Uniform）](sq_uniform.md)
- [量化变换](../advanced/quantization_transform.md)
- [量化总览](README.md)
