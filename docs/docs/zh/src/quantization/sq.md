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

`sq8` 和 `sq4` 支持 `l2`、`ip`、`cosine`。距离通过把整数码解码回逐维缩放浮点后
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
- [量化总览](./)

## Local SQ4 / SQ8（float32 L2、IP 和 cosine）

HGraph 的 `base_quantization_type` 可设为 `sq4_local` 或 `sq8_local`；启用 `use_reorder: true` 时，`precise_quantization_type` 也支持两者。两种量化器均支持稠密 float32 的 L2、IP 和 cosine。例如，使用 RaBitQ 遍历并用 local SQ4 精排：

```json
{
    "dtype": "float32",
    "metric_type": "cosine",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "rabitq",
        "use_reorder": true,
        "precise_quantization_type": "sq4_local",
        "max_degree": 32,
        "ef_construction": 300
    }
}
```

将 `sq4_local` 替换为 `sq8_local` 可使用八位精度。直接检索可设置 `base_quantization_type: "sq8_local"` 和 `use_reorder: false`。原有可选 FP32 精排保持不变；local 精排不会隐式增加 FP32 存储或第三阶段精排。`precise_quantization_type` 不保证精确距离：两者均有损，SQ4 较粗的量化等级可能明显降低召回率，应使用自己的数据选择精度和搜索参数。

### 设计与布局

每个向量保存独立的 FP32 最小值 `m` 和 FP32 步长 `s = (max - min) / (2^bits - 1)`。偏移 0–3 为 `m`，4–7 为 `s`，从偏移 8 开始存储编码，无填充及记录对齐要求。

| 类型 | 等级数 | 编码负载 | 每向量字节数 |
| --- | --- | --- | --- |
| `sq8_local` | 256 | 每分量一个无符号字节 | `dim + 8` |
| `sq4_local` | 16 | 偶数分量位于低四位，奇数分量位于高四位 | `ceil(dim / 2) + 8` |

SQ4 奇数维的末字节高四位清零。N 个向量的编码占用 `N * 每向量字节数`，不含图、ID、分配容量及可选其他存储。每个查询计算器需要 `4 * dim` 字节；cosine 编码额外使用临时 `4 * dim` 字节归一化缓冲区。

编码对 `(x - m) / s` 四舍五入（半步向上）并截断到 [0, 255] 或 [0, 15]。常量向量使用零步长及零编码。解码采用融合舍入计算 `m + s * code`，并限制在有限 FP32 端点内。与按坐标训练范围的 `sq8` / `sq4` 不同，范围逐向量计算；与 uniform SQ 不同，查询保持 FP32。

### 距离定义与归一化

- **L2：** FP32 查询与重建数据库向量之间的平方欧氏距离；编码间距离分别重建两侧。
- **IP：** 保留原始幅度，计算 `1 - dot(query, decoded_vector)`；编码间计算 `1 - dot(decoded_a, decoded_b)`。内积包含两侧最小值偏移的贡献。
- **Cosine：** 量化器采用现有 `ScalarQuantizer` 的 `Normalize`，编码前归一化原始数据库向量，ADC 前归一化 FP32 查询，然后计算 `1 - dot`。不重新归一化解码向量；量化可能改变其范数，因此距离不一定严格位于 [0, 2]。零向量保持为零，其与任意向量的 cosine 距离为 1。

HGraph 向这些存储传入原始向量，由量化器负责 cosine 归一化。极端值沿用既有 FP32 归一化语义：范数溢出可能使归一化分量变为零，下溢或非正规数处理可能保留极小分量。本功能不重新定义已有策略。

### 数值与训练约定

拒绝其他输入类型及包裹 local SQ 的变换链。支持负数及常量向量，编码和查询处理拒绝 NaN 与无穷。范围使用 double 中间值；极端有限值记录使用 double 中间值距离回退，重建值限制在有限 FP32 端点内。超出 FP32 范围的距离可能为无穷。极小范围的步长可能舍入为零，也受平台非正规数处理影响。SIMD 结果在浮点容差内一致，不保证逐位相同。

支持 1 至 `INT_MAX` 的维度、奇数维及非对齐记录。共享内核支持标量、SSE、AVX、AVX2、AVX512 和 NEON；SVE 回退到标量。SSE/AVX 在 SIMD 累加前逐分量采用融合舍入重建，使 ADC 与公开解码结果一致。本次本地验证无法执行原生 ARM 路径。

无共享训练边界及额外参数。`Train` / `ReTrain` 无需读取样本即可完成；编码时逐向量计算元数据。任一本地类型用作基础存储时，HGraph 的 `NEED_TRAIN` 为 false。仍遵循 HGraph 正常 Build/Train 生命周期。

### 持久化与 Tune

现有 `sq8`、`sq4`、`sq8_uniform`、默认值及格式不变。

新类型需要支持它们的读取版本。记录及元数据沿用原生二进制序列化约定，不引入跨字节序格式。加载校验维度、度量和记录长度。HGraph 序列化保留基础及精排存储，包括 RaBitQ 加任一 local 量化器。Tune 可从保留的原始数据或 FP32 编码生成这些存储；仅有有损 local 编码时不能执行后续改变精度的 Tune。原有 `ignore_reorder`、`reorder_source` 及 Tune 限制保持不变。

本次变更未开展大规模召回率或吞吐基准测试。
