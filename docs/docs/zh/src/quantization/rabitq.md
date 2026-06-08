# RaBitQ

`rabitq` 是 VSAG 的二值 / 低比特量化器。默认模式下每个坐标用 **1 比特**
编码，给出所有内建量化器中最高的压缩率。另一种模式
（`rabitq_version = "split_1bit_7bit"`）把表示拆分为 1 比特基础 + 7 比特
精化，在保留 1 比特快速距离内核的同时，以约 8 比特/维换回大部分精度。

![RaBitQ：按坐标相对随机超平面的符号进行编码](../figures/quantization/rabitq-hyperplane.svg)

> 实现：`src/quantization/rabitq_quantization/rabitq_quantizer.cpp`，
> 参数文件 `rabitq_quantizer_parameter.cpp`。
> 设计说明：`docs/rabitq_1xbit_new_repo_guide.md`、
> `docs/rabitq_split_1bit_7bit.md`。

## 何时使用

- **最强压缩。** 1 比特码是稠密向量可能的最小存储。
- **高维嵌入**——旋转 + 二值化后仍能保留足够近邻搜索所需的几何信息。
- 配合精确重排存储（`fp16` / `fp32`）——标准做法就是 "RaBitQ + 重排"，
  因为 1 比特距离本身噪声较大。

为获得最佳精度，请同时启用 `rabitq_use_fht: true`，或者用
[量化变换](../advanced/quantization_transform.md) 链路如
`"pca, rom, rabitq"` 包一层。

## 内存代价（仅码）

- `rabitq_bits_per_dim_base = 1`：每向量 `ceil(dim / 8)` 字节。`dim = 768`
  时为 96 字节（对比 fp32 的 3072 → 小 32×）。
- `rabitq_bits_per_dim_base = 8`（split 1+7 模式存储更多比特）：每向量
  约 `dim` 字节。

## 参数

| Key | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `pca_dim` | int | `0`（= 输入维度） | RaBitQ 内部可选的 PCA 预处理维度。`0` 表示不做 PCA 降维（`rabitq_quantizer_parameter.cpp:30-32`）。 |
| `rabitq_bits_per_dim_query` | int | `32` | 搜索时**查询**的每维位数。允许值：`4` 或 `32`（`rabitq_quantizer_parameter.cpp:38-43`）。 |
| `rabitq_bits_per_dim_base` | int | `1` | **底库**（存储）码的每维位数。范围 `[1, 8]`（`rabitq_quantizer_parameter.cpp:45-54`）。纯 1 比特 RaBitQ 取 `1`。 |
| `rabitq_version` | string | `"standard"` | 取值：`"standard"`（1 比特）或 `"split_1bit_7bit"`。split 模式要求 `rabitq_bits_per_dim_query = 32`（`rabitq_quantizer_parameter.cpp:55-67`）。 |
| `rabitq_error_rate` | float | `1.9` | 控制编码器误差预算；必须为有限正数（`rabitq_quantizer_parameter.cpp:68-75`）。 |
| `use_fht` | bool | `false` | `true` 时在二值化前应用快速 Hadamard 变换旋转。以 O(dim log dim) 的廉价代价提升各向异性数据上的精度（`rabitq_quantizer_parameter.cpp:76-78`）。 |

在 HGraph 上，这些以顶层 key 暴露：`rabitq_pca_dim`、
`rabitq_bits_per_dim_query`、`rabitq_bits_per_dim_base`、`rabitq_version`、
`rabitq_error_rate`、`rabitq_use_fht`。其中 `rabitq_use_fht` 是 HGraph
对量化器内部 `use_fht` key 的别名，由索引层重写（`src/algorithm/hgraph.cpp:473-480`，
名称定义见 `src/constants.cpp:142-148`）。Pyramid 同样暴露相应的 `rabitq_*` key
（`src/algorithm/pyramid.cpp:698-699`）。

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 768,
    "index_param": {
        "base_quantization_type": "rabitq",
        "rabitq_use_fht": true,
        "rabitq_pca_dim": 0,
        "rabitq_bits_per_dim_base": 1,
        "rabitq_bits_per_dim_query": 32,
        "max_degree": 32,
        "ef_construction": 300,
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

切换到高精度的 split 模式。split 布局由两个 key 共同决定：
`rabitq_version: "split_1bit_7bit"` 选择 1+7 RaBitQ 编码，
`base_codes_type: "rabitq_split"` 切换存储 datacell。仅设置
`rabitq_version` **不会**走 split datacell 路径，二者必须同时设置（详见
`docs/rabitq_split_1bit_7bit.md`）：

```json
{
    "base_quantization_type": "rabitq",
    "base_codes_type": "rabitq_split",
    "rabitq_version": "split_1bit_7bit",
    "rabitq_bits_per_dim_base": 8,
    "rabitq_bits_per_dim_query": 32,
    "rabitq_use_fht": true
}
```

## 训练

设置了 `NEED_TRAIN`。训练学习让 1 比特编码均衡的旋转与逐维统计。可选的
FHT 旋转是固定的（无需学习），因此不增加训练代价；PCA 预处理（`pca_dim
> 0`）会训练一个投影矩阵。

## 度量兼容性

`l2`、`ip`、`cosine`——全部支持。二值距离内核是对 XOR 后的码字做 popcount；
对 `ip` / `cosine`，实现还会追踪一份残差范数，使内积估计无偏。

## 实践要点

- **始终启用重排**，除非你已经验证 1 比特召回在你的数据上可接受。
  `use_reorder: true` + `precise_quantization_type: "fp32"` 是稳妥默认。
- **先旋转。** 对未归一化数据，设 `rabitq_use_fht: true`，或在 `tq` 链路
  中包含 `rom` / `fht`。
- **精度优先时用 split 模式。** `rabitq_version: "split_1bit_7bit"` 保留
  图遍历的 1 比特快速路径，再添加 7 比特精化用于重排；相对纯 1 比特，
  代价约为 8× 码大小，召回明显更高。

## 相关页面

- [量化变换](../advanced/quantization_transform.md)
- [HGraph 索引](../indexes/hgraph.md)
- 设计说明：`docs/rabitq_1xbit_new_repo_guide.md`、
  `docs/rabitq_split_1bit_7bit.md`
- [量化总览](README.md)
