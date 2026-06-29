# RaBitQ x+y Split

RaBitQ x+y split 是 HGraph 面向低比特底库码的存储与搜索模式。每条向量拆成
两条记录：

- 图遍历和 lower-bound 过滤只读取 `x` 个 filter bits。
- 只有进入重排的候选才读取 `y` 个 supplement bits。
- 最终重排距离使用完整的 `x+y` bits。

这种布局缩小了图遍历的热数据，同时保留更高精度的 RaBitQ 距离用于最终排序。
它也支持把 filter record 留在内存中，把访问频率更低的 supplement record 放到磁盘。

## 启用 split 模式

当 base 和 precise 的量化类型都为 `rabitq`，并且配置了
`rabitq_bits_per_dim_precise` 时，HGraph 自动选择 split 模式：

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 960,
    "index_param": {
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "use_reorder": true,
        "rabitq_bits_per_dim_query": 32,
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_precise": 5,
        "rabitq_error_rate": 1.9,
        "max_degree": 64,
        "ef_construction": 400
    }
}
```

相关参数如下：

| 参数 | 含义 |
| --- | --- |
| `base_quantization_type` | 必须为 `"rabitq"`。 |
| `precise_quantization_type` | split 模式下同样必须为 `"rabitq"`。 |
| `rabitq_bits_per_dim_base` | `x`，图遍历时读取的 filter bit 数。 |
| `rabitq_bits_per_dim_precise` | `y`，重排时额外读取的 supplement bit 数。 |
| `rabitq_bits_per_dim_query` | split storage 必须使用 `32`。 |
| `rabitq_error_rate` | lower-bound 误差项的默认正数倍率。 |
| `use_reorder` | 建议设为 `true`，使用 `x+y` 距离排序候选。 |

参数约束为：

```text
1 <= x <= 8
1 <= y <= 8
x + y <= 8
```

如果不配置 `rabitq_bits_per_dim_precise`，HGraph 使用 standard RaBitQ 路径，
不会创建 split storage。

使用以下搜索参数启用 filter/lower-bound 搜索路径：

```json
{
    "hgraph": {
        "ef_search": 200,
        "parallelism": 4,
        "rabitq_one_bit_search": true,
        "rabitq_error_rate": 1.9
    }
}
```

外部搜索参数仍命名为 `rabitq_one_bit_search`，但对 split 索引，它会使用
`rabitq_bits_per_dim_base` 配置的全部 `x` 个 filter bits。
`hgraph.rabitq_error_rate` 可以为单次搜索覆盖索引默认值。record 中保存的是乘倍率前的
几何误差尺度，因此 sweep 这个搜索参数不需要重建索引。

## 搜索流程

split 搜索分为四个阶段：

1. query 只做一次变换和归一化；对支持的 filter bit 数，还会为每个 query
   构建一次 byte lookup table。
2. 图遍历只读取 filter record，为每个访问到的向量计算 x-bit 距离估计和
   保守的 lower bound。
3. 重排先丢弃 lower bound 不可能进入结果集的候选，只为剩余候选读取 y-bit
   supplement record。
4. 最终距离把 filter contribution 与 supplement contribution 合成为
   `x+y`-bit RaBitQ 估计。

因此，HGraph 不会为每个访问到的向量都计算 `x+y` 距离并放入搜索堆。图遍历由
低成本的 x-bit 距离驱动，更精确的距离只在候选重排阶段计算。

## 编码和 bit-plane

定义：

```text
d       = 变换后的维度
x       = 每维 filter bit 数
y       = 每维 supplement bit 数
B       = x + y
P       = ceil(d / 8)，一个 bit-plane 的字节数
q_i     = 变换并归一化后的 query 坐标
u_i     = 无符号 B-bit 底库码，0 <= u_i < 2^B
```

完整 code 的中心化表示为：

```text
c_B = (2^B - 1) / 2
z_i = u_i - c_B
N_B = sqrt(sum_i z_i^2)
```

`PackIntoPlanes` 把 `u_i` 的每一个逻辑 bit 存成独立 bit-plane。filter 和
supplement 的划分为：

```text
f_i = floor(u_i / 2^y)    # 高 x bits
s_i = u_i mod 2^y         # 低 y bits
u_i = 2^y * f_i + s_i
```

物理布局让高位 filter planes 连续存储：

```text
filter record:     logical B-1, B-2, ..., B-x
supplement record: logical 0, 1, ..., y-1
```

因此图遍历只需扫描 `x * P` 字节的 plane payload；重排只额外读取 `y * P`
字节的 plane payload，不计元数据和对齐。

## Datacell 布局

`RaBitQSplitDataCell` 内部维护两个 `RaBitQSplitCodeStorage`。

### Filter record

`x_bit_cell_` 中的 filter record 包含：

```text
x 个高位 bit-plane
base norm
x > 1 时的 filter-code norm
可选 MRQ residual norm
IP/cosine 使用的可选 raw norm
lower-bound error
filter approximation error
```

每条向量的 filter plane payload 为：

```text
FilterPlanesSize = x * ceil(d / 8)
```

filter record 是图遍历的热数据。只要 x-bit 估计有效，图搜索和预取都不需要
访问 supplement record。

### Supplement record

`supplement_cell_` 中的 supplement record 包含：

```text
y 个低位 bit-plane
full-code norm
full-code approximation error
当前 metric 和 transform 所需的其他元数据
```

每条向量的 supplement plane payload 为：

```text
SupplementPlanesSize = y * ceil(d / 8)
```

完整 code 的 payload 约为每条向量 `(x+y) * d / 8` 字节，此外还有对齐后的
norm、error 和可选 transform 元数据。

## X-bit filter 距离和 lower bound

第 `i` 维 filter code 为 `f_i`，取值范围 `[0, 2^x - 1]`。定义：

```text
c_x   = (2^x - 1) / 2
N_x   = sqrt(sum_i (f_i - c_x)^2)
S_x   = sum_i q_i * f_i
Q_sum = sum_i q_i
rho_x = (S_x - c_x * Q_sum) / N_x
```

构建索引时，RaBitQ 保存 filter approximation error 的绝对值 `E_x`，并计算
几何误差尺度：

```text
E_safe    = clamp(abs(E_x), 1e-5, 1)
epsilon_x = sqrt(max(0, 1 - E_safe^2) / max(1, d - 1))
```

修正后的 filter 内积估计为：

```text
rho_hat_x = rho_x / abs(E_x)
```

对 L2，设 base norm 为 `N_o`、query norm 为 `N_q`，x-bit 距离和 lower bound 为：

```text
D_x = N_o^2 + N_q^2 - 2 * N_o * N_q * rho_hat_x

LB = D_x
     - 2 * N_o * N_q * rabitq_error_rate * epsilon_x / abs(E_x)
```

实现还会从 `LB` 中减去一个很小的浮点保护量。IP 和 cosine 会按各自的 metric
换算误差项。

lower bound 只用于安全地排除候选。`D_x` 是图遍历距离，最终排序使用完整的
`x+y` 距离。

## Query lookup table 和 SIMD

当 `x = 2` 或 `x = 3` 时，query computer 会构建 FastScan 风格的 byte lookup
table。每一行对应八个 query 坐标，并包含 256 个表项：

```text
LUT[block][byte_value]
    = byte_value 在该 8-D block 中置位位置对应的 q_i 之和
```

随后每个 filter plane 的每个字节只需要查表一次，不必逐坐标解码八次。不同
filter plane 再按二进制权重合成为 `S_x`。

AVX2 和 AVX512 kernel 会同时 gather 多个 LUT 表项，并提供 batch-of-four 路径；
scalar 实现作为可移植 fallback。关键入口为：

- `RaBitQFloatMultiBitIPByLookup`
- `RaBitQFloatMultiBitIPBatch4ByLookup`
- `RaBitQFloatBuildByteIPLookupTable`

不在专用范围内的 x-bit 宽度仍由通用 bit-plane 计算路径支持。

## Reorder 只扫描 y 个 supplement bits

完整无符号 code 满足：

```text
sum_i q_i * u_i
    = 2^y * sum_i q_i * f_i
      + sum_i q_i * s_i
```

对使用 x-bit lookup filter 的 L2 搜索，HGraph 会把之前计算的 filter distance
作为 hint 传给 reorder。`ComputeDistWithSplitCodeAndFilterDist` 从 hint 恢复第一项，
只从 y 个 supplement planes 计算第二项：

```text
full contribution = shifted filter contribution + supplement contribution
```

因此 `3+5` 索引会复用 3-bit filter 结果，每个重排候选只扫描 5 个新的 bit-plane。
如果 hint 不存在或不能使用，代码会回退到 `ComputeDistWithSplitCode`，直接从两个
split records 计算相同的最终距离。

## 内存、磁盘和混合 IO

如果没有单独配置 supplement IO，两个 record 使用相同的 base IO 类型。

### 两个 record 都在内存

```json
{
    "base_io_type": "block_memory_io"
}
```

### 两个 record 都在磁盘

```json
{
    "base_io_type": "async_io",
    "base_file_path": "/data/hgraph_rabitq_split"
}
```

VSAG 会为 filter 和 supplement record 创建不同的 backing path。

### Filter 在内存，supplement 在磁盘

```json
{
    "base_io_type": "block_memory_io",
    "base_supplement_io_type": "async_io",
    "base_file_path": "/data/hgraph_rabitq_split"
}
```

当前支持的 mixed-IO 组合把 `x_bit_cell_` 保存在 block memory，把
`supplement_cell_` 放在 async IO。批量重排时，filter record 通过直接指针读取，
`MultiRead` 只拉取 supplement records。可以显式设置
`base_supplement_file_path`；否则 VSAG 根据 `base_file_path` 生成 supplement path。

## 序列化和加载

使用标准的索引级序列化接口即可，业务侧不需要分别持久化两个 record。

```cpp
std::ofstream out("/path/to/index.bin", std::ios::binary);
auto serialized = index->Serialize(out);

auto loaded = vsag::Factory::CreateIndex("hgraph", index_params).value();
std::ifstream in("/path/to/index.bin", std::ios::binary);
auto deserialized = loaded->Deserialize(in);
```

split datacell 按以下顺序序列化：

1. datacell 基础状态和 supplement IO type。
2. filter storage。
3. supplement storage。
4. RaBitQ quantizer 状态。

创建目标索引时必须使用与序列化索引兼容的参数，尤其是 `dim`、`metric_type`、
x/y bit 数和 query bits。修改编码参数需要重建索引；只调整搜索参数
`hgraph.rabitq_error_rate` 不需要。

## 实现位置

| 模块 | 文件 / 入口 |
| --- | --- |
| 外部 x/y 参数映射 | `src/algorithm/hgraph/hgraph_param_mapping.cpp` |
| split record 和 IO | `src/datacell/rabitq_split_datacell.h` |
| plane 布局和 code 拆分 | `RaBitQuantizer::StoredPlaneIndex`、`SplitCode` |
| filter 距离和 lower bound | `ComputeDistWithOneBitLowerBound` |
| 直接计算 split distance | `ComputeDistWithSplitCode` |
| 使用 filter hint 的 reorder | `ComputeDistWithSplitCodeAndFilterDist` |
| SIMD dispatch | `src/simd/rabitq_simd.cpp` |
| AVX2 / AVX512 lookup kernel | `src/simd/avx2.cpp`、`src/simd/avx512.cpp` |
| 内存/磁盘/混合 IO 示例 | `examples/cpp/323_index_hgraph_rabitq_split.cpp` |

## 使用注意

- split storage 当前是 HGraph 功能，并且要求 fp32 query code。
- 支持 `l2`、`ip` 和 `cosine`；利用 filter hint 的 reorder 快速路径当前针对 L2。
- 除非已经验证仅靠 x-bit 遍历距离能满足召回要求，否则应保持
  `use_reorder: true`。
- 修改 x、y、metric 或 transform 参数后必须重建索引；在搜索参数中覆盖
  `hgraph.rabitq_error_rate` 不需要重建。
- RaBitQ 通用说明见 [RaBitQ](rabitq.md)，完整 HGraph 参数见
  [HGraph 索引](../indexes/hgraph.md)。
