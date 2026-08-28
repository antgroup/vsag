# RaBitQ x+y Split

RaBitQ x+y split 是 HGraph 和 Pyramid 面向低比特底库码的存储与搜索模式。每条向量拆成
两条记录：

- 图遍历和 lower-bound 过滤只读取 `x` 个 filter bits。
- 只有进入重排的候选才读取 `y` 个 supplement bits。
- 最终重排距离使用完整的 `x+y` bits。

这种布局缩小了图遍历的热数据，同时保留更高精度的 RaBitQ 距离用于最终排序。
它也支持把 filter record 留在内存中，把访问频率更低的 supplement record 放到磁盘。

## 启用 split 模式

当 base 和 precise 的量化类型都为 `rabitq`，并且配置了
`rabitq_bits_per_dim_precise` 时，HGraph 和 Pyramid 自动选择 split 模式：

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
| `rabitq_fused_datacell` | 仅用于 HGraph；启用融合布局，默认值为 `false`。 |
| `train_sample_count` | HGraph 训练采样数，默认 `65536`；设为不小于数据集大小时，fused KMeans 使用全部向量训练。 |

参数约束为：

```text
1 <= x <= 8
1 <= y <= 8
x + y <= 8
```

如果不配置 `rabitq_bits_per_dim_precise`，HGraph 和 Pyramid 使用 standard RaBitQ 路径，
不会创建 split storage。

### HGraph 融合内存布局

仅对 HGraph，将 `rabitq_fused_datacell` 设为 `true` 后，底层节点的邻居、
cluster id、label、x-bit code 和 y-bit supplement 会存入同一个 cache-line
对齐的 record。Pyramid 使用普通 split storage；`rabitq_fused_datacell` 不是
Pyramid 参数。HGraph 专用搜索循环直接读取该 record，并联合预取图邻居和
量化码。codec 使用固定随机种子可复现训练的 16 个 residual clusters。

当数据集大于 `train_sample_count` 时，fused HGraph 使用固定 seed 的均匀 reservoir
sampling 选择指定数量的向量，并在样本上训练 16-cluster KMeans codec。将
`train_sample_count` 设置为数据集大小（例如 `500000`）即可使用全部向量训练。
全量训练可能改善 cluster centroid，但会增加 KMeans 构建时间和工作内存。

该选项创建支持增量修改的内存索引。完成 `Build` 或 Deserialize 后，仍支持
`Add`、mark remove、向量/ID/属性/extra-info 更新、检索（包括过滤、iterator、
Range Search 和并发查询）、按 ID 计算距离、内存统计以及 Serialize/Deserialize。
force remove、Merge、
Tune、Clone、ExportModel 和 Build Cache 导入导出仍不支持。`Build` 和 Deserialize
默认保留 mutable 状态以及搜索锁和节点锁；确认不再修改后，可以显式调用
`SetImmutable`，以终态切换移除搜索路径上的这些锁，之后的修改请求会被
拒绝。

每个 record 以 4-byte 邻居数量开头，随后是纯 `InnerIdType` 邻居 ID；节点不再有
version，邻居 ID 也不编码 version。之后依次保存 cluster id、对齐后的 external
label、filter code 和 supplement，record stride 仍向上对齐到 64-byte。存储可以在
`Build` 以及后续 `Add` 期间按需增长。

优化构建会临时使用 SQ8 scalar code 计算对称的 base-to-base 图距离，构建
完成后立即释放；最终索引只保留 fused RaBitQ record。构建后的 `Add` 会按需
解码 fused record 来更新图，不会常驻保存 SQ8 或原始向量副本；关注插入
吞吐时建议使用批量 Add。

融合布局是显式启用的，并且比普通 split storage 有更严格的约束：

- `1 <= x <= 4`、`y >= 1` 且 `x + y <= 8`。
- metric 必须是 L2 或内积。
- graph、filter code 和 supplement code 必须全部使用内存 IO。
- 必须关闭 MCI、`deduplicate_storage`、remove metadata、reverse edges 和 force remove。
- fused 不支持 PCA；请省略 `rabitq_pca_dim` 或将其设为 `0`。
- 不支持旧版 v0.14 序列化格式。
- fused slab 使用独立 wire version；当前格式会明确拒绝曾包含 node version 和
  remove flags 的开发期旧格式，不提供迁移分支。

未启用该参数的索引保持原有布局、行为和序列化格式。

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

Pyramid 需要把对应搜索参数放在 `pyramid` 下：

```json
{
    "pyramid": {
        "ef_search": 200,
        "rabitq_one_bit_search": true,
        "rabitq_error_rate": 1.9
    }
}
```

外部搜索参数仍命名为 `rabitq_one_bit_search`，但对 split 索引，它会使用
`rabitq_bits_per_dim_base` 配置的全部 `x` 个 filter bits。
`hgraph.rabitq_error_rate` 和 `pyramid.rabitq_error_rate` 可以分别为对应索引的
单次搜索覆盖默认值，且不需要重建索引。原生 HGraph fused record 保存
乘倍率前的几何误差尺度；HNSW-compatible fused `1+7` record 保留按规范
默认值缩放的 metadata，并在查询时按相对该默认值的倍率应用 override。

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

因此，图搜索不会为每个访问到的向量都计算 `x+y` 距离并放入搜索堆。图遍历由
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

当 `x >= 2` 时，HGraph fused 专用 search/reorder 路径会把遍历阶段算出的精确
x-bit filter inner product 直接传给 reorder，并直接从 node record 读取 code，
因此 full rerank 只计算 y supplement planes 对应的第二项：

```text
full contribution = shifted filter contribution + supplement contribution
```

因此 fused `2+y`、`3+y`、`4+y` 会复用精确的 x-bit filter inner product，每个
重排候选只扫描 y 个 supplement planes。携带 filter-IP/full-distance 的 richer
candidate 只存在于 HGraph fused 专用路径；通用 HGraph searcher、Pyramid 和
`ReorderInterface` 继续使用原有 distance/id candidate 协议。
fused `1+y` 的遍历使用 4-bit query bit-plane 与 popcount 近似值；精确重排会
重新计算它的 1-bit 精确贡献，因为该近似值不能作为精确 full-distance hint。
如果没有可用 hint，代码会直接从两个 split records 计算相同的最终距离。

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
`hgraph.rabitq_error_rate` 或 `pyramid.rabitq_error_rate` 不需要。

对于 fused 索引，codec model 随 split datacell 序列化，每个节点的 code 只在
bottom-graph slab 中序列化一次。普通和 streaming 往返都会保留该布局，
不会再生成一份随节点数增长的 split code 副本。

## 实现位置

| 模块 | 文件 / 入口 |
| --- | --- |
| 外部 x/y 参数映射 | `hgraph_param_mapping.cpp`、`pyramid.cpp` |
| split record 和 IO | `src/datacell/rabitq_split_datacell.h` |
| plane 布局和 code 拆分 | `RaBitQuantizer::StoredPlaneIndex`、`SplitCode` |
| filter 距离和 lower bound | `ComputeDistWithOneBitLowerBound` |
| 直接计算 split distance | `ComputeDistWithSplitCode` |
| 使用 filter hint 的 reorder | `ComputeDistWithSplitCodeAndFilterDist`、`ComputeDistWithSplitCodeAndFilterIP` |
| SIMD dispatch | `src/simd/rabitq_simd.cpp` |
| AVX2 / AVX512 lookup kernel | `src/simd/avx2.cpp`、`src/simd/avx512.cpp` |
| 内存/磁盘/混合 IO 示例 | `examples/cpp/323_index_hgraph_rabitq_split.cpp` |

## 使用注意

- split storage 当前可用于 HGraph 和 Pyramid，并且要求 fp32 query code。
  Pyramid 的 split 索引默认启用 one-bit split 搜索路径；如需强制使用普通搜索路径，
  可以在 `pyramid` 搜索参数下传 `rabitq_one_bit_search: false`。
- 支持 `l2`、`ip` 和 `cosine`。当 `x >= 2` 时，canonical 普通 split 路径和
  HGraph fused 路径会为 L2 和内积直接复用精确 filter inner product；其他情况
  会安全地计算完整 split distance。
- fused datacell 只支持 L2、内积以及上文所述的纯内存配置。
- HGraph fused 在 `Build` 和反序列化后保持可变，支持构建后 `Add`、mark remove
  以及向量/ID/属性/extra-info 更新，但不支持 force remove。需要无锁只读检索
  路径时，显式调用 `SetImmutable`。
- 启用 `support_duplicate: true` 时，重复向量 build probe 和展开 alias 的查询使用
  HGraph canonical searcher；fused slab 仍负责保存 code 和 graph。
- 除非已经验证仅靠 x-bit 遍历距离能满足召回要求，否则应保持
  `use_reorder: true`。
- 修改 x、y、metric 或 transform 参数后必须重建索引；在搜索参数中覆盖
  `hgraph.rabitq_error_rate` 或 `pyramid.rabitq_error_rate` 不需要重建。
- RaBitQ 通用说明见 [RaBitQ](rabitq.md)，完整 HGraph 参数见
  [HGraph 索引](../indexes/hgraph.md) 和 [Pyramid 索引](../indexes/pyramid.md)。
