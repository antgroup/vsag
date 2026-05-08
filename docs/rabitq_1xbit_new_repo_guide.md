# RabitQ 1+xbit 新仓库实现与实验说明

本文汇总 `/root/vsag` 新仓库中 RabitQ split 路径的公式、代码实现、参数配置、运行方法、
索引和实验结果存储位置，以及 GIST1M 实验结果解读。

## 1. 当前支持范围

新仓库中的 split RabitQ 用一个 one-bit search record 加若干 supplement bit planes 存储
base code。当前实现的实际语义是：

```text
B = rabitq_bits_per_dim_base
x = B - 1
1+xbit = 1 个 one-bit search plane + x 个 supplement planes
```

约束如下：

| 参数 | 当前约束 | 说明 |
| --- | --- | --- |
| `rabitq_version` | `"split_1bit_7bit"` | 启用 split storage。名字保留 1+7bit，但底层按 `B` 参数化。 |
| `rabitq_bits_per_dim_query` | `32` | split path 只支持 fp32 query code。 |
| `rabitq_bits_per_dim_base` | `1..8` | 因此 `x=0..7`。`x=8` 需要 `B=9`，当前参数校验不允许。 |
| `base_codes_type` | `"rabitq_split"` | 选择 split datacell，而不是普通 flatten datacell。 |

示例：

| 配置 | 实际含义 |
| --- | --- |
| `rabitq_bits_per_dim_base = 1` | `1+0bit` |
| `rabitq_bits_per_dim_base = 2` | `1+1bit` |
| `rabitq_bits_per_dim_base = 4` | `1+3bit` |
| `rabitq_bits_per_dim_base = 8` | `1+7bit` |

## 2. 公式说明

### 2.1 多 bit base code 编码

设随机旋转、中心化、归一化后的 base 向量为 `o'`，base 总 bit 数为：

```text
B = rabitq_bits_per_dim_base
```

每一维编码成无符号整数：

```text
u_i in [0, 2^B - 1]
```

中心点为：

```text
c_B = (2^B - 1) / 2
```

完整多 bit RabitQ code 代表的方向近似为：

```text
y_i = u_i - c_B
y_norm = sqrt(sum_i y_i^2)
```

源码中 `EncodeExtendRaBitQ` 生成每维的 `u_i`，`PackIntoPlanes` 把 `u_i` 拆成 bit-plane
存储。

### 2.2 bit-plane 内积

final full-distance 计算时，完整 base code 被视为 `B` 个 bit-plane。第 `b` 个逻辑 bit-plane
的二进制内积是：

```text
s_b = sum_i q'_i * bit_b(u_i)
```

这里 `q'` 是处理后的 fp32 query。注意 final full-distance 路径传给
`RaBitQFloatBinaryIP` 的 `inv_sqrt_d` 是 `0`，因此 bit 被解释为 `0/1` mask：

```text
bit = 1 -> 乘 1
bit = 0 -> 乘 0
```

所有 bit-plane 按二进制权重合成：

```text
ip_yu_q = sum_{b=0}^{B-1} 2^b * s_b
        = sum_i q'_i * u_i
```

再恢复到中心化、多 bit RabitQ 内积估计：

```text
ip_bq_estimate = (ip_yu_q - c_B * sum_i q'_i) / y_norm
ip_est = ip_bq_estimate / base_error
```

L2 距离估计为：

```text
dist = ||base||^2 + ||query||^2 - 2 * ||base|| * ||query|| * ip_est
```

### 2.3 one-bit search 和 lower bound

split path 把逻辑最高位 `B-1` 放入 one-bit record。图搜索阶段只读这个 record：

```text
ip_sign_q = <q', sign_plane>
ip_est_1bit = ip_sign_q / one_bit_error
dist_1bit = ||base||^2 + ||query||^2 - 2 * ||base|| * ||query|| * ip_est_1bit
```

这里 one-bit search 调用 `RaBitQFloatBinaryIP(query, one_bit_code, dim, inv_sqrt_d)`，
`inv_sqrt_d = 1 / sqrt(dim)`，因此 bit 被解释为 `+1/sqrt(dim)` 或 `-1/sqrt(dim)`。

lower-bound error 元数据在构建时写入 one-bit record：

```text
safe_one_bit_error = clamp(one_bit_error, 1e-5, 1.0)
low_bound_error = rabitq_error_rate
                * sqrt(max(0, 1 - safe_one_bit_error^2) / max(1, dim - 1))
```

搜索时 lower bound 为：

```text
lower_bound = dist_1bit
            - 2 * ||base|| * ||query|| * low_bound_error / one_bit_error
```

HGraph 会把 one-bit graph search 中收集到的 lower-bound candidates 传给 reorder。reorder
按 lower bound 过滤候选，必要时直接读取 one-bit record 和 supplement record 计算 `1+xbit`
full distance，避免在每个候选上先合并完整 full-code buffer。

## 3. 代码实现位置

### 3.1 参数校验

文件：`src/quantization/rabitq_quantization/rabitq_quantizer_parameter.cpp`

职责：

- 校验 `rabitq_bits_per_dim_query` 只能是 `4` 或 `32`。
- 校验 `rabitq_bits_per_dim_base` 在 `[1, 8]`。
- 校验 split version 只支持 `rabitq_bits_per_dim_query = 32`，base bits 沿用全局 `[1, 8]`
  约束，因此 split path 覆盖 `x=0..7`。
- 把 `rabitq_version` 和 `rabitq_error_rate` 写入兼容性检查。

### 3.2 RabitQ quantizer

文件：`src/quantization/rabitq_quantization/rabitq_quantizer.cpp`

关键函数：

| 函数 | 作用 |
| --- | --- |
| `EncodeExtendRaBitQ` | 把归一化 base 向量编码成 `B` bit 标量 code。 |
| `PackIntoPlanes` | 把每维 `u_i` 拆成 `B` 个 bit-plane。 |
| `StoredPlaneIndex` | split storage 中把逻辑最高位 `B-1` 放到物理 plane 0。 |
| `RaBitQFloatSQIPByPlanes` | full-code layout 入口，会把 one-bit plane 和 supplement planes 交给 fused kernel。 |
| `RaBitQFloatSQIPBySplitCode` | final reorder 中直接从 one-bit plane 和 supplement planes 计算 `ip_yu_q`。 |
| `RaBitQFloatSplitCodeIP` | fused SIMD kernel，一次 query pass 解码 one-bit 和 `x=0..7` 个 supplement planes。 |
| `SplitCode` | 把完整 code 拆成 one-bit record 和 supplement record。 |
| `MergeSplitCode` | 把 one-bit record 和 supplement record 合回完整 code，保留给兼容接口和 fallback。 |
| `ComputeDistWithOneBitLowerBound` | one-bit search 距离和 lower bound 计算。 |
| `ComputeDistWithSplitCode` | 直接基于 split records 计算 full distance，避免 reorder 热路径 merge full code。 |
| `ComputeQueryBaseImpl` | full distance 的入口，处理 binary、multi-bit RabitQ 等路径。 |

核心 full-distance 逻辑等价于：

```cpp
float ip_yu_q = 0.0F;
for (int b = 0; b < num_bits_per_dim_base_; ++b) {
    float sb = RaBitQFloatBinaryIP(query, GetStoredPlane(planes, b, plane_bytes), dim, 0);
    ip_yu_q += sb * float(1U << b);
}
ip_bq_estimate = (ip_yu_q - c_B * query_sum) / y_norm;
ip_est = ip_bq_estimate / base_error;
```

### 3.3 split datacell

文件：`src/datacell/rabitq_split_datacell.h`

职责：

- 维护两个 IO：`one_bit_io_` 和 `supplement_io_`。
- `write_encoded_vector` 构建完整 RabitQ code 后调用 `SplitCode` 拆分存储。
- `QueryWithDistanceLowerBound` 只读取 one-bit record，用于图搜索。
- `Query` 和 `GetCodesById` 仍可合并出完整 full code，服务兼容调用路径。
- `compute_full_dist` 是 final reorder 热路径，会直接读取 one-bit 和 supplement 并调用
  `ComputeDistWithSplitCode` 计算 full distance；只有 direct split compute 不可用时才 fallback 到
  `MergeSplitCode`。

### 3.4 HGraph search 和 reorder

相关文件：

- `src/algorithm/hgraph.h`
- `src/algorithm/hgraph.cpp`
- `src/impl/searcher/basic_searcher.cpp`
- `src/impl/reorder/flatten_reorder.cpp`

关键语义：

- `reorder_source = "base"` 时，final reorder 使用 `basic_flatten_codes_`。
- `has_precise_reorder() = use_reorder_ && !reorder_by_base_`，因此 base reorder 不会走 SQ8
  precise codes。
- one-bit search 且 base reorder 时，HGraph 会把 lower-bound candidates 从 searcher 传给
  reorder。
- `FlattenReorder` 已有 lower-bound 排序结果后，会直接对仍可能进入 topK 的候选调用 full
  `Query`；不再先走 `QueryWithDistanceFilter` 重算 one-bit lower bound，避免 final reorder 中
  重复读取 one-bit record。
- RabitQ base-only 建图不能直接用 RabitQ base-base distance，因此 HGraph 中有临时 SQ8 build data
  路径，用于保持 build 可用。

## 4. 底层数据存储布局

这一节说明“当前数据底层如何存储”。这里的“数据”指每个 base vector 编码后写入
`RaBitQSplitDataCell` 的 RabitQ code。业务侧看到的是一个 HGraph 索引目录或序列化文件，内部
split datacell 会把每条向量的 RabitQ code 拆成两个连续存储区。

### 4.1 完整 RabitQ full code 布局

在 quantizer 内部，完整 RabitQ base code 先按 full code 组织。设：

```text
dim = 向量维度
B = rabitq_bits_per_dim_base = x + 1
plane_bytes = ceil(dim / 8)
align(size) = 按 float/error/norm 字段大小对齐后的 size
```

完整 full code 的主要字段顺序如下：

```text
offset_code_:
  B 个 bit-plane，总大小 align(plane_bytes * B)

offset_norm_code_:
  norm_code，也就是多 bit code 的 y_norm
  仅当 B != 1 时存在

offset_norm_:
  base 原始向量经随机旋转、中心化前后的 norm

offset_error_:
  base_error，用于把内积估计恢复为 RabitQ 距离估计

offset_mrq_norm_:
  可选，PCA/MRQ 残差 norm

offset_raw_norm_:
  可选，IP/COSINE metric 使用

offset_low_bound_error_:
  split path 额外写入，用于 one-bit lower bound

offset_one_bit_error_:
  split path 额外写入，用于 one-bit 距离估计
```

GIST1M 当前是 `dim=960`、L2、无 MRQ，因此：

```text
plane_bytes = ceil(960 / 8) = 120 bytes
one-bit plane = 120 bytes
每增加 1 个 supplement bit，额外增加约 120 bytes/point 的 plane 数据
```

### 4.2 bit-plane 物理顺序

`PackIntoPlanes` 会把每维无符号 code `u_i` 拆成 `B` 个 bit-plane。逻辑 bit `b` 的权重是
`2^b`。split storage 会调整物理顺序：

```text
逻辑最高位 bit (B - 1) -> 物理 plane 0
逻辑低位 bit 0..B-2  -> 物理 plane 1..B-1
```

这样做的目的，是让图搜索阶段只读物理 plane 0，也就是 one-bit search plane。final reorder
需要完整距离时，会把物理 plane 0 作为最高逻辑位、把 supplement 中的剩余 planes 作为低位，
直接按逻辑权重计算 full distance。

### 4.3 one-bit record 布局

`SplitCode` 会把 full code 拆出 one-bit record，写入 `one_bit_io_`。每个向量一条 record，
位置为：

```text
one_bit_offset(id) = id * one_bit_code_size_
```

one-bit record 主要字段：

```text
0:
  one-bit plane，大小 plane_bytes

OneBitRecordNormOffset():
  base norm

OneBitRecordMrqNormOffset():
  可选 MRQ norm

OneBitRecordRawNormOffset():
  可选 raw norm，IP/COSINE 使用

OneBitRecordLowBoundErrorOffset():
  low_bound_error

OneBitRecordOneBitErrorOffset():
  one_bit_error
```

对于 GIST1M L2、`dim=960`，one-bit record 近似为：

```text
120B one-bit plane + 4B norm + 4B low_bound_error + 4B one_bit_error = 132B/point
```

这个 record 是 search hot path 主要读取的数据；`QueryWithDistanceLowerBound` 和 prefetch 都只触碰
`one_bit_io_`。

### 4.4 supplement record 布局

剩余内容写入 `supplement_io_`。每个向量一条 supplement record，位置为：

```text
supplement_offset(id) = id * supplement_code_size_
```

supplement record 主要字段：

```text
0:
  剩余 x 个 supplement bit-plane，大小 plane_bytes * x

SupplementMetaOffset():
  full code 中除 bit-plane 以外的元数据，包含 norm_code、base_error 等字段
```

其中：

```text
x = B - 1
SupplementPlanesSize() = plane_bytes * x
GetSupplementCodeSize() = SupplementPlanesSize() + (full_code_size - CodeMetaOffset())
```

因此 GIST1M L2 下 supplement plane 大小随 x 线性增长：

| 变体 | B | x | supplement planes 大小/point |
| --- | ---: | ---: | ---: |
| 1+1bit | 2 | 1 | 120B |
| 1+2bit | 3 | 2 | 240B |
| 1+3bit | 4 | 3 | 360B |
| 1+4bit | 5 | 4 | 480B |
| 1+5bit | 6 | 5 | 600B |
| 1+6bit | 7 | 6 | 720B |
| 1+7bit | 8 | 7 | 840B |

final full-distance 热路径计算时，`compute_full_dist` 会同时读取：

```text
one_bit_io_[id]
supplement_io_[id]
```

然后调用 `ComputeDistWithSplitCode(computer, one_bit, supplement, dist)`。该函数直接读取
supplement meta 中的 `norm_code`、`base_error` 等字段，并调用 `RaBitQFloatSQIPBySplitCode`
从 split bit-planes 计算完整多 bit RabitQ 距离。

`GetCodesById` 仍会调用 `MergeSplitCode(one_bit, supplement, full_code)` 生成完整 code，用于需要
完整 code buffer 的兼容接口；它不再是 final reorder 的默认热路径。

### 4.5 datacell 序列化顺序

`RaBitQSplitDataCell::Serialize` 的顺序是：

```text
FlattenInterface 基础信息
one_bit_io_
supplement_io_
RabitQ quantizer 模型参数
```

反序列化时按同样顺序读回，并调用 `refresh_code_sizes()` 从 quantizer 恢复
`one_bit_code_size_` 和 `supplement_code_size_`。因此业务侧不需要知道两个 IO 的内部格式；只要
使用 VSAG 的索引级 `Serialize` / `Deserialize` 即可。

### 4.6 HGraph 索引目录中的存储

benchmark 中每个 index_path 指向一个 HGraph 索引目录，例如：

```text
/root/vsag/benchs/indexes/gist1m/vsag_hgraph_rabitq_split_1plus4bit_base_reorder
```

目录内部由 VSAG 的 IO/serialization 机制管理，包含 HGraph 基础信息、label table、base split
datacell、bottom graph、route graph 等内容。对于 split RabitQ，base split datacell 中包含上面
描述的 `one_bit_io_` 和 `supplement_io_` 两块数据。

需要删除或重建某个 bit 数的索引时，只删除对应 `index_path` 目录即可，不要混用不同
`rabitq_bits_per_dim_base` 的索引目录。

## 5. 参数配置指南

### 5.1 传统 baseline：1bit search + SQ8 reorder

```json
{
  "dim": 960,
  "dtype": "float32",
  "metric_type": "l2",
  "index_param": {
    "base_quantization_type": "rabitq",
    "max_degree": 64,
    "ef_construction": 300,
    "precise_quantization_type": "sq8",
    "use_reorder": true,
    "build_thread_count": 32,
    "graph_storage_type": "compressed"
  }
}
```

搜索参数：

```json
{
  "hgraph": {
    "ef_search": 200
  }
}
```

### 5.2 split 1+xbit：one-bit search + base full-code reorder

把 `x` 转成 `B=x+1`，例如 `1+4bit` 需要设置
`rabitq_bits_per_dim_base = 5`。

```json
{
  "dim": 960,
  "dtype": "float32",
  "metric_type": "l2",
  "index_param": {
    "base_quantization_type": "rabitq",
    "base_codes_type": "rabitq_split",
    "rabitq_version": "split_1bit_7bit",
    "rabitq_bits_per_dim_query": 32,
    "rabitq_bits_per_dim_base": 5,
    "rabitq_error_rate": 1.9,
    "max_degree": 64,
    "ef_construction": 300,
    "precise_quantization_type": "sq8",
    "use_reorder": true,
    "reorder_source": "base",
    "build_thread_count": 32,
    "graph_storage_type": "compressed"
  }
}
```

搜索参数：

```json
{
  "hgraph": {
    "ef_search": 200,
    "rabitq_one_bit_search": true
  }
}
```

参数要点：

- `base_codes_type = "rabitq_split"` 和 `rabitq_version = "split_1bit_7bit"` 必须同时出现。
- `rabitq_bits_per_dim_base = x + 1`，当前支持 `x=0..7`。
- `reorder_source = "base"` 是本次迁移的关键参数，表示 final reorder 使用 base split full code。
- YAML 中保留 `precise_quantization_type = "sq8"` 不会改变 base final reorder 语义；
  `reorder_source = "base"` 时 high precise reorder codes 不参与 final reorder。
- `rabitq_error_rate` 影响构建时写入的 lower-bound metadata，修改后需要重建索引。

## 6. benchmark 建议

本特性可以使用 `tools/eval/eval_performance` 做 baseline 与 split 配置对比，也可以 sweep
`rabitq_bits_per_dim_base` 观察不同 supplement bit 数对 recall、QPS、build time 和内存的影响。
具体数据集路径、YAML 配置、运行日志、索引文件和图表属于本地实验记录，不随本说明提交。

建议至少覆盖以下组合：

- 传统 RabitQ one-bit search 加 precise/SQ8 reorder 的 baseline。
- split `1+7bit`，并设置 `reorder_source = "base"`。
- split `1+xbit` sweep，其中 `rabitq_bits_per_dim_base = x + 1`，`x=0..7`。
- `rabitq_one_bit_search = true` 时不同 `ef_search` 和 `parallelism` 的性能点。

本地 GIST1M sweep 结果文件约定：

- 合并表：`/root/vsag/benchs/results/gist1m_rabitq_1xbit_sweep_merged_table.csv`
- 高召回 QPS/Recall 图：`/root/vsag/benchs/results/figures/gist1m_recall_vs_qps_1xbit_sweep_high_recall.svg`
- 报告：`/root/vsag/benchs/results/gist1m_rabitq_1xbit_sweep_report.md`

当前 `1+7bit base reorder` 行已刷新为 direct split-code full-distance 结果；对应曲线在高召回图中
使用菱形 marker，方便和其他 `1+xbit` 曲线区分。由于本次只复用已有索引重测 search/reorder，CSV
中 build time、build memory 和 TPS 仍保留原建图结果。

## 7. 索引保存和加载

业务侧不需要直接保存 one-bit 文件或 supplement 文件。使用 VSAG 标准序列化接口即可。

保存：

```cpp
std::ofstream out("/path/to/index.bin", std::ios::binary);
auto ret = index->Serialize(out);
out.close();
```

加载：

```cpp
auto loaded = vsag::Factory::CreateIndex("hgraph", index_params).value();
std::ifstream in("/path/to/index.bin", std::ios::binary);
auto ret = loaded->Deserialize(in);
in.close();
```

加载时必须使用兼容的 `index_params`，尤其是：

- `base_codes_type = "rabitq_split"`
- `rabitq_version = "split_1bit_7bit"`
- `rabitq_bits_per_dim_query = 32`
- `rabitq_bits_per_dim_base` 与构建时一致
- `rabitq_error_rate` 与构建时一致
- `reorder_source` 与构建时一致

## 8. 注意事项

- 当前 `split_1bit_7bit` 版本名没有随 `x` 改名，实际由 `rabitq_bits_per_dim_base` 控制总 bit 数。
- `x=8` 当前不支持，因为 `rabitq_bits_per_dim_base` 上限是 `8`。
- `rabitq_one_bit_search` 是搜索参数，不改变索引存储格式。
- 若修改 `rabitq_error_rate`、`rabitq_bits_per_dim_base`、`reorder_source` 等参数，应重建索引。