# SINDI DMQ Elias–Fano 正排压缩与检索优化报告

日期：2026-07-30

范围：SINDI DMQ 正排 term ID 的压缩、解码、集合交与候选重排

## 1. 摘要

SINDI 使用倒排表召回 candidate，并使用按文档组织的 DMQ 正排数据完成
reorder。DMQ 将每个非零 value 压缩为 8-bit code，但原有 compact term ID
仍采用定宽 packed 编码。

本次工作利用文档内 term ID 单调有序的特点，引入 Hybrid Elias–Fano（EF）
编码，并将 EF 重排从“完整解码候选文档 ID”优化为“查询 term 驱动的跳跃式
位置查找”。

主要结果：

- 每篇文档分别计算 packed 与 EF 的实际字节数，只在 EF 更小时选择 EF；
- EF payload 不额外保存格式 tag，seek 优化不改变已有 Hybrid EF payload；
- EF 只负责有序 term ID 集合，DMQ value code 继续按 ordinal 平行存储；
- 查询重排通过 EF seek 直接返回 ordinal，再访问 `value_codes[ordinal]`；
- `wholenet-sparse-1m-ip` 上完整索引从 419.49 MiB 降至 401.00 MiB，
  节省 18.49 MiB，即 4.41%；
- DMQ rerank backend 从 146.96 MiB 降至 128.47 MiB，节省 12.58%；
- EF seek 相对 EF 标量全量扫描，Release 延迟降低 5.74%，QPS 提升 5.96%；
- Recall 保持 0.9863；
- 单元测试共 13 个 case、2220 个断言，全部通过。

## 2. 背景

SINDI 的检索分为两个阶段：

```text
查询稀疏向量
    |
    v
倒排 posting 累加
    |
    v
取得 n_candidate 个候选
    |
    v
读取候选的 DMQ 正排编码
    |
    v
重新计算近似内积并返回 top-k
```

一篇包含 \(L\) 个非零项的文档，DMQ 正排布局为：

```text
+----------------------+----------------------+------------------+
| header               | compact term IDs     | value codes      |
| len, mean, alpha     | packed or EF         | L bytes          |
+----------------------+----------------------+------------------+
```

其中：

- header 固定为 12 字节；
- compact term ID 是全局 term 映射后的有序整数；
- 每个 value code 固定为 1 字节；
- 第 \(i\) 个 term ID 与第 \(i\) 个 value code 一一对应。

term ID 与 value code 的对应关系不需要额外 offset：

```text
term IDs:    [id_0, id_1, id_2, ...]
value codes: [q_0,  q_1,  q_2,  ...]
                 同一 ordinal
```

因此只要得到 term 在文档中的 ordinal \(i\)，即可直接读取
`value_codes[i]`。

## 3. Hybrid Packed / Elias–Fano 策略

### 3.1 Compact term ID

训练完成后，所有实际出现的 term 按原始 ID 升序映射到：

$$
[0,V-1]
$$

其中 \(V\) 是 DMQ 词表中的不同 term 数量。

文档的 compact ID 序列满足：

$$
0\le x_0\le x_1\le\cdots\le x_{L-1}<V
$$

正常稀疏向量中的 term ID 唯一；底层 EF 实现同时允许重复值。

### 3.2 Packed 大小

定宽 packed 使用：

$$
b=\max\left(1,\left\lceil\log_2V\right\rceil\right)
$$

位保存每个 compact ID，payload 大小为：

$$
B_{\mathrm{packed}}
=
\left\lceil\frac{Lb}{8}\right\rceil
$$

字节。

### 3.3 Elias–Fano 大小

EF 低位宽度为：

$$
l=
\begin{cases}
\left\lfloor
\log_2\left(\left\lfloor V/L\right\rfloor\right)
\right\rfloor,
& \left\lfloor V/L\right\rfloor>1,\\
0,
& \left\lfloor V/L\right\rfloor\le1.
\end{cases}
$$

低位区域大小为：

$$
B_{\mathrm{low}}
=
\left\lceil\frac{Ll}{8}\right\rceil
$$

高位 bitmap 的有效位数为：

$$
H
=
\left\lfloor\frac{V-1}{2^l}\right\rfloor+L+1
$$

对应字节数为：

$$
B_{\mathrm{high}}
=
\left\lceil\frac{H}{8}\right\rceil
$$

因此：

$$
B_{\mathrm{EF}}
=
B_{\mathrm{low}}+B_{\mathrm{high}}
$$

### 3.4 逐文档选择

Hybrid 策略为：

$$
\operatorname{encoding}
=
\begin{cases}
\text{Elias--Fano},
&B_{\mathrm{EF}}<B_{\mathrm{packed}},\\
\text{packed},
&B_{\mathrm{EF}}\ge B_{\mathrm{packed}}.
\end{cases}
$$

大小相等时继续使用 packed，避免在没有空间收益时承担 EF 解码成本。

格式没有额外 tag。解码端从文档 header 获得 \(L\)，从 DMQ 模型获得
\(V\)，重做同一确定性比较，即可恢复：

- ID payload 的编码类型；
- ID payload 的字节数；
- value codes 的起始位置。

空向量的 ID payload 为 0 字节。

## 4. Elias–Fano 编码布局

每个 ID 拆分为：

$$
\operatorname{low}_i=x_i\bmod 2^l
$$

$$
\operatorname{high}_i
=
\left\lfloor\frac{x_i}{2^l}\right\rfloor
$$

低位按顺序紧密排列。第 \(i\) 个高位在 bitmap 中的置位位置为：

$$
p_i=\operatorname{high}_i+i
$$

由于 \(\operatorname{high}_i\) 单调不减，\(p_i\) 严格递增，因此重复 ID
也可以表示。

最终布局：

```text
+-----------------------------+----------------------------------+
| low bits                    | high bitmap                      |
| L 个 l-bit 整数，紧密排列   | 在 high_i + i 处设置第 i 个 1   |
+-----------------------------+----------------------------------+
```

完整的字节级示例与边界条件见：

- `docs/docs/zh/src/quantization/elias-fano.md`
- `src/impl/elias_fano_stream.{h,cpp}`

## 5. 解码优化演进

### 5.1 原始标量解码

标量 reader 为每个 ID：

1. 从 high bitmap 找到下一个 1；
2. 计算 \(\operatorname{high}_i=p_i-i\)；
3. 从 low bit stream 读取 \(\operatorname{low}_i\)；
4. 重建：

$$
x_i
=
\operatorname{high}_i2^l+\operatorname{low}_i
$$

Packed 解码主要是 load、shift 和 mask；EF 还需要维护两条 bit stream、
执行 trailing-zero count、跨 word refill 和位置重建。因此虽然顺序解码
摊销复杂度为 \(O(1)\)，常数仍明显大于 packed。

### 5.2 Stage A：64-bit word 优化

第一阶段优化包括：

- high bitmap 每次安全加载最多 8 字节；
- 使用 64-bit buffer；
- 使用 trailing-zero count 定位最低置位；
- 使用 `buffer &= buffer - 1` 清除已消费置位；
- 预计算 low mask；
- 增加最多 8 个 ID 的 `ReadBatch`；
- 支持标量与批量 reader 混用。

在 `wholenet-sparse-1m-ip` 的干净 Release 测试中：

| 指标 | Stage A 前 | Stage A 后 | 变化 |
| --- | ---: | ---: | ---: |
| QPS | 2802.13 | 2891.81 | +3.20% |
| 平均延迟 | 2.854 ms | 2.760 ms | -3.30% |
| Recall | 0.9863 | 0.9863 | 不变 |

### 5.3 Layout 复用

早期距离计算会为同一候选重复计算多次 Hybrid layout，用于：

- 判断是否使用 EF；
- 计算 ID payload 字节数；
- 定位 value codes；
- 构造 reader。

当前实现只在候选进入距离计算时计算一次 layout，并复用其结果，消除了
“每个候选重复计算四次 layout”的固定开销。

## 6. 查询驱动的 EF Seek

### 6.1 为什么不应完整扫描

旧 EF 重排方式由文档驱动：

```text
for each document ID:
    解码 id_i
    判断 id_i 是否属于查询
    若匹配则读取 value_codes[i]
```

这会恢复候选文档的全部 \(L\) 个 ID，即使查询只包含很少的 term。
当 `n_candidate=1000` 时，该成本对每次查询重复约 1000 个文档。

EF 本质上只负责有序 ID 集合。更合适的方式是由查询 term 驱动，直接寻找
匹配 term 的 ordinal。

### 6.2 High bitmap 的 bucket

EF high bitmap 中的 0 可以视为 high bucket 分隔符。设第 \(h\) 个 0 的
位置为：

$$
z_h=\operatorname{select}_0(h)
$$

则 high 等于 \(h\) 的 ordinal 范围为：

$$
\operatorname{begin}_h
=
\begin{cases}
0,&h=0,\\
\operatorname{rank}_1(z_{h-1}),&h>0,
\end{cases}
$$

$$
\operatorname{end}_h
=
\operatorname{rank}_1(z_h)
$$

目标 ID \(t\) 拆分为：

$$
h_t=t\mathbin{>>}l,
\qquad
low_t=t\bmod2^l
$$

Seek reader 先取得 high bucket 的 ordinal 范围，再在该范围内二分查找
low bits。

### 6.3 Word-level 跳跃

当前实现不增加 rank/select 辅助索引，而是每次读取一个 64-bit high bitmap
word：

```text
ones  = popcount(word)
zeros = valid_bits - ones
```

如果目标 zero delimiter 不在当前 word，则一次跳过整个 word：

```text
ordinal   += ones
zero_count += zeros
```

如果目标 delimiter 在当前 word，则在 zero mask 中找到目标 zero，并使用
此前消费的 1 数量得到 bucket 的 ordinal 边界。

查询 compact ID 已按升序排列，因此 high cursor 只向前移动；一个候选的
high bitmap 最多扫描一次。

### 6.4 Low bits 随机访问

low bits 是定宽 packed 数组。ordinal \(i\) 的位偏移为：

$$
offset_i=il
$$

因此可以直接读取 `low[i]`，无需解码此前的 low values。同一 high bucket
中的 low values 仍然有序，可以使用 `lower_bound` 和 `upper_bound` 得到
目标 ID 的完整 ordinal range。

返回 range 而不是单个位置，是为了保持底层重复 ID 的语义。正常稀疏向量
term 唯一时，range 长度最多为 1。

### 6.5 Value code 定位

Seek 返回：

```cpp
struct EliasFanoOrdinalRange {
    uint32_t begin;
    uint32_t end;
};
```

DMQ 直接使用 ordinal 访问 value code：

```cpp
for (uint32_t position = range.begin; position < range.end; ++position) {
    qualifier_product +=
        query.code_lut[query_index * 256 + value_codes[position]];
}
```

这里有两个不同的位置：

- `query_index`：term 在查询中的位置，用于选择 query value 和 LUT；
- `position`：term 在候选文档中的 ordinal，用于读取 `value_codes[position]`。

EF 不编码 value，也不需要在 EF payload 中保存 value offset。

### 6.6 复杂度

旧标量完整扫描为：

$$
O(L)
$$

查询驱动 seek 为：

$$
O\left(
\frac{H_s}{w}
+\sum_{q\in Q}\log(k_q+1)
+M
\right)
$$

其中：

- \(H_s\)：扫描到最大查询 high 所经过的 bitmap 位数；
- \(w=64\)：机器字位数；
- \(k_q\)：查询 term 所在 high bucket 的元素数；
- \(M\)：实际匹配数量。

EF 的 \(l\) 选择使 high bitmap 总大小为 \(O(L)\)，因此最坏情况下的
word 扫描约为 \(O(L/64)\)，并且只读取目标 bucket 的 low values。

## 7. 实现改动

### 7.1 EF stream

文件：

- `src/impl/elias_fano_stream.h`
- `src/impl/elias_fano_stream.cpp`
- `src/impl/elias_fano_stream_test.cpp`

新增：

- `EliasFanoStreamLayout`
- `EliasFanoStream::Encode`
- `EliasFanoStreamReader::Read`
- `EliasFanoStreamReader::ReadBatch`
- `EliasFanoSeekReader::FindEqualRange`
- `EliasFanoOrdinalRange`

### 7.2 DMQ

文件：

- `src/quantization/sparse_quantization/sparse_dmq_quantizer.h`
- `src/quantization/sparse_quantization/sparse_dmq_quantizer.cpp`
- `src/quantization/sparse_quantization/sparse_dmq_quantizer_test.cpp`

主要改动：

- 增加 Hybrid ID encoding type；
- 逐文档计算 packed/EF layout；
- 编码时选择更小格式；
- Decode/Compute 支持 packed 与 EF；
- Query ComputeDist 的 EF 分支使用 query-driven seek；
- layout 在单个候选内只计算一次；
- 内存估算同步考虑 packed/EF 最小值。

### 7.3 SINDI 与 DataCell

文件：

- `src/algorithm/sindi/sindi.cpp`
- `src/datacell/sparse_dmq_datacell.cpp`
- `src/datacell/sparse_dmq_datacell_test.cpp`

主要改动：

- SINDI rerank backend 内存估算支持 Hybrid EF；
- 可变长度 DMQ code 的 offset、序列化与恢复保持一致；
- seek 优化不改变 Hybrid EF 索引结构和 DMQ value code 布局。

## 8. 正确性验证

### 8.1 EF 测试

覆盖：

- 单值序列；
- 严格递增序列；
- 单调不减与重复值；
- \(l=0\)；
- 标量 round-trip；
- 完整与不完整 batch；
- 标量和 batch 混用；
- high bitmap 全零 word 跳过；
- seek 命中和缺失；
- 多个 query term 位于同一 high bucket；
- 最后一个不完整字节的 padding；
- seek target 单调约束；
- 乱序和越界输入。

Seek 结果与标准库 `equal_range` 逐项对照。

### 8.2 DMQ 测试

覆盖：

- Hybrid 编码大小不超过 packed；
- EF 编码后的完整 Decode；
- EF seek 距离与完整解码后计算的距离一致；
- 未知查询 term；
- 模型序列化与恢复；
- 共享码本；
- 空模型。

目标测试结果：

```text
13 test cases
2220 assertions
all passed
```

格式检查：

- `clang-format-15 --dry-run --Werror` 通过；
- `git diff --check` 通过。

## 9. 实验环境

主性能数据集：

```text
/root/data/wholenet-sparse-1m-ip.hdf5
```

配置：

| 参数 | 值 |
| --- | ---: |
| base count | 1,000,000 |
| query count | 1,000 |
| metric | inner product |
| top-k | 10 |
| n_candidate | 1,000 |
| query prune ratio | 0 |
| term prune ratio | 0 |
| search threads | 8 |
| build type | Release |

测试使用同一份 Hybrid EF 索引完成标量扫描与 seek A/B，避免构建差异影响。
索引在测试后保留。

## 10. 索引大小

实际索引文件：

```text
Hybrid EF: /tmp/vsag_sindi_dmq_profile_hybrid.index
Packed:    /tmp/vsag_sindi_dmq_profile_packed.index
```

结果：

| 部分 | Hybrid EF | Packed | EF 节省 |
| --- | ---: | ---: | ---: |
| DMQ rerank backend | 128.47 MiB | 146.96 MiB | 18.49 MiB / 12.58% |
| 完整 SINDI 索引 | 401.00 MiB | 419.49 MiB | 18.49 MiB / 4.41% |

精确字节数：

| 格式 | 字节 |
| --- | ---: |
| Hybrid EF | 420,474,836 |
| Packed | 439,864,866 |
| 差值 | 19,390,030 |

完整索引的节省比例低于 rerank backend，是因为 EF 只压缩 DMQ 正排 term ID；
倒排表、value codes、码本和其他公共结构约 272.52 MiB，均保持不变。

## 11. 检索性能

### 11.1 Packed 与标量 EF 的阶段分析

在加入 query-driven seek 之前，对 Packed 与 Hybrid EF 标量路径进行了阶段
计时：

| 格式 | 倒排 candidate | 正排 reorder | 总延迟 |
| --- | ---: | ---: | ---: |
| Packed | 1.510 ms | 0.931 ms | 2.446 ms |
| Hybrid EF 标量 | 1.504 ms | 1.364 ms | 2.890 ms |

结论：

- 倒排 candidate 阶段基本相同，差异属于测量波动；
- 编码格式只影响正排 reorder；
- 标量 EF reorder 比 Packed 慢约 46.6%；
- 当时约 97.7% 的总延迟差来自 reorder。

这说明优化重点应是 EF 集合交和 ordinal 定位，而不是倒排召回。

### 11.2 EF Seek A/B

同一 Release 构建方式、同一 Hybrid EF 索引、相同检索参数下：

| 实现 | 轮数 | 平均延迟 | 平均 QPS |
| --- | ---: | ---: | ---: |
| EF 标量完整扫描 | 5 | 2.905 ms | 2750 |
| EF query-driven seek | 10 | 2.738 ms | 2914 |
| 变化 | — | -5.74% | +5.96% |

两组 seek 分别相对标量基线取得：

- 第一组：延迟降低 7.1%，QPS 提升 7.4%；
- 第二组：延迟降低 4.4%，QPS 提升 4.5%。

结果存在正常运行波动，但两组方向一致。

### 11.3 Recall 与内存

完整质量校验结果：

| 指标 | 结果 |
| --- | ---: |
| Recall@10 | 0.9863 |
| DMQ rerank backend | 134,712,241 bytes |
| 额外 seek 索引 | 0 bytes |

Recall 与 seek 优化前一致。

## 12. 结果文件

性能结果：

- `/tmp/sindi_dmq_scalar_release.json`
- `/tmp/sindi_dmq_seek_release.json`
- `/tmp/sindi_dmq_seek_release_after.json`
- `/tmp/sindi_dmq_seek_recall.json`
- `/tmp/sindi_dmq_profile_hybrid_latency.json`
- `/tmp/sindi_dmq_profile_packed_latency.json`

保留索引：

- `/tmp/vsag_sindi_dmq_profile_hybrid.index`
- `/tmp/vsag_sindi_dmq_profile_packed.index`

## 13. 取舍与限制

### 13.1 空间与速度

EF 节省 term ID 空间，但标量逐项解码常数高于 packed。Query-driven seek
显著缩小差距，但当前结果仍不能说明 EF reorder 已达到 Packed 水平。

### 13.2 查询分布

Seek 最适合查询 term 数远小于候选文档 term 数的场景。短文档或查询非常
稠密时，顺序扫描可能更快，后续可以根据 \(L\) 和查询 term 数自适应选择
seek 或 scan。

### 13.3 非严格随机访问

当前 seek 不保存 rank/select checkpoint，而是以 64-bit word 为粒度向前
跳跃。它不增加索引内存并兼容已有 Hybrid EF payload，但不是任意位置
\(O(1)\) 的随机访问。

### 13.4 SIMD

当前优化使用 word-level `popcount`、trailing-zero count 和定宽 low random
access，属于标量宽字优化。真正 SIMD 化仍需处理：

- 变长 high bitmap；
- 跨 word 边界；
- 多查询 term 的 bucket 定位；
- ordinal 到 value code 的不连续 gather；
- LUT 累加的数据依赖。

### 13.5 索引兼容性

Query-driven seek 没有修改 EF payload，因此本次性能测试中 seek 前生成的
Hybrid EF 索引可以直接复用。

但从更早的 packed-only DMQ 格式升级到 Hybrid EF 时，payload 本身没有保存
encoding tag 或格式版本。旧 packed-only 索引无法仅凭现有 header 与新
Hybrid decoder 自动区分。正式合入前应通过索引版本、显式 encoding metadata
或兼容分支明确处理 legacy packed DMQ 索引。

## 14. 后续建议

建议按以下顺序继续：

1. 增加 seek/scan 自适应阈值，并在不同文档长度和查询长度上测试；
2. 为 Packed 路径实现同样的 query-driven lower-bound，建立公平上限；
3. 统计每个候选的 EF/Packed 选择比例和平均 bucket occupancy；
4. 将 high-word `popcount`、zero select 和 low 比较做手工展开；
5. 评估 BMI2 `pdep` 或平台相关 select-zero 加速；
6. 仅对长文档评估稀疏 rank/select checkpoint，且把 checkpoint 内存计入
   Hybrid 选择；
7. 在 `codefilter-10k-384-angular-f32` 和更多稀疏数据集上补充泛化结果；
8. 持续分别记录 candidate 与 reorder 阶段，避免仅看总 QPS。

## 15. 结论

Hybrid Elias–Fano 证明了 SINDI DMQ 正排 term ID 可以在不改变 value code
布局、且不为每篇文档增加格式 tag 的前提下获得实际空间收益。Query-driven
seek 继续复用相同 Hybrid EF payload；packed-only legacy 索引兼容性仍需在
正式合入前显式解决。

在 `wholenet-sparse-1m-ip` 上：

- DMQ rerank backend 节省 12.58%；
- 完整索引节省 4.41%；
- query-driven EF seek 相对 EF 标量扫描降低 5.74% 延迟；
- QPS 提升 5.96%；
- Recall 保持 0.9863。

EF 在该方案中的核心职责不是 value 解码，而是压缩有序 term ID 集合并返回
集合交命中的 ordinal。DMQ 再利用 ordinal 访问平行存储的 value code。
从完整扫描改为查询驱动 seek，是本次检索性能优化的关键。
