# DMQ 数学理论与 SIMD 加速研究

## 结论

这份调研面向 SINDI 的下一阶段 DMQ rerank 后端。当前 VSAG 里的
`rerank_type="dmq"` 已经能用 bit-packed term id 和 value code 降低内存，但它还不是
DMQ-main 文档里的完整 DMQ 数学模型：现在的 SINDI 后端是“逐向量 min/max 均匀量化 +
标量 merge 交集 + 标量 bit unpack”。这解释了 Wholenet 上内存下降明显，但 QPS 低于
fp32 rerank。

后续要同时解决两个问题：

- 数学模型上，用 DMQ 的中心化残差、分位点代表值和每向量 rescale 代替当前简单
  min/max 解码。
- 工程实现上，低 bit 走 FastScan 的 `vpshufb` 批量查表，高 bit 走 gather 的替代方案，
  例如 8bit LUT 的低秩分解、分块量化和两阶段精排。

最值得优先验证的路线是：2bit/4bit 作为高速 rerank 或 coarse rerank，8bit 用 rank-1
或 rank-R nibble 分解恢复 QPS，并用小规模 exact/top-M 修正召回。

## 代码线索

DMQ-main 中和本报告直接相关的文件如下：

- `多比特DMQ.md`：中心化 DMQ 推导、trace/LUT 形式、拆出 1bit 的误差界、2bit/4bit/8bit
  SIMD 草案。
- `include/rabitqlib/quantization/rabitq_impl.hpp`：`one_bit_code_dmq_error`、
  `two_bit_code_dmq`、`four_bit_code_dmq` 的量化和 rescale 计算。
- `include/rabitqlib/fastscan/fastscan.hpp`：1bit FastScan、AVX512 fused 64、2bit/4bit
  batch size 常量和 LUT pack。
- `include/rabitqlib/fastscan/highacc_fastscan.hpp`：16bit LUT 的 high-accuracy FastScan。
- `include/rabitqlib/index/query.hpp` 和 `include/rabitqlib/index/estimator.hpp`：查询侧
  LUT 量化、`delta * accumulator + sum_vl` 还原、距离公式。
- `include/rabitqlib/index/ivf/ivf_2bits_dmq_segment.hpp`、
  `ivf_4bits_dmq_segment.hpp`、`ivf_8bits_dmq_segment.hpp`：分段 DMQ、全局 qualifier、
  segment scalar 和 batch scan。
- `include/rabitqlib/index/ivf/ivf_8bits_multirank_dmq_segment.hpp`：8bit 256 项 LUT 的
  nibble rank-1 加性分解，用 `vpshufb` 替代 AVX2 gather。
- 当前 VSAG SINDI 后端在 `src/algorithm/sindi/sindi_rerank_backend.cpp`：它只做按向量
  value code 解码，不包含 DMQ-main 的残差中心化和 LUT/FastScan。

## 1. DMQ 数学理论

### 1.1 中心化残差

设聚类中心为 `c`，数据向量为 `x`，查询向量为 `y`：

```text
r   = x - c
rq  = y - c
a0  = mean(r)
a1  = mean(rq)
r'  = r  - a0 * 1
rq' = rq - a1 * 1
```

这里的关键是 `sum(r') = 0` 且 `sum(rq') = 0`。因此：

```text
||x - y||^2
= ||r - rq||^2
= ||r||^2 - 2<r, rq> + ||rq||^2
= ||r||^2 - 2<r' + a0, rq' + a1> + ||rq||^2
= ||r||^2 - 2<r', rq'> - 2D*a0*a1 + ||rq||^2
```

不使用随机旋转时，直接对 `r` 做符号或均匀量化会受坐标偏置影响。DMQ 的 `a0/a1`
中心化把每个 residual 移到零均值空间，使 `r'` 和查询的对齐关系更稳定。

### 1.2 多 bit 代表值和编码

对每个 cluster、每个维度，从该 cluster 内所有 `r'[d]` 的分布中建立 `b` bit 量化。
需要 `2^b - 1` 个 separator 和 `2^b` 个代表值，所以 qualifier 行数是：

```text
(2^b - 1) + 2^b = 2^(b+1) - 1
```

例如：

- 2bit：3 个 separator，4 个代表值，共 7 行。
- 4bit：15 个 separator，16 个代表值，共 31 行。
- 8bit：255 个 separator，256 个代表值，共 511 行。

对每个维度 `d`，编码是：

```text
code[d] = count(r'[d] > separator[k][d])
u_tilde[d] = f_d(code[d])
```

DMQ-main 里的分位点不是普通等频分位，而是带权重的分位。实现中权重为：

```text
w(v) = v^2 * n + sum(v_i^2) - 2 * v * sum(v_i)
```

它等价于把候选代表值 `v` 到该维度所有样本的平方误差总和作为权重相关量，目的是让
separator/代表值更服务于后续内积估计，而不是只服务于坐标本身的重构误差。

### 1.3 Rescale: 用 `u_tilde` 逼近 `r'`

量化后的 `u_tilde` 与原始 `r'` 不在同一尺度。DMQ 用每个向量自己的尺度修正：

```text
alpha = ||r'||^2 / <u_tilde, r'>
```

于是有近似：

```text
<r', rq'> ≈ alpha * <u_tilde, rq'>
```

代回 L2 距离：

```text
dist(x, y)
≈ ||r||^2 - 2 * alpha * <u_tilde, rq'> - 2D*a0*a1 + ||rq||^2
```

代码里通常把 `-2 * alpha` 存成 `f_rescale`：

```text
f_add     = ||r||^2
f_rescale = -2 * ||r'||^2 / <u_tilde, r'>
f_a0      = a0
f_error   = ||r + 0.5 * f_rescale * u_tilde||
```

搜索时：

```text
est = f_add + g_add + f_rescale * <u_tilde, rq'> - 2D*a1*f_a0
```

其中 L2 下 `g_add = ||rq||^2`。

### 1.4 Trace/LUT 形式

对一个查询和 cluster，`rq'` 固定。定义查询侧 LUT：

```text
V[d][j] = rq'[d] * f_d(j),  j in [0, 2^b)
```

定义 one-hot 矩阵 `X`：

```text
X[code[d]][d] = 1
```

则：

```text
<u_tilde, rq'> = sum_d V[d][code[d]] = trace(VX)
```

这个形式是 SIMD 加速的核心。搜索时不需要把 code 解码成 `u_tilde`，而是把 `code[d]`
直接作为 LUT 下标，累加查询侧预先算好的 `V[d][code[d]]`。

### 1.5 LUT 量化与还原

FastScan 不是把 code 反量化成 float 再乘查询，而是把查询侧浮点 LUT 量化成整数 LUT：

```text
lut_u8[d][j] = round((V[d][j] - min_v) / delta)
delta        = (max_v - min_v) / 255
```

SIMD 累加得到整数 `acc` 后，还原为：

```text
sum_float ≈ delta * acc + sum_vl
```

其中 `sum_vl` 是每个参与维度的 `min_v` 之和。DMQ-main 的普通 `Lut` 对每 4 维构成
16 项表，因此 `sum_vl_lut = vl * num_table`。2bit/4bit segment 代码为了降低量化误差，
按 chunk 计算 `delta` 和 `sum_vl`。

### 1.6 拆分 1bit 和误差界

多 bit code 可以写成：

```text
code = 2^(B-1) * x0 + x_last
```

其中 `x0` 是最高 bit 或第一层粗编码。文档给出的分解是：

```text
f0(x0) = f(0) + (f(2^(B-1)) - f(0)) * x0
flast  = f(code) - f0(x0)
```

于是：

```text
<f(code), rq'> = <f0(x0), rq'> + <flast(x_last), rq'>
```

价值在于：第一项可以用 1bit RabitQ/FastScan 的高速估计和误差界做过滤，第二项只对通过
过滤的候选做补偿。文档中的误差概率界是：

```text
Pr(Error > a) < 2 * exp(
  - D * a^2 / (8 * ||r' - alpha*u_tilde||^2 * ||rq'||^2)
)
```

这给了一个可工程化的两阶段搜索方式：先用 1bit 估计和 lower bound 淘汰不可能进 topK 的
batch，再用 residual/multi-bit 估计精排剩余候选。

### 1.7 Segment 公式

DMQ-main 的 segment 版本用于 MRL/分段渐进搜索，也可借鉴到 SINDI 的 term-block rerank。
设一个维度区间为 `S=[l,r)`，`delta_i = a0_i - a1`，则：

```text
||x_S - y_S||^2
= ||r'_S - rq'_S + delta_i * 1||^2
= ||r'_S||^2 + ||rq'_S||^2 - 2<r'_S, rq'_S>
  + 2*delta_i*(sum(r'_S) - sum(rq'_S)) + |S|*delta_i^2
```

用 DMQ 近似内积项：

```text
-2<r'_S, rq'_S> ≈ f_rescale_S * <u_tilde_S, rq'_S>
```

又因为：

```text
rq'_S = (q - mean(q))_S - (c - mean(c))_S
```

可以把 cluster centroid 相关项提前预计算。DMQ-main 里保存：

```text
seg_frescale[S][i]
seg_rdot[S][i]      = ||r'_S||^2
seg_sigma[S][i]     = sum(r'_S)
seg_rdot_adj[S][i]  = seg_rdot[S][i] - f_rescale_S * <u_tilde_S, c_tilde_S>
```

查询时只需要构建 `<u_tilde_S, q_tilde_S>` 的 LUT，再组合：

```text
est_S = rdot_adj + g_add_S + f_rescale_S * ip_S
        + 2*delta_i*(sigma_r'_S - sigma_q'_S)
        + |S|*delta_i^2
```

2bit/4bit 代码中 `ip_S` 由 FastScan 得到；1bit 代码还会额外加入 `k1x_S`，因为 1bit 的
`f(code)` 被写成 `a + (b-a)*x`。

## 2. 低 bit FastScan 与 SIMD 实现

### 2.1 为什么低 bit 能快

低 bit 的 LUT 每维只有 2、4 或 16 个入口，正好适配 `vpshufb`。`vpshufb` 的每个 128-bit
lane 可以用 16 个 byte 当寄存器内查找表，并用另一个 byte 向量作为下标。因此 4bit 是最
自然的边界：一个 nibble 能直接索引 16 项 LUT。

核心思想是改变 code 布局：不要按“一个向量连续存所有维度”扫描，而是按“一个维度连续存
一批向量的 code”扫描。这样一个 SIMD load 就能拿到很多候选在同一维度的 code，随后用
一次或少数几次 `vpshufb` 得到这一批候选的 LUT 分数。

### 2.2 1bit FastScan

DMQ-main 的 1bit 路线来自 RabitQ/FastScan：每 4 个维度形成一个 4bit 组合，LUT 有 16 项。
`fastscan::pack_lut(dim, query, lut)` 会把 4 个维度的所有 bit 组合和查询值预加好：

```text
lut[j] = sum_{bit k set in j} query[k]
```

搜索时 code 是 1bit packed 的组合码，`fastscan::accumulate` 通过 `vpshufb` 查 16 项表。
普通路径用 uint8 LUT 和 uint16 累加；`highacc_fastscan` 把 uint16 LUT 拆成低 8bit 和高
8bit 两张表，分别查表后合并成 int32，提高长维度和高精度场景下的准确性。

AVX512 路径中还有 fused 64 的优化：两个 32-vector block 共享同一份 LUT load，减少约一半
LUT 带宽，并把距离公式向量化成 16 个 float 一组。

### 2.3 2bit FastScan 布局

2bit 每维有 4 个 code。DMQ-main 的 2bit segment 使用 batch size 128。对每个维度，存
32 个 byte：

```text
byte[k].bits[1:0] = vector k      的 code
byte[k].bits[3:2] = vector k + 32 的 code
byte[k].bits[5:4] = vector k + 64 的 code
byte[k].bits[7:6] = vector k + 96 的 code
```

扫描一个维度时：

```text
raw = load 32 bytes
d0  = shuffle(lut16, raw & 0x03)          // vectors 0..31
d1  = shuffle(lut16, (raw >> 2) & 0x03)   // vectors 32..63
d2  = shuffle(lut16, (raw >> 4) & 0x03)   // vectors 64..95
d3  = shuffle(lut16, (raw >> 6) & 0x03)   // vectors 96..127
```

每个 `d*` 是 32 个 uint8 LUT 值，再拆成低/高 16 个 byte 扩展到 uint16 累加。由于
`256 * 255 = 65280` 接近 uint16 上限，维度超过 256 时要按 chunk 累加，然后提升到 uint32。

### 2.4 4bit FastScan 布局

4bit 每维有 16 个 code，完全匹配 `vpshufb` 的 16 项 LUT。DMQ-main 的 4bit segment 使用
batch size 64。对每个维度，存 32 个 byte，每个 byte 装两个 4bit code。扫描时：

```text
raw = load 32 bytes
lo  = raw & 0x0F
hi  = (raw >> 4) & 0x0F
score_lo = shuffle(lut16, lo)
score_hi = shuffle(lut16, hi)
```

`score_lo` 和 `score_hi` 分别覆盖 32 个候选，合起来是 64 个候选。和 2bit 一样，LUT 先按
chunk 量化到 uint8，累加后用：

```text
ip_float += delta_chunk * acc + sum_vl_chunk
```

还原。

### 2.5 LUT chunking 和 accumulator 宽度

低 bit SIMD 的两个精度风险是：

- uint8 LUT 的动态范围太粗，尤其高维时全局 min/max 会牺牲局部精度。
- uint16 accumulator 可能溢出。

DMQ-main 的 segment 版本用 chunk 解决这两个问题：

```text
chunk_dim = min(1024, segment_dim)
每个 chunk 独立计算 min/max、delta、sum_vl
每个 chunk 独立 FastScan，再在 float 中合并
```

如果不使用 high-accuracy LUT，2bit/4bit 内核最好每 256 维左右提升一次到 uint32；如果使用
16bit hacc LUT，则需要使用拆表路径，内核复杂度更高，但长维度误差更低。

### 2.6 对 SINDI sparse rerank 的适配

SINDI 的难点和 dense IVF 不一样。SINDI rerank 的候选是稀疏向量，当前代码必须先做 sorted
term id 交集，再读取 value code。FastScan 要求一批候选在同一“维度序列”上对齐，因此不能
直接套到当前逐向量 sparse merge 上。

可选的适配方向如下：

1. 当前后端的轻量 SIMD 化。
   保持 per-vector sorted term id，先优化 bit unpack 和交集：固定 `id_bits`/`value_bits` 的
   block unpack、query term 的 dense lookup table 或 hash-free direct array、批量比较 term id。
   这是工程风险最低的 QPS 修复，但不是完整 FastScan。

2. 高频 term-block FastScan。
   把 term id 空间按高频 term 分成小块，例如 16、32 或 64 个 term 一块。每个向量在 block 内
   存 presence mask 和 value code，查询为该 block 构建 LUT。这样一批候选在同一个 block 上
   可以 `vpshufb` 批量查表。缺点是长尾 term 和稀疏缺失会带来 mask 处理和 padding 成本。

3. Posting-list 侧批量打分。
   SINDI 已经按 term 扫 posting list 生成候选。可以在 posting list 中把 value code 按 32/64
   个 posting 分块排列，先用 SIMD 累加候选分数，再对 top candidates 做更精细 rerank。这更像把
   DMQ 下沉到 coarse search，而不是只替换 rerank 存储。

4. Dense segment rerank。
   对特定数据集，如果 term universe 可控且查询/向量稠密度较高，可以把候选投影到固定 segment
   或高频 term 子空间，用 2bit/4bit FastScan 做第一阶段 rerank，再用当前 exact sparse merge 或
   8bit DMQ 做 top-M 修正。

短期建议先做第 1 项，恢复当前 DMQ8 的 QPS；中期做第 2 或第 4 项，才能真正吃到低 bit
FastScan 的吞吐优势。

## 3. 高 bit 的低秩分块 SIMD

### 3.1 8bit gather 为什么慢

8bit 每维有 256 个 LUT 入口，超出 `vpshufb` 的 16 项 lane 内查表能力。DMQ-main 的基础
8bit 实现采用布局 `[dim][batch]`：每维连续存 32 个候选的 code。扫描时每 8 个候选做一次：

```text
idx  = cvtepu8_epi32(load 8 codes)
vals = i32gather_ps(lut_row, idx)
acc += vals
```

这条路的优点是准确，缺点是 gather 延迟高、LUT 访问随机、内存带宽压力大。维度和候选数一高，
吞吐会明显低于 2bit/4bit 的寄存器内查表。

### 3.2 Rank-1 nibble 加性分解

DMQ-main 的 `ivf_8bits_multirank_dmq_segment.hpp` 给了一条很实用的高 bit 加速路线。把
8bit code 拆成高低两个 nibble：

```text
code = 16 * h + l
h, l in [0, 16)
```

对每个维度的 256 项 LUT 重排成 `16 x 16` 矩阵：

```text
M[h][l] = q_tilde[d] * f_d(16h + l)
```

rank-1 加性近似为：

```text
M[h][l] ≈ row_mean[h] + col_mean[l] - grand_mean
```

于是每个 code 的查表可以改成两个 16 项查表：

```text
score(code) ≈ LUT_hi[h] + LUT_lo[l]
```

SIMD 扫描时：

```text
codes = load 32 bytes
hi    = (codes >> 4) & 0x0F
lo    = codes & 0x0F
sum_hi += shuffle(lut_hi16, hi)
sum_lo += shuffle(lut_lo16, lo)
```

这把 8bit 的 gather 路径变成两个 4bit 风格的 `vpshufb` 路径。DMQ-main 中该实现还会打印
rank-1 相对误差，用来评估这种近似对 recall 的影响。

### 3.3 Rank-R 乘性低秩

rank-1 加性分解只能表示行效应和列效应，无法表达 `h` 与 `l` 的 interaction。更一般的低秩
近似是：

```text
M[h][l] ≈ beta0 + A0[h] + B0[l] + sum_{t=1..R} s_t * U_t[h] * V_t[l]
```

工程上可选两类实现：

- 加性多表：继续使用若干组 `A_t[h] + B_t[l]` 近似 residual，优点是仍然只有 shuffle 和 add，
  缺点是表达力有限。
- 乘性 rank-R：分别 `vpshufb` 查 `U_t[h]` 和 `V_t[l]`，再用低精度 multiply-add 累加。
  AVX2 可用 int8/uint8 查表值加 `vpmaddubsw`、`vpmaddwd` 或扩展到 int16 后乘加。它比加性
  rank-1 更贵，但仍可能比 float gather 快，而且准确性更高。

乘性方案需要处理量化零点和符号：

```text
U ≈ scale_u * (u8 - zp_u)
V ≈ scale_v * (u8 - zp_v)
U*V = scale_u*scale_v*(u8*v8 - zp_u*v8 - zp_v*u8 + zp_u*zp_v)
```

因此最好按 chunk 保存 correction 所需的 `sum_u`、`sum_v` 或把表量化成 signed int8，减少
零点修正。初版可以先用 rank-1 加性做 coarse，再对 top-M 用 exact 8bit gather 校正，避免
一次性引入复杂乘性量化。

### 3.4 分块和两阶段校正

高 bit 加速不必只依赖一个近似。更稳的组合是：

- 查询侧按 dimension chunk 量化 LUT，保留每个 chunk 的 `delta` 和 `sum_vl`。
- 扫描阶段用 rank-1 或 rank-R 快速估计所有候选。
- 对进入候选堆附近的 top-M，再用 exact 8bit gather 或标量 float LUT 重算。
- 如果需要更高 recall，可把 top-M correction 与 1bit lower bound filter 组合，先过滤 batch，
  再 exact 修正剩余向量。

这种方式符合当前 Wholenet 目标：保持接近 fp32 的 recall，同时让多数候选走高速近似路径。

### 3.5 Segment 分解对高 bit 的意义

8bit per-cluster qualifier 的内存非常大。DMQ-main 的 8bit segment 版本明确指出：

```text
per-cluster qualifier size = K * 511 * D * sizeof(float)
```

这在大 K、大 D 下会膨胀到数十 GB。因此它改用 global qualifier，并只保存 per-vector segment
scalar：

```text
seg_frescale, seg_rdot, seg_sigma, seg_rdot_adj
```

这背后的工程取舍很适合 VSAG：

- qualifier 用全局或分组共享，避免 per-cluster 表爆炸。
- per-vector 只保留 code 和少量 segment scalar。
- centroid/query 相关项在搜索时或构建时预计算。
- segment 可以对应 dense 维度区间，也可以在 SINDI 中对应 term-block。

## 4. 与当前 SINDI DMQ 后端的差异

当前 VSAG 的 `SINDIDmqRerankBackend` 做的是：

```text
每个 sparse vector:
  sorted term ids -> bit-packed id_codes
  fp32 values     -> per-vector min/max uniform codes
  metadata        -> lower_bound, step, len, offsets

rerank:
  merge(query sorted ids, base packed ids)
  code -> lower_bound + step * code
  inner_product += query_value * decoded_value
```

它没有：

- cluster residual `r=x-c` 和 query residual `rq=y-c`。
- `a0/a1` 中心化。
- weighted quantile qualifier。
- `u_tilde=f(code)` 的每维代表值。
- `f_rescale=-2||r'||^2/<u_tilde,r'>`。
- 查询侧 LUT 和 FastScan batch layout。
- error bound 或 1bit filter。

所以当前实现更准确地说是“低 bit sparse value quantization rerank”，不是完整 DMQ。它能省内存，
但 QPS 被 scalar bit unpack 和 sparse merge 交集限制。

## 5. 建议的落地路线

### 5.1 短期：修复当前后端 QPS

目标是在不改变数学语义的情况下，让 DMQ8 的 QPS 接近 fp32 rerank。

- 为固定 `id_bits` 和 `dmq_bits` 写 block unpack，减少 `LoadId`/`LoadCode` 逐 bit 循环。
- 对 `dmq_bits=8` 走 byte-aligned 快路径，避免 bit-by-bit load。
- 对 query term id 建 direct lookup 或 compact hash，避免双指针 merge 中大量分支。
- 对 sorted id 交集使用 block compare/galloping，优先优化高 nnz 查询。
- 保留当前测试和 Wholenet direct benchmark，确认 recall 不变，只观察 QPS 和 latency。

这一步不会引入完整 DMQ，但能验证 bit-packed memory win 是否能和可接受吞吐共存。

### 5.2 中期：Sparse-native DMQ 数学模型

把当前 value quantizer 替换成稀疏版 DMQ：

- 以 term id 或 term-block 建共享 qualifier，而不是每向量 min/max。
- 对每个向量保存 `a0`、`f_rescale`、`f_error` 等 metadata。
- 查询时为 query nonzero terms 构建 LUT 或 direct value table。
- 对候选交集计算 `<u_tilde, q>`，再套用 DMQ 距离或 IP 估计公式。

SINDI 是 sparse IP，公式需要从 L2 版改写。若目标是最大内积，可以直接近似：

```text
<x, q> = <c + a0 + r', q>
       ≈ <c, q> + a0 * sum(q) + alpha * <u_tilde, q>
```

如果没有显式 cluster center，可以把 term-block/global mean 作为 `c` 的弱替代，或只对 rerank
值做 `alpha*u_tilde` 近似。这个部分需要单独实验确认 recall。

### 5.3 中期：2bit/4bit FastScan rerank stage

在 SINDI 候选产生后，不直接对每个候选做完整 sparse merge，而是：

- 选择高频 term-block 或固定 segment。
- 将候选按 block code layout 组织成 64/128 向量 batch。
- 对 query 的 active block 构建 uint8 LUT。
- 2bit/4bit FastScan 快速估计候选得分。
- top-M 再回到当前 DMQ8 或 fp32/sparse exact 方式精排。

这条路线可以让低 bit FastScan 真正发挥批量吞吐优势，同时把 recall 风险限制在 coarse rerank。

### 5.4 中长期：8bit low-rank SIMD

如果 8bit 是保持 recall 的主力，则优先验证：

1. Exact 8bit gather baseline。
   实现 `[dim][batch]` code layout 和 AVX2 gather，得到准确但偏慢的基线。
2. Rank-1 nibble decomposition。
   使用 `row_mean + col_mean - grand_mean`，把 256 LUT 拆成两张 16 LUT，用 `vpshufb` 扫描。
3. Top-M exact correction。
   rank-1 先扫所有候选，只对靠近堆顶的候选用 exact 8bit LUT 重算。
4. Rank-R 或 residual table。
   若 rank-1 recall 损失明显，再实现 signed int8 的乘性 rank-R，或对 residual 采用小表补偿。

推荐先做 rank-1 + top-M correction，因为它和 DMQ-main 已有代码最接近，工程验证最快。

## 6. 风险和验证指标

### 数学风险

- Sparse IP 和 dense L2 的目标不同，L2 segment 公式不能直接照搬到 SINDI IP。
- 全局 qualifier 降低内存，但可能牺牲 cluster/local 分布适配能力。
- rank-1 8bit 分解有 interaction residual，可能影响最终 topK 顺序。

### 工程风险

- FastScan 需要 batch layout，和当前 per-vector sparse storage 冲突。
- 高维下 uint8 LUT 量化和 uint16 accumulator 都可能引入误差或溢出。
- AVX2/AVX512 分支需要和 VSAG 的编译选项、CPU feature dispatch 风格一致，不能直接使用
  DMQ-main 的 `-march=native` 假设。

### 建议指标

- 内存：index memory、search peak RSS、rerank backend memory、index file size。
- 召回：avg recall、低 recall query 的分布、top-M correction 前后差异。
- 性能：QPS、avg/p95/p99 latency、candidate rerank time 占比。
- SIMD 诊断：每查询 LUT build 时间、scan 时间、exact correction 时间、rank-1 relative error。

## 7. 推荐下一步实验

1. 为当前 SINDI DMQ8 加 byte-aligned code 快路径和 block id unpack，先恢复 QPS。
2. 在独立 microbench 中复现 2bit/4bit FastScan：输入模拟 `[dim][batch]` codes 和 query LUT，
   测量 AVX2/AVX512 scan throughput。
3. 用当前 Wholenet 候选集离线导出 top candidates，验证 rank-1 8bit nibble 分解的排序误差。
4. 做 rank-1 + top-M exact correction 曲线：top-M 取 50、100、200、500，对比 recall/QPS。
5. 如果 rank-1 误差过大，再实现 rank-R int8 乘性补偿或 residual two-stage。

这条路径能把“当前已经证明的内存节省”继续推进到“高 recall 下吞吐也可接受”。