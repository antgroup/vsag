# DMQ（稀疏向量分布维持量化）

DMQ（Distribution Maintenance Quantization，分布维持量化）是 VSAG 为
[SINDI](../indexes/sindi.md) 重排正排存储实现的稀疏向量量化方法。当前实现
名为 `dmq8`：每个非零 value 使用一个 8-bit code，每篇文档额外保存均值和
缩放因子；term ID 则先映射为紧凑 ID，再选择定宽 bit-pack 或 Elias–Fano
编码。

本文描述的是 VSAG 中 `SparseDmqQuantizer` 的具体算法。DMQ 目前不是密集索引
可以通过 `base_quantization_type` 选择的通用量化器。

## 背景

SINDI 使用倒排表完成候选召回。启用 `use_reorder` 后，还需要保存一份按文档
组织的完整稀疏向量，用于对候选重新计算内积。若一篇文档含有 `L` 个
非零项，直接保存 `uint32` term ID 和 `float32` value，仅两组数组
就需要约 `8L` 字节；大规模语料中，这份正排副本会成为显著的内存开销。

普通的全局均匀量化没有利用稀疏检索数据的两个特点：

- 不同 term 的 value 分布可能明显不同；
- 同一篇文档内的 value 通常具有文档相关的中心和尺度。

DMQ 因此组合了三个层次的压缩：

1. **按文档中心化：** 每篇文档保存自己的均值，将 value 转换为残差；
2. **按 term 建模：** 高频 term 使用独立码本，低频 term 可共享码本；
3. **按文档校准：** 每篇文档保存一个缩放因子，使量化代表值适配
   该文档。

此外，DMQ 只保存实际出现 term 的紧凑 ID，并利用一篇文档内 term ID 有序的
特点压缩 ID 序列。

## 符号

设训练集包含文档集合 `D`。文档 `d` 的稀疏向量写作

```text
x_d = {(t_{d,i}, x_{d,i}) | i = 0, ..., L_d - 1}
```

其中：

- `L_d` 是文档的非零项数；
- `t_{d,i}` 是 term ID；
- `x_{d,i}` 是对应 value；
- `K = 2^8 = 256` 是 DMQ8 的码字数量；
- `τ` 是低频 term 共享码本阈值，默认值为 1024。

## 训练

### 1. 文档中心化

DMQ 先计算每篇训练文档非零 value 的均值：

```text
μ_d = (1 / L_d) Σ_i x_{d,i}
```

并得到残差：

```text
r_{d,i} = x_{d,i} - μ_d
```

空向量的均值定义为 0。只对非零项求均值，不把稀疏向量中未出现的维度
当作 0 参与计算。

### 2. term 分桶与共享码本

记 term `t` 在训练集中的出现次数为 `f_t`。VSAG 按以下规则分配训练桶：

```text
f_t <= τ    -> 所有低频 term 进入同一个共享桶
f_t >  τ    -> term t 使用自己的独立桶
```

每个桶收集其所属 term 的全部残差，并训练一个 256 项码本。设置
`dmq_shared_codebook_threshold: 0` 时，每个已出现的 term 都使用独立码本。

共享低频码本可以避免为样本很少的 term 分别保存约 2 KiB 的码本，同时让
这些 term 合并训练样本。代价是不同低频 term 共用同一残差分布模型。

### 3. 分布加权码本

设某个训练桶包含 `N` 个残差，排序后为
`z_0 ≤ z_1 ≤ ⋯ ≤ z_{N-1}`。实现为每个样本定义权重：

```text
w_i = Σ_j (z_i - z_j)²
    = N z_i² + Σ_j z_j² - 2 z_i Σ_j z_j
```

总权重为：

```text
W = Σ_i w_i
```

`w_i` 衡量残差 `z_i` 与桶内整体分布的平方距离。靠近分布主体的
样本权重较小，远离主体的样本权重较大，因此码本会给分布尾部保留更多
分辨率，而不是简单地按样本数做等频切分。

定义累计权重 `C_i = Σ_{j=0}^{i} w_j`。对于 `k=0,…,K-1`，码本代表值为：

```text
c_k = z_min{i : C_i >= ((2k + 1) / (2K)) W}
```

对于 `k=0,…,K-2`，编码阈值为：

```text
θ_k = z_min{i : C_i >= ((k + 1) / K) W}
```

因此 256 个代表值位于相邻阈值的加权中间位置，255 个阈值负责把
残差映射到 `[0, 255]`。若桶内残差完全相同，使 `W` 接近 0，则所有
代表值和阈值都设为该残差值。

## 编码

### 1. 紧凑 term ID

训练完成后，所有实际出现的 term 按原始 ID 升序排列，并映射为
`[0, V - 1]` 中的 compact ID，其中 `V` 是不同 term 的数量。定宽编码
需要的位数为：

```text
b = max(1, ceil(log₂ V))
```

长度为 `L` 的 ID 序列使用 bit-pack 时占用：

```text
B_packed = ceil(Lb / 8) bytes
```

当前序列化格式还会计算同一有序 ID 序列的 Elias–Fano 大小，并逐向量选择
更小的表示。令

```text
l = floor(log₂(V / L))，当 V / L <= 1 时 l = 0
```

则当前 Elias–Fano 布局的大小为：

```text
B_low  = ceil(Ll / 8)
B_high = ceil((((V - 1) >> l) + L + 1) / 8)
B_EF   = B_low + B_high
```

仅当 `B_{EF} < B_{packed}` 时选择 Elias–Fano；大小相等时仍使用 bit-pack。
上述 Elias–Fano 公式假设 `L > 0`；空向量的 ID payload 为 0 字节。
完整的字节布局、编码示例、流式解码状态和复杂度分析见
[Elias–Fano 有序整数压缩](elias-fano.md)。

#### Elias–Fano 数据布局

设一篇文档的有序 compact ID 为
`x_0 ≤ x_1 ≤ ... ≤ x_{L-1} < V`。对每个 ID 按 `l` 位拆分：

```text
low_i  = x_i & (2^l - 1)
high_i = x_i >> l
```

编码结果由两个连续区域组成：

```text
+---------------------------+--------------------------------+
| low bits                  | high bits                      |
| L 个 l-bit 整数，紧密排列 | 在位置 high_i + i 处设置一个 1 |
+---------------------------+--------------------------------+
```

低位区域按 ID 顺序保存 `low_i`。高位区域不直接保存 `high_i`，而是把第 `i`
个 1 放在：

```text
p_i = high_i + i
```

因为 ID 单调不减，`p_i` 严格递增；即使两个 ID 相同，它们在高位 bitmap
中仍占据不同位置。bitmap 的长度为：

```text
((V - 1) >> l) + L + 1 bits
```

例如 `V=16`、ID 序列为 `[3, 5, 8, 13]` 时，`L=4`、`l=2`：

```text
low  = [3, 1, 0, 1]
high = [0, 1, 2, 3]
p    = [0, 2, 4, 6]
```

因此高位 bitmap 的第 0、2、4、6 位为 1。

当前格式不为每篇文档额外保存“packed/EF”标记。解码器从 header 得到 `L`，
从 DMQ 模型得到 `V`，使用与编码端相同的大小比较规则，即可确定 ID 区域
采用哪一种格式及 value codes 的起始偏移。

#### Elias–Fano 编码过程

编码器先调用 `GetLayout(L, V)` 计算 `l` 和两个区域的字节数，然后清零输出：

1. 检查每个 ID 小于 `V`，且序列单调不减；
2. 把 `low_i` 写入低位 bit stream 的第 `i·l` 位；
3. 在高位 bitmap 的第 `high_i+i` 位设置 1。

当前 `store_packed` 逐位写低位，因此实现成本为 `O(L·l)`；对固定的
32-bit ID 上界可视为 `O(L)`。输出空间为：

```text
O(L log₂(V/L) + L) bits
```

Elias–Fano 的收益主要出现在 `L` 相对 `V` 较小、且 ID 有序的文档。Hybrid
策略仍逐文档比较实际字节数，避免在 EF 不能节省空间时使用它。

#### Elias–Fano 流式解码

解码器同时维护低位和高位两个游标：

1. 从低位 bit stream 取出下一个 `low_i`；
2. 从高位 bitmap 找到下一个置位位置 `p_i`；
3. 计算 `high_i = p_i - i`；
4. 重建 `x_i = (high_i << l) | low_i`。

`EliasFanoStreamReader` 每次最多加载 8 字节高位 bitmap 到一个 64-bit
buffer。它使用 trailing-zero count 找到最低置位，随后执行
`buffer &= buffer - 1` 清除该位。若 buffer 为 0，则跳过当前块并加载后续
字节。低位也通过一个 64-bit buffer 按需补充。

顺序解码全部 `L` 个 ID 的复杂度为：

```text
O(L + high_bits_count / machine_word_bits)
```

因此摊销到每个 ID 为 `O(1)`；额外 reader 状态为 `O(1)`。不过这是顺序解码
保证，不代表任意位置都能常数时间随机访问。当前 reader 没有 select 索引，
定位第 `i` 个 ID 需要从流的当前位置继续扫描。

reader 还提供 `ReadBatch`，一次最多读取 8 个 ID。它先集中提取高位，再集中
提取低位，并支持与标量 `Read` 混用。当前 SINDI DMQ 查询重排改用
`EliasFanoSeekReader`，通过 high bitmap 的 bucket 直接定位查询 term 的
ordinal；`ReadBatch` 仍用于独立的顺序解码能力。

#### 与定宽 packed 解码的性能差异

定宽 packed 解码主要是加载、移位和 mask。EF 每个 ID 还需要维护两条 bit
stream、查找下一个高位 1、处理跨 64-bit 块的 refill，并执行
`p_i-i` 重建。高位 bitmap 中出现长零区间时，还会增加 refill 和分支。

逐项扫描会为每个候选文档恢复完整正排 ID 序列；当 `n_candidate=1000` 时，
这些额外操作会重复约 1000 次。当前 seek 路径改为按有序查询 term 扫描
64-bit high bitmap word，以 `popcount` 跳过整块，只在目标 high bucket 中
查找 low bits，并直接返回匹配的 value-code ordinal。Hybrid 的 layout 在
每个候选进入距离计算时计算一次，并被复用于确定 ID reader 和 value-code
偏移，不再为同一候选重复计算四次。

### 2. value 量化

对待编码文档 `d`，重新计算其非零 value 均值 `μ_d` 和残差
`r_{d,i}`。根据 term 对应码本的阈值，将每个残差编码为：

```text
q_{d,i} = min{k | r_{d,i} <= θ_k}，若不存在则 q_{d,i} = K - 1
```

这与实现中对 255 个阈值执行 `lower_bound` 一致。记该 code 对应的码本
代表值为：

```text
ĉ_{d,i} = c_{q_{d,i}}
```

DMQ 再为整篇文档计算一个缩放因子：

```text
             Σ_i r_{d,i}²
α_d = ---------------------------
             Σ_i ĉ_{d,i} r_{d,i}
```

当分母绝对值不超过 `10^{-12}` 时，`α_d` 设为 0。该校准使

```text
Σ_i r_{d,i}(α_d ĉ_{d,i}) = Σ_i r_{d,i}²
```

也就是让缩放后的量化残差在原残差方向上保持相同的投影。它不同于
以最小均方误差为目标的普通最小二乘缩放。

### 3. 单向量布局

一篇含 `L` 个非零项的文档编码为：

```text
+----------------------+----------------------+------------------+
| header (12 bytes)    | compact term IDs     | value codes      |
| len, μ_d, α_d        | packed or Elias–Fano | L bytes          |
+----------------------+----------------------+------------------+
```

因此单向量编码大小为：

```text
B_DMQ = 12 + min(B_packed, B_EF) + L bytes
```

空向量只保存 12 字节 header。

实际内存还包括每篇文档一个 offset、全局 term 映射和码本。每个码本包含
255 个 `float32` 阈值与 256 个 `float32` 代表值，共 2044 字节，不含
容器开销。

## 解码与重排打分

### 解码

根据 term 所属码本，value 的近似重建值为：

```text
x̂_{d,i} = μ_d + α_d c_{q_{d,i}}
```

compact ID 同时被映射回原始 term ID。

### 查询打分

SINDI 只支持内积度量。对于查询
`q=\{(t,q_t)\}`，DMQ 在查询准备阶段为每个已知查询 term 建立 256 项查找表：

```text
LUT[t, k] = q_t c_{t,k}
```

候选文档 `d` 的近似内积为：

```text
IP(q, x̂_d)
  = Σ_{t ∈ supp(q) ∩ supp(d)} q_t (μ_d + α_d c_{t,q_{d,t}})
  = μ_d Σ_t q_t + α_d Σ_t LUT[t, q_{d,t}]
```

VSAG 返回的距离为：

```text
distance(q, d) = 1 - IP(q, x̂_d)
```

实现直接在压缩数据上累加均值项和 LUT 项，不需要先完整解码候选向量。
未出现在 DMQ 训练词表中的查询 term 不参与打分；空交集的内积为 0，
距离为 1。

## 算法流程

```text
构建阶段
原始稀疏向量
  -> 计算逐文档均值与残差
  -> 按 term 频率分配独立/共享训练桶
  -> 按残差分布权重训练 256 项码本
  -> term ID 映射为 compact ID
  -> 每篇文档编码 ID、8-bit value code、均值和缩放因子

检索阶段
SINDI 倒排粗排
  -> 取得 n_candidate 个候选
  -> 为查询构造 DMQ code LUT
  -> 直接扫描候选的压缩正排编码并计算近似内积
  -> 按 1 - inner_product 排序并返回 top-k
```

## 使用方式

DMQ8 只能作为 SINDI 的重排正排格式使用：

```json
{
    "dtype": "sparse",
    "metric_type": "ip",
    "dim": 1024,
    "index_param": {
        "term_id_limit": 30000,
        "window_size": 50000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "rerank_type": "dmq8",
        "dmq_shared_codebook_threshold": 1024
    }
}
```

`use_quantization` 控制倒排 posting value 的格式，而 `rerank_type` 控制重排正排
副本的格式；两者彼此独立。常见组合是倒排使用 SQ8、正排使用 DMQ8。

## 取舍与限制

- **精度：** DMQ 是有损量化，最终召回变化取决于数据分布、候选数和剪枝
  参数，应在目标数据集上评估。
- **内存：** value 从 4 字节降为 1 字节，term ID 也被压缩；收益需扣除每向量
  12 字节 header、offset、term 映射和码本。
- **码本共享：** 增大 `dmq_shared_codebook_threshold` 会减少码本内存，但可能降低
  term 专属分布的拟合能力。
- **仅支持内积：** 当前 `SparseDmqQuantizer` 的度量固定为 inner product。
- **只支持离线建模：** SINDI 的 DMQ 码本在首次构建时固定，不支持增量
  `Add` 或 `UpdateVector`。
- **编码长度可变：** 不同文档的非零项数量和 ID 编码方式不同，因此 DMQ
  不支持通用的定长 batch encode/decode 接口。

## 相关实现

- `src/quantization/sparse_quantization/sparse_dmq_quantizer.{h,cpp}`：训练、编码、
  解码与距离计算；
- `src/datacell/sparse_dmq_datacell.{h,cpp}`：变长编码存储、offset 管理与序列化；
- `src/impl/elias_fano_stream.{h,cpp}`：有序 compact ID 的 Elias–Fano 编解码；
- [SINDI](../indexes/sindi.md)：索引流程、构建参数与检索参数。
