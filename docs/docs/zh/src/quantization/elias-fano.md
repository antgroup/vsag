# Elias–Fano 有序整数压缩

本文描述 VSAG 中 `EliasFanoStream` 的实际存储格式、编码和解码算法，以及它在
SINDI DMQ 正排 term ID 压缩中的使用方式。

Elias–Fano（EF）适合压缩取值范围已知的单调整数序列。VSAG 当前实现支持
**单调不减**序列，因此允许重复值；它提供顺序流式解码，不提供带 select
索引的随机访问。

## 适用条件与符号

输入序列记为：

$$
0 \le x_0 \le x_1 \le \cdots \le x_{N-1} < U
$$

其中：

- $N$：序列元素个数，对应一篇文档的非零 term 数；
- $U$：值域上界，不包含在有效取值中；DMQ 中等于 compact term 词表大小；
- $x_i$：第 $i$ 个 compact term ID。

当 $N=0$ 时，编码 payload 为 0 字节。当 $N>0$ 时，$U$ 必须大于 0。

## 1. Layout 计算

EF 把每个整数拆成高位和低位。低位宽度为：

$$
l =
\begin{cases}
\left\lfloor \log_2 \left(\left\lfloor U/N \right\rfloor\right) \right\rfloor,
    & \left\lfloor U/N \right\rfloor > 1, \\
0,  & \left\lfloor U/N \right\rfloor \le 1.
\end{cases}
$$

这里内层的 $\lfloor U/N \rfloor$ 表示 VSAG 使用整数除法计算商，再对商取
向下整的 $\log_2$。

低位区域包含 $N$ 个 $l$-bit 整数：

$$
B_{\mathrm{low}} = \left\lceil \frac{Nl}{8} \right\rceil
$$

高位 bitmap 的有效位数和存储字节数为：

$$
H = \left\lfloor \frac{U-1}{2^l} \right\rfloor + N + 1
$$

$$
B_{\mathrm{high}} = \left\lceil \frac{H}{8} \right\rceil
$$

完整 payload 大小为：

$$
B_{\mathrm{EF}} = B_{\mathrm{low}} + B_{\mathrm{high}}
$$

`EliasFanoStreamLayout` 保存上述四个结果。调用方可以只计算一次 layout，并在
大小计算、编码或 reader 构造时复用。

## 2. 编码格式

### 2.1 拆分高低位

对每个 $x_i$：

$$
\operatorname{low}_i = x_i \bmod 2^l
$$

$$
\operatorname{high}_i = \left\lfloor \frac{x_i}{2^l} \right\rfloor
$$

当 $l=0$ 时不存在低位区域，$\operatorname{low}_i=0$，所有信息都在高位
bitmap 中。

### 2.2 低位区域

$\operatorname{low}_i$ 按序紧密排列，从每个字节的最低位开始写入：

$$
\operatorname{bitOffset}(\operatorname{low}_i) = il
$$

一个整数可以跨越字节边界。区域末尾不足一个字节的位保持为 0。

### 2.3 高位区域

高位使用 unary/select 形式编码。第 $i$ 个元素对应的置位位置为：

$$
p_i = \operatorname{high}_i + i
$$

因为 $\operatorname{high}_i$ 单调不减，$p_i$ 严格递增，所以每个元素都对应
唯一的 1 bit。即使 $x_i=x_{i-1}$，额外的 $+i$ 仍能区分两个元素。

最终内存布局为：

```text
+-----------------------------+----------------------------------+
| low bits                    | high bitmap                      |
| N 个 l-bit 整数，紧密排列   | 在 high_i + i 处设置第 i 个 1   |
+-----------------------------+----------------------------------+
```

payload 本身不保存 $N$、$U$、$l$ 或编码类型。调用方必须从外部提供 $N$ 和
$U$，或提供预先计算好的 layout。

## 3. 完整编码示例

设：

$$
U=16,\qquad N=4,\qquad \boldsymbol{x}=[3,5,8,13]
$$

低位宽度：

$$
l=\left\lfloor\log_2(16/4)\right\rfloor=2
$$

拆分结果：

| $i$ | $x_i$ | $\operatorname{low}_i$ | $\operatorname{high}_i$ | $p_i=\operatorname{high}_i+i$ |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 3 | 0 | 0 |
| 1 | 5 | 1 | 1 | 2 |
| 2 | 8 | 0 | 2 | 4 |
| 3 | 13 | 1 | 3 | 6 |

低位按 LSB-first 紧密排列后是 `0x47`，高位 bitmap 在第 0、2、4、6 位
置 1，得到 `0x55`：

```text
payload = [0x47, 0x55]
           low    high
```

总大小为 2 字节。若用 `uint32_t` 直接保存则需要 16 字节；若按
$\lceil\log_2 U\rceil=4$ 位定宽打包，也需要 2 字节。因此 DMQ Hybrid 策略在这个
例子中会选择 packed，因为大小相等时不选择 EF。

## 4. 编码算法

`EliasFanoStream::Encode` 的流程为：

```text
layout = GetLayout(N, U)
清零 layout.SizeInBytes() 字节

for i in [0, N):
    检查 x_i < U
    检查 i == 0 或 x_(i-1) <= x_i
    写入 low_i
    high_bitmap[high_i + i] = 1
```

当前低位写入函数逐 bit 处理，编码时间为 $O(Nl)$。由于输入是
`uint32_t`，$l\le31$，也可以把它视为相对于元素个数的 $O(N)$。

编码器首先清零整个输出，因此还需要与 payload 字节数成正比的初始化成本。
除输出缓冲区外，编码过程只使用 $O(1)$ 额外空间。

## 5. 标量流式解码

`EliasFanoStreamReader` 分别维护低位和高位状态：

- `low_cursor_`、`low_buffer_`、`low_available_bits_`；
- `high_cursor_`、`high_buffer_`、`high_available_bits_`；
- `high_base_position_`：当前 64-bit 高位块在完整 bitmap 中的起始位置；
- `index_`：下一个要解码的元素序号。

一次 `Read()` 的实际顺序如下：

### 5.1 找到下一个高位

如果 `high_buffer_` 为 0，reader 从高位区域加载最多 8 字节。字节通过显式
移位组装为 `uint64_t`，因此不依赖主机端序。

若加载出的 64-bit 块仍为 0，则增加 `high_base_position_` 并继续加载，
直到找到包含置位的块。最低置位的块内偏移为：

$$
\delta_i=\operatorname{ctz}(\texttt{high\_buffer})
$$

清除最低置位后，恢复其绝对位置与高位：

$$
p_i=\texttt{high\_base\_position}+\delta_i
$$

$$
\operatorname{high}_i=p_i-i
$$

对应的位操作为：

```text
high_buffer &= high_buffer - 1
```

`count_trailing_zeros` 在 GCC/Clang 下映射到 `__builtin_ctzll`，通常会生成
硬件 bit-scan/tzcnt 指令。`buffer &= buffer-1` 清除刚消费的最低置位。

### 5.2 读取低位

当 low buffer 中不足 $l$ 位时，reader 逐字节补充，直到可以返回一个完整
的 $\operatorname{low}_i$。数学上等价于：

$$
\operatorname{low}_i=\texttt{low\_buffer}\bmod 2^l
$$

然后通过右移 $l$ 位消费它：

```text
low_buffer >>= l
```

当 $l=0$ 时直接返回 0，不访问低位区域。

### 5.3 重建整数

$$
x_i=\operatorname{high}_i\,2^l+\operatorname{low}_i
$$

随后 `index_` 加 1。reader 到达 $N$ 后再次调用 `Read()` 会报错。

## 6. 批量解码

`ReadBatch(values, max_count)` 最多返回
`EliasFanoStreamReader::MAX_BATCH_SIZE`，当前值为 8。它分两轮处理：

1. 连续提取本批元素的所有 $\operatorname{high}_i$；
2. 恢复批次起始 `index_`，连续读取所有 $\operatorname{low}_i$ 并完成重建。

这种组织减少了高低位状态交替带来的依赖，并为后续展开或 SIMD 化提供了
入口。批量和标量调用可以混用，最后不足 8 个元素时返回实际数量。

`ReadBatch` 已有正确性测试，但当前 SINDI DMQ 查询重排使用
`EliasFanoSeekReader` 直接定位查询 term 的 ordinal，并不使用这个批量接口。
顺序 reader 仍用于完整向量解码和向量间 merge。

## 7. 复杂度

### 空间

当前实现的空间界为：

$$
Nl+O(N)\ \text{bits}
$$

当序列严格递增、因而 $N\le U$ 时，这就是经典的

$$
N\left\lfloor\log_2(U/N)\right\rfloor+O(N)
$$

Elias–Fano 空间界。使用 $l=0$ 时，该表达式也能覆盖允许重复值导致 $N>U$
的情况。

对应当前实现的精确字节数：

$$
B_{\mathrm{EF}}
=
\left\lceil\frac{Nl}{8}\right\rceil
+
\left\lceil
\frac{
    \left\lfloor (U-1)/2^l \right\rfloor+N+1
}{8}
\right\rceil
$$

### 时间

| 操作 | 时间复杂度 | 额外空间 |
| --- | --- | --- |
| layout 计算 | $O(\log U)$，最多 32-bit 整数宽度 | $O(1)$ |
| 编码 | 当前实现 $O(Nl)$ | $O(1)$ |
| 顺序解码全部 ID | $O(N+H/w)$，$w$ 为机器字位数 | $O(1)$ |
| 摊销单个顺序解码 | $O(1)$ | $O(1)$ |
| 有序查询 term 的 seek | $O(H_s/w+\sum_q\log(k_q+1)+M)$ | $O(1)$ |
| 随机读取第 $i$ 个 ID | 当前未提供 | — |

高位 bitmap 中的每个机器字最多加载一次，每个 1 bit 也只消费一次，因此完整
顺序扫描是线性的。若相邻高位相差很大，reader 可能连续跳过多个全零机器字；
该成本已经包含在 $H/w$ 项中。

## 8. SINDI DMQ Hybrid 策略

DMQ 同时支持定宽 packed 与 EF。设 compact term 词表大小为 $V$，定宽位数和
payload 大小为：

$$
b=\max\left(1,\left\lceil\log_2 V\right\rceil\right)
$$

$$
B_{\mathrm{packed}}=\left\lceil\frac{Nb}{8}\right\rceil
$$

每篇文档编码时比较：

$$
\text{encoding} =
\begin{cases}
\text{Elias--Fano}, & B_{\mathrm{EF}} < B_{\mathrm{packed}}, \\
\text{packed},      & B_{\mathrm{EF}} \ge B_{\mathrm{packed}}.
\end{cases}
$$

格式没有额外 tag。解码时使用 header 中的 $N$ 和模型中的 $V$ 重做相同的
确定性比较，即可推导格式和后续 value codes 的偏移。空向量没有 ID payload。

候选进入距离计算时只计算一次 Hybrid layout，结果同时用于：

- 选择 `EliasFanoStreamReader` 或 `PackedReader`；
- 得到 ID payload 字节数；
- 定位 DMQ value codes。

这避免了早期实现中同一候选重复计算四次 layout 的开销。

其中 $H_s$ 是扫描到最大查询 high 所经过的 bitmap 位数，$k_q$ 是查询 term
所在 high bucket 的元素数，$M$ 是实际匹配数。

## 9. 查询驱动 seek 与性能

定宽 packed reader 的核心操作是加载、移位和 mask。EF reader 每个 ID 还要：

- 维护两条独立 bit stream；
- 查找高位 bitmap 的下一个 1；
- 清除已消费的置位；
- 处理跨 64-bit 块的 refill 和全零块；
- 执行 $p_i-i$ 和高低位合并；
- 承担更多数据依赖与条件分支。

若 SINDI reorder 对每个候选扫描完整 ID 序列，例如
$n_{\mathrm{candidate}}=1000$，这些额外操作会重复约 1000 次。当前查询打分
因此改用 `EliasFanoSeekReader`：

1. 查询 compact ID 按升序处理；
2. 将 high bitmap 中的 0 作为 high bucket 分隔符；
3. 使用 64-bit `popcount` 一次跳过整个 bitmap word；
4. 通过已消费的 1 数得到 bucket 对应的 ordinal 范围；
5. 在该范围内二分查找 low bits；
6. 命中后直接访问同 ordinal 的 DMQ value code。

该路径不再逐项恢复候选的所有 ID，也不修改 EF payload 或索引格式。它是
scalar word-level 跳跃优化，不等同于已经启用 SIMD。进一步 SIMD 化仍需解决
变长高位 bitmap、跨块边界和查询匹配循环的数据依赖。

## 10. 校验与边界条件

实现会检查：

- 非空序列的 $U>0$；
- 输入、输出和 batch 指针在需要使用时非空；
- 每个值满足 $x_i<U$；
- 输入序列单调不减；
- 高位 bitmap 在读完 $N$ 个置位前没有耗尽；
- `Read()` 不超过序列结尾；
- `ReadBatch` 的请求数量不超过 8。

重复 ID 是合法输入；乱序 ID 和等于或超过 $U$ 的 ID 会被拒绝。

## 11. 相关实现与测试

- `src/impl/elias_fano_stream.h`：layout、编码器和 reader 接口；
- `src/impl/elias_fano_stream.cpp`：layout、编码、标量和批量解码；
- `src/impl/elias_fano_stream_test.cpp`：有序序列 round-trip、重复值、批量读取、
  跨空高位块、标量/批量混用和错误输入测试；
- `src/quantization/sparse_quantization/sparse_dmq_quantizer.cpp`：Hybrid
  packed/EF 选择和 SINDI DMQ 距离计算；
- [DMQ](dmq.md)：value 量化、正排布局和重排打分。
