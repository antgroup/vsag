# 随机投影树分区

`vsag::rpt_partition` 使用随机投影树（Random Projection Tree，RPT）将一组稠密向量划分为
**大小接近均匀的分区**。它是一个独立的工具函数：不会构建索引，也不会持有输入数据。典型用途包括
在构建分片索引前对大规模向量集合进行分片、均衡并行构建任务的负载，以及为下游算法生成保持局部性的分桶。

与随机分桶或按单一维度排序切分相比，RPT 能在保证分区大小几乎一致的同时，让邻近向量更多地落入同一分区。

## 距离空间

RPT 的局部性保持基于欧氏空间。函数使用随机方向与向量的内积计算投影值，不会自动归一化输入向量。
如果下游使用余弦相似度，应在调用 `rpt_partition` 前先对每个输入向量做 L2 归一化。对于单位长度向量，
余弦相似度和欧氏距离具有相同的邻近关系排序。

## API 概览

```cpp
struct RPTPartitionParams {
    uint64_t bucket_size{1000};  // 目标桶大小 L
    uint64_t max_depth{64};      // 树深度上限
    uint64_t seed{0};            // 随机种子
};

tl::expected<std::vector<std::vector<int64_t>>, Error>
rpt_partition(uint64_t d, uint64_t n, const float* x, const RPTPartitionParams& params);
```

声明位于
[`include/vsag/utils.h`](https://github.com/antgroup/vsag/blob/main/include/vsag/utils.h)。

| 参数 | 说明 |
| --- | --- |
| `d` | 向量维度，必须为正。 |
| `n` | 向量数量。`n == 0` 时返回空结果。 |
| `x` | 行优先存放的 `n * d` 个浮点数。由调用方持有，函数不会修改也不会保留该指针。 |
| `params.bucket_size` | 目标桶大小 `L`，每个分区最多包含 `L` 个向量。必须为正。 |
| `params.max_depth` | 递归深度的安全上限。为 `0` 时不执行投影切分，按输入 ID 顺序返回大小均衡的连续区间，不保证局部性；达到正数深度上限后，剩余向量按当前排列顺序切分为大小尽量均匀的块。 |
| `params.seed` | 随机种子。相同输入、参数和种子总是得到相同的分区结果。节点方向由种子和节点位置决定；如需独立的分区结果，应使用不同的种子。 |

返回值为每个分区一个 `std::vector<int64_t>`，存放该分区中向量在 `x` 中的行下标。

如果需要将返回分区所占的内存纳入 VSAG allocator 的统计，可以使用带有 `Allocator*` 参数的五参数
重载。该重载返回 `RPTPartitionResult`，外层和内层向量都使用传入的 allocator；在结果销毁前，调用方
必须保证该 allocator 一直有效。上面的四参数重载仍然可用，并返回标准库 vector。

## 尺寸保证

当 `n > 0` 时，给定 `n` 个向量和桶大小 `L`，令 `K = ceil(n / L)`，输出始终满足：

- 恰好产生 `K` 个分区；
- 每个分区包含 `floor(n / K)` 或 `ceil(n / K)` 个向量。

前两项保证由构造过程直接给出，不依赖数据分布，也不依赖随机种子。由此可得每个分区最多
包含 `L` 个向量；当 `n >= L` 时，还可得每个分区至少包含 `floor(L / 2)` 个向量，且最大分区
不超过最小分区的两倍。

## 工作原理

在每个节点，本实现先沿随机投影方向计算投影值，再为该节点指定必须产出的叶子数
`k = ceil(n_node / L)`。记 `n_node = q * k + r`，切分时让左子树获得 `ceil(k / 2)` 个叶子和
`q * ceil(k / 2) + min(r, ceil(k / 2))` 个向量。因此，切分排名由目标叶子数量决定，通常
不是投影值的中位数，也不会额外加入随机阈值偏移。切分位置通过对投影值执行
`std::nth_element` 得到；投影值相同时以向量 id 决定次序，因此重复向量与常量投影无需特殊
处理。投影值为 `NaN` 的向量排在最后。

上面的尺寸保证来自这一构造过程。局部性表现取决于随机投影方向和输入数据，并不声称具备其他
使用随机投影阈值扰动的 RPT 变体所证明的理论保证。

每个节点的投影方向由「用户种子 + 节点位置」的哈希值作为种子生成，因此输出与遍历顺序无关。
因此，对于给定的种子和树形结构，节点方向是确定的。仅改变输入数量或桶大小不会得到独立的
随机分区；如需独立的分区结果，应使用不同的种子。

当 `max_depth == 0` 时不会执行投影切分，而是将初始 ID 顺序划分为大小均衡的连续区间。尺寸保证
仍然成立，但不保证局部性。

除输入外的额外内存开销为一个 id 排列数组（`n * 4` 字节）、一个投影值缓冲（`n * 4` 字节）和一个
方向向量（`d * 4` 字节），不会复制向量数据。

## 示例

```cpp
#include <vsag/vsag.h>

vsag::RPTPartitionParams params;
params.bucket_size = 1000;
params.seed = 42;

auto result = vsag::rpt_partition(dim, num_vectors, datas.data(), params);
if (not result.has_value()) {
    std::cerr << result.error().message << std::endl;
    return -1;
}
for (const auto& partition : result.value()) {
    // partition 中存放一个桶内向量的行下标
}
```

完整程序见
[`examples/cpp/601_utils_rpt_partition.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/601_utils_rpt_partition.cpp)。

## 错误

| 条件 | 结果 |
| --- | --- |
| `d == 0` | `INVALID_ARGUMENT` |
| `params.bucket_size == 0` | `INVALID_ARGUMENT` |
| `x == nullptr` 且 `n > 0` | `INVALID_ARGUMENT` |
| `n` 大于 `2^32 - 1` | `INVALID_ARGUMENT` |

## 基准测试输入

`rpt_bench` 的 `--bucket_sizes` 参数接受逗号分隔的十进制整数列表，取值范围为 `[1, UINT64_MAX]`，
允许每个数字前后有空白。空项或格式错误的项会在加载数据集前被拒绝，错误信息会指出 `--bucket_sizes`
和具体的错误项。
