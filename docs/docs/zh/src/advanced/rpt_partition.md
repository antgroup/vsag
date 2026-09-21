# 随机投影树分区

`vsag::rpt_partition` 使用随机投影树（Random Projection Tree，RPT）将一组稠密向量划分为
**大小接近均匀的分区**。它是一个独立的工具函数：不会构建索引，也不会持有输入数据。典型用途包括
在构建分片索引前对大规模向量集合进行分片、均衡并行构建任务的负载，以及为下游算法生成保持局部性的分桶。

与随机分桶或按单一维度排序切分相比，RPT 能在保证分区大小几乎一致的同时，让邻近向量更多地落入同一分区。

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
| `params.max_depth` | 递归深度的安全上限。达到上限后，剩余向量按 id 顺序等分切块，尺寸保证仍然成立。 |
| `params.seed` | 随机种子。相同输入与相同种子总是得到相同的分区结果。 |

返回值为每个分区一个 `std::vector<int64_t>`，存放该分区中向量在 `x` 中的行下标。

## 尺寸保证

给定 `n` 个向量和桶大小 `L`，令 `K = ceil(n / L)`，输出始终满足：

- 恰好产生 `K` 个分区；
- 每个分区最多包含 `L` 个向量；
- 当 `n >= L` 时，每个分区至少包含 `floor(L / 2)` 个向量，且最大分区不超过最小分区的两倍。

这些保证由构造过程直接给出，不依赖数据分布，也不依赖随机种子。每个分区的大小恒为
`floor(n / K)` 或 `ceil(n / K)`。

## 工作原理

朴素的 RPT 在每个节点按中位数切分，直到节点能放入一个桶为止。这样得到的叶子数是 2 的幂，比
`ceil(n / L)` 最多多出 45%，且桶的填充率只有约 64%。

本实现改为给每个节点指定它必须产出的叶子数 `k = ceil(n_node / L)`。记 `n_node = q * k + r`，
切分时让左子树获得 `ceil(k / 2)` 个叶子和 `q * ceil(k / 2) + min(r, ceil(k / 2))` 个向量。切分
位置通过对投影值执行 `std::nth_element` 得到；投影值相同时以向量 id 决定次序，因此重复向量与
常量投影无需特殊处理。投影值为 `NaN` 的向量排在最后。

每个节点的投影方向由「用户种子 + 节点位置」的哈希值作为种子生成，因此输出与遍历顺序无关。

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
