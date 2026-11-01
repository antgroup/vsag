# 聚类基准工具（`cluster_benchmark`）

`cluster_benchmark` 在同一批向量上比较 IVF 的两种训练方式 `kmeans` 与 `balanced_kmeans`：
报告簇大小的均匀程度、聚类损失、训练时间，以及训练器分配的峰值内存；加上 `--ivf` 时，
它会分别用两种训练方式构建 IVF 索引并执行检索。每次运行都有固定种子，结果可以复现。

均衡训练器的原理以及用本工具测得的结果，见
[IVF：均衡 k-means 训练](../indexes/ivf.md#均衡-k-means-训练)。

## 构建

```bash
VSAG_ENABLE_TOOLS=ON make release
# 产物：./build-release/tools/cluster_benchmark/cluster_benchmark
```

## 聚类对比

```bash
# 均匀、偏斜、不能整除三种合成数据，两种训练方式
./build-release/tools/cluster_benchmark/cluster_benchmark --dataset suite \
    --count 100000 --dim 64 -k 256 --threads 32

# 分量相互重叠的偏斜混合
./build-release/tools/cluster_benchmark/cluster_benchmark --dataset skewed --spread 0.5 \
    --count 100000 --dim 64 -k 256 --threads 32

# 取 ann-benchmarks 文件的前 262144 条向量，按 IVF 的规模测试
./build-release/tools/cluster_benchmark/cluster_benchmark \
    --dataset sift-128-euclidean.hdf5 --count 262144 -k 4096 --threads 32
```

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--dataset` | `suite` | `uniform`、`skewed`、`indivisible`、`suite`（三者全部），或 `.hdf5`（ann-benchmarks 的 `train` 部分）、`.fvecs` 文件 |
| `--count` | `100000` | 向量数（合成数据），或从文件读取的最大条数 |
| `--dim` | `64` | 合成数据的维度 |
| `-k`、`--clusters` | `256` | 簇数（IVF 桶数） |
| `--iterations` | `25` | 两种训练方式的 Lloyd 迭代次数 |
| `--threads` | `16` | 共享线程池的线程数 |
| `--seed` | `2870` | 数据与两种训练方式的随机种子 |
| `--algorithms` | `kmeans,balanced_kmeans` | 要运行的训练方式 |
| `--candidates` | `16` | `balanced_kmeans`：每条向量记住的簇数 |
| `--init` | `random` | `balanced_kmeans`：初始化方式，`random` 或 `kmeans++` |
| `--epsilon` | `1e-4` | `balanced_kmeans`：拍卖容差，相对代价差 |
| `--work-budget` | `1` | `balanced_kmeans`：每次分配工作量上限的倍数；`inf` 永不切换到两级分解，`0` 总是切换 |
| `--spread` | `4` | 偏斜数据：混合分量中心所在立方体的半宽（噪声为 0.25） |
| `--duplicates` | `0.05` | 偏斜数据：重复 8 条热点向量之一的行所占比例 |
| `--json` | 无 | 同时把结果写入该文件 |

合成数据集分别为：`uniform`，在 `[0, 1)^dim` 中均匀分布；`skewed`，64 个按 Zipf 权重
（指数 1.2）混合的高斯分量，中心在 `[-spread, spread]^dim` 中均匀分布，外加重复的热点向量；
`indivisible`，向量数比 `k` 的整数倍多 `k / 2` 的均匀数据。

输出的每一行对应一种训练方式：

| 列 | 含义 |
|---|---|
| `min`、`max`、`mean`、`std`、`CV` | 簇大小及其变异系数（`std / mean`） |
| `loss (MSE)` | 每条向量到其中心的均方距离 |
| `time (s)` | 训练时间 |
| `peak MiB` | 训练器通过 VSAG 分配器分配的内存峰值，不含输入向量 |
| `bids/n`、`rescored` | `balanced_kmeans`：每条向量的拍卖出价次数，以及重新打分的向量数 |
| `path` | `balanced_kmeans`：`flat`，或工作量上限触发两级分解时为 `two-level` |

## IVF 对比

加上 `--ivf` 时，工具读取 ann-benchmarks 文件（`train`、`test` 与 `neighbors`），用公共 API
为每种训练方式各构建一个 IVF 索引，并在多个 `scan_buckets_count` 取值下检索：

```bash
./build-release/tools/cluster_benchmark/cluster_benchmark --ivf \
    --dataset sift-128-euclidean.hdf5 --metric l2 -k 4096 --scan 8,16,32,64 --threads 32
```

它报告构建时间、`GetStats()` 给出的桶大小、相对文件中真值的 recall@10、每秒查询数，以及
平均每次查询的距离计算次数（随被扫描桶中的向量数增长）。

在有多个 NUMA 节点的机器上，请把它限定在一个节点上运行
（`numactl --cpunodebind=0 --membind=0 ./build-release/...`）。否则两个索引的内存可能落在
不同节点上，二者 QPS 的差距会超出距离计算次数所能解释的范围。
