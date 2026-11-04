# Hamming kernel 基准测试工具（`hamming_benchmark`）

`tools/hamming_benchmark` 包含两个面向[二进制向量](../guide/binary_vectors.md)的基准测试：

- `hamming_benchmark`（C++）：校验并测量当前 CPU 能运行的每一个 Hamming 距离 kernel。
- `bench_index.py`（Python）：在 ann-benchmarks 的 Hamming 数据集上测量二进制 `hgraph` 与
  `brute_force` 索引的召回率和单线程吞吐量；安装了 `faiss` 包时，同时给出 FAISS 二进制索引作为参照。

## 构建

使用项目 Makefile 时，通过 `VSAG_ENABLE_TOOLS=ON` 打开工具构建；`bench_index.py` 还需要 Python
绑定（`VSAG_ENABLE_PYBINDS=ON`，或已安装的 `pyvsag`）：

```bash
VSAG_ENABLE_TOOLS=ON make release
# 输出：./build-release/tools/hamming_benchmark/hamming_benchmark
```

直接调用 CMake 时，传入 `-DENABLE_TOOLS=ON`。

## Kernel 基准测试

```bash
./build-release/tools/hamming_benchmark/hamming_benchmark \
    --dims 64,256,1024 --vectors 2048 --min-time 0.3 --json kernels.json
```

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `--dims` | `64,128,256,512,1024,2048,4096` | 逗号分隔的维度，单位为 bit |
| `--vectors` | `2048` | 每轮使用的编码数（向下取整到 4 的倍数） |
| `--min-time` | `0.3` | 每项测量的最短时间（秒） |
| `--json` | 无 | 同时把结果写入该 JSON 文件 |

对每个维度，工具先校验每个 kernel 在逐个计算和扫描两种方式下都能复现通用实现对每个编码的距离；
出现不一致时打印出来，并以退出码 1 结束。随后对每个 kernel 报告三个入口每个距离的纳秒数：

| 列 | 入口 | 使用者 |
| --- | --- | --- |
| `single` | `HammingDistance`，每次调用计算一个编码 | 单个距离 |
| `batch4` | `HammingDistanceBatch4`，每次调用计算四个任意地址上的编码 | 图搜索 |
| `scan` | `HammingDistanceScan`，一次调用扫描所有连续存放的编码 | `brute_force` 扫描 |

`scan GB/s` 是扫描每秒读取的编码字节数；加速比是该 kernel 最快的入口相对通用单编码 kernel 的倍数。
每轮最多读取 1 MiB 编码，因此这些数字反映的是计算能力而不是内存带宽。请在空闲核上运行，例如
`taskset -c 3 ./hamming_benchmark`，并保持该物理核的另一个超线程空闲。

## 索引基准测试

```bash
pip install numpy h5py faiss-cpu   # faiss-cpu 可选
wget http://ann-benchmarks.com/sift-256-hamming.hdf5
taskset -c 0-15 python3 tools/hamming_benchmark/bench_index.py \
    --data sift-256-hamming.hdf5 --threads 16 --json sift256.json
```

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `--data` | 必填 | ann-benchmarks Hamming HDF5 文件（`train`、`test`，可选 `distances`） |
| `--base` | 全部 | 最多使用的底库向量数 |
| `--queries` | `1000` | 查询数 |
| `--k` | `10` | 每个查询返回的近邻数 |
| `--ef` | `10 20 40 80 160 320` | `hgraph` 的 `ef_search` 取值（也是 FAISS HNSW 的 `efSearch`） |
| `--max-degree` | `32` | `hgraph` 的 `max_degree`；FAISS HNSW 使用 `M = max_degree / 2` |
| `--ef-construction` | `200` | 两种图的构建搜索宽度 |
| `--threads` | `16` | 构建线程数；搜索始终单线程 |
| `--exact-limit` | `2000000` | 底库超过该数量时跳过精确索引 |
| `--exact-queries` | `200` | 精确索引使用的查询数 |
| `--repeat` | `3` | 每个设置把全部查询运行几遍，报告最好的一遍 |
| `--json` | 无 | 同时把结果写入该 JSON 文件 |

脚本用 `numpy.packbits` 按 MSB-first 打包数据集中的 0/1 元素。Recall@k 对并列距离友好：只要返回的
近邻距离不大于真实的第 k 个距离就算命中。第 k 个距离取自 `distances` 数据集（ann-benchmarks 存储的
是除以 bit 数之后的值，脚本会换算回来）；文件中没有该数据集时用暴力计算得到。FAISS
`IndexBinaryHNSW(M)` 在底层保留 `2 * M` 个邻居，对应 `hgraph` 的 `max_degree`。

请把进程绑定到同一 NUMA 节点上的空闲物理核，使构建线程及其分配的内存都留在该节点上。

## 参考结果

以下数据于 2026-11-01 测得：AMD EPYC 9A75 服务器（Zen 5，支持 AVX-512 VPOPCNTDQ，两个 NUMA 节点，
1.5 TB 内存），Linux 5.4，GCC 13.4 编译的 `make release` 构建，Python 3.12、NumPy 2.5 与
`faiss-cpu` 1.15.1。搜索在一个空闲核上运行，构建线程运行在同一 NUMA 节点的 16 个空闲核上。

### Kernel

每个距离的纳秒数；标量各列取该 kernel 三个入口中最快的一个。

| bit 数 | generic | popcnt | avx2 | avx512bw | vpopcntdq single | vpopcntdq batch4 | vpopcntdq scan | scan GB/s | 加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.29 | 0.66 | 0.74 | 0.40 | 1.81 | 0.67 | 0.05 | 153 | 43.9x |
| 100 | 7.73 | 1.96 | 2.05 | 1.25 | 2.01 | 0.97 | 0.51 | 25 | 15.1x |
| 128 | 3.94 | 0.85 | 0.86 | 0.60 | 2.01 | 0.85 | 0.09 | 173 | 42.5x |
| 256 | 5.80 | 1.22 | 1.10 | 1.27 | 2.21 | 1.03 | 0.19 | 170 | 30.9x |
| 512 | 9.98 | 2.06 | 1.40 | 1.03 | 2.21 | 0.84 | 0.44 | 145 | 22.6x |
| 768 | 14.00 | 2.90 | 1.74 | 1.83 | 1.96 | 1.35 | 0.70 | 138 | 20.1x |
| 1000 | 22.26 | 4.98 | 4.24 | 1.81 | 2.09 | 1.19 | 0.77 | 162 | 28.9x |
| 1024 | 17.73 | 3.74 | 2.23 | 1.49 | 1.82 | 1.37 | 0.74 | 173 | 24.0x |
| 2048 | 33.95 | 7.09 | 4.20 | 2.45 | 2.55 | 2.06 | 1.64 | 156 | 20.7x |
| 4096 | 72.99 | 14.21 | 8.16 | 5.21 | 5.28 | 4.60 | 3.53 | 145 | 20.7x |

在 AVX-512 VPOPCNTDQ 上，扫描每步处理八个编码且只做一次水平归约；当编码为 8、16 或 32 字节时，
一个 64 字节寄存器同时装入二到八个编码，所以 64、128 和 256 bit 的编码扫描最快。NEON 与 SVE kernel
在 QEMU 下做了逐 bit 校验（仅 NEON，以及 128 到 2048 bit 向量长度的 SVE），没有给出 ARM 上的耗时。

### 精确搜索

随机编码，top 10，每次一个查询；给出每扫描一个向量的纳秒数（越低越好）和每秒查询数。

| 向量数 | bit 数 | `brute_force` ns/向量 | `brute_force` QPS | `IndexBinaryFlat` ns/向量 | `IndexBinaryFlat` QPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 256 | 1.02 | 49,070 | 0.74 | 67,121 |
| 100,000 | 256 | 0.56 | 17,750 | 0.66 | 15,236 |
| 400,000 | 256 | 0.46 | 5,452 | 0.61 | 4,083 |
| 988,000 | 256 | 0.75 | 1,344 | 0.69 | 1,469 |
| 20,000 | 1024 | 1.90 | 26,321 | 2.51 | 19,917 |
| 100,000 | 1024 | 1.39 | 7,196 | 2.32 | 4,309 |
| 400,000 | 1024 | 2.49 | 1,003 | 2.78 | 898 |
| 988,000 | 1024 | 2.63 | 385 | 2.78 | 364 |

数据量小时主要是单次查询的固定开销，数据量最大时主要受内存带宽限制。

### sift-256-hamming

988,258 个 256 bit 向量，1,000 个查询，k = 10，`max_degree` 32，`ef_construction` 200。

| 索引 | ef_search | recall@10 | QPS（单线程） | 构建耗时（秒） |
| --- | ---: | ---: | ---: | ---: |
| vsag hgraph | 10 | 0.8838 | 33,118 | 16.2 |
| vsag hgraph | 20 | 0.9619 | 25,224 | 16.2 |
| vsag hgraph | 40 | 0.9914 | 15,935 | 16.2 |
| vsag hgraph | 80 | 0.9977 | 9,294 | 16.2 |
| vsag hgraph | 160 | 0.9993 | 5,227 | 16.2 |
| vsag hgraph | 320 | 0.9998 | 2,875 | 16.2 |
| vsag brute_force | - | 1.0000 | 1,533 | 0.2 |
| faiss IndexBinaryHNSW | 10 | 0.8735 | 57,808 | 10.2 |
| faiss IndexBinaryHNSW | 20 | 0.9501 | 41,033 | 10.2 |
| faiss IndexBinaryHNSW | 40 | 0.9888 | 26,269 | 10.2 |
| faiss IndexBinaryHNSW | 80 | 0.9974 | 14,835 | 10.2 |
| faiss IndexBinaryHNSW | 160 | 0.9995 | 7,898 | 10.2 |
| faiss IndexBinaryHNSW | 320 | 1.0000 | 4,167 | 10.2 |
| faiss IndexBinaryFlat | - | 1.0000 | 1,491 | 0.0 |

### word2bits-800-hamming

399,000 个 800 bit 向量，1,000 个查询，k = 10，`max_degree` 32，`ef_construction` 200。

| 索引 | ef_search | recall@10 | QPS（单线程） | 构建耗时（秒） |
| --- | ---: | ---: | ---: | ---: |
| vsag hgraph | 10 | 0.7885 | 28,275 | 6.8 |
| vsag hgraph | 20 | 0.8830 | 22,718 | 6.8 |
| vsag hgraph | 40 | 0.9404 | 14,932 | 6.8 |
| vsag hgraph | 80 | 0.9717 | 9,075 | 6.8 |
| vsag hgraph | 160 | 0.9856 | 5,308 | 6.8 |
| vsag hgraph | 320 | 0.9928 | 2,965 | 6.8 |
| vsag brute_force | - | 1.0000 | 1,134 | 0.1 |
| faiss IndexBinaryHNSW | 10 | 0.7775 | 43,734 | 5.4 |
| faiss IndexBinaryHNSW | 20 | 0.8720 | 27,714 | 5.4 |
| faiss IndexBinaryHNSW | 40 | 0.9332 | 17,036 | 5.4 |
| faiss IndexBinaryHNSW | 80 | 0.9649 | 9,911 | 5.4 |
| faiss IndexBinaryHNSW | 160 | 0.9829 | 5,510 | 5.4 |
| faiss IndexBinaryHNSW | 320 | 0.9893 | 2,965 | 5.4 |
| faiss IndexBinaryFlat | - | 1.0000 | 979 | 0.0 |

在相同的 `ef_search` 下，`hgraph` 找回的真实近邻在 word2bits-800-hamming 的所有设置上、以及
sift-256-hamming 的 `ef_search` 不超过 80 时都多于 FAISS HNSW；而 FAISS 在相同 `ef_search` 下回答一个
查询更快。请在相同召回率下比较两种图，并在目标硬件上实测后再下结论。
