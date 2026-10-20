# 三数据集统一 resident CRUD 验收数据

这是修复前 Lite（a3a76a8）的测量E9，不是最新修复版本的数据，也不是旧Lite/Full数字的拼接。
修复后的数据另见[版本复验E10](../review-fixed-resident-20261020/README.md)；本组数字和原始归档保持不变。
SIFT、GIST、Cohere 各10000向量、1000查询行，FP32/FP16/RaBitQ8各3个新进程。
测量前已提交固定协议；所有27个进程均保留，未选择最快一次或失败后改参数重跑。

## 结论和边界

- 405000次真实向量修改、405000次交错查询、540次Save及540次Load完成；每轮持久化结果一致。
- 初始/最终共54000次FP64穷举真值对照（查询集合重复，非54000个独立查询）；每轮1000查询对保存前/加载后索引比较，共1080000次额外搜索，不计入API吞吐。
- 原始数据审计通过不等于质量达标。Recall下降如实保留；当前未收到最低召回/允许退化的数值阈值，不自定“全部达标”。
- 本组仅比较Lite内部的三种存储；没有新原生Full结果，不证明新Full包体、加载或内存收益。
- 外部串行、每进程单线程、固定轮数；实际进程时长见CPU表，不称小时级并发/耐久验收。

## 协议和输入

degree16、construction ef128、query ef512、k10；CPU0、OMP/OpenBLAS/MKL1。
每进程20轮×250循环：5000 Update、5000 Remove、5000 Add、15000混合query。
Update值为FP32(.875×current+.125×original donor)，Remove后检查live IDs，随后Add新值；
resident原对象始终保留，Load副本只用于一致性检查。trial间轮换模式顺序，数据集顺序固定。

1000查询行在混合阶段重复15遍，不是15000个独立query；它们跨改值状态，P99不是静态初始索引P99。
准备阶段SIFT/GIST工具的真值实现保持原样；本组质量判定独立使用mixed probe的当前状态FP64真值。
Cohere初始FP64归一化后存FP32；初始cosine与平方L2 Top10核对一致。
改值后不再次归一化，最终质量是当前向量的平方L2质量，不能写成持续余弦检索质量。

| 数据集 | 维度 | 向量数 | 查询行数 | 不同查询向量数 |
| --- | --- | --- | --- | --- |
| SIFT | 128 | 10000 | 1000 | 1000 |
| GIST | 960 | 10000 | 1000 | 998 |
| Cohere | 768 | 10000 | 1000 | 1000 |

输入来源：SIFT ANN-Benchmarks HDF5、GIST maknee/gist1m镜像fbin、
Cohere cohere_small_100k Parquet。归档包含来源manifest、完整输入hash与命令；不上传原始向量。
SIFT/GIST的base与旧10k数据逐字节一致，查询的前100行也一致；更大的查询集合不是算法优化收益。

## 查询与质量

下表各值为3个进程指标的中位数，非合并样本分位数。每进程query有15000个API样本，
Add/Remove/Update各5000样本；Save/Load各20样本，其P99实际上是该进程最大值，不能作高置信长尾承诺。
所有三次进程值、min/max、CPU明细和原始CSV在归档中，不只保留表中中位数。

| 数据集 | 存储 | P50（μs） | P99（μs） | 查询调用/s | 初始→最终 Recall@10 |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 168.985 | 202.422 | 5873.7 | 0.9971 → 0.9960 |
| SIFT | FP16 | 166.545 | 195.982 | 5966.8 | 0.9971 → 0.9959 |
| SIFT | RaBitQ8 | 198.982 | 231.719 | 5000.9 | 0.9907 → 0.9900 |
| GIST | FP32 | 493.855 | 647.511 | 2025.4 | 0.9646 → 0.9583 |
| GIST | FP16 | 383.664 | 508.145 | 2604.5 | 0.9645 → 0.9581 |
| GIST | RaBitQ8 | 441.859 | 525.773 | 2282.5 | 0.9607 → 0.9648 |
| Cohere | FP32 | 389.155 | 507.363 | 2534.3 | 0.9865 → 0.9775 |
| Cohere | FP16 | 333.340 | 398.844 | 2981.0 | 0.9862 → 0.9776 |
| Cohere | RaBitQ8 | 411.492 | 457.279 | 2428.3 | 0.9831 → 0.9792 |

| 数据集 | RaBitQ P50相对FP32 | RaBitQ P50相对FP16 |
| --- | --- | --- |
| SIFT | +17.75% | +19.48% |
| GIST | -10.53% | +15.17% |
| Cohere | +5.74% | +23.45% |

正值表示RaBitQ更慢，负值表示更快。还需同时看质量、索引大小和内存，不能只按一项选“最优”。

## Add 单次调用

| 数据集 | 存储 | P50（μs） | P99（μs） | 调用/s | 累计user CPU（ms） | 累计system CPU（ms） |
| --- | --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 68.234 | 84.822 | 14780.7 | 337.708 | 1.721 |
| SIFT | FP16 | 66.745 | 82.233 | 15066.3 | 331.381 | 1.181 |
| SIFT | RaBitQ8 | 100.991 | 118.040 | 9986.8 | 500.659 | 1.467 |
| GIST | FP32 | 185.083 | 248.298 | 5396.3 | 927.292 | 2.431 |
| GIST | FP16 | 153.506 | 195.033 | 6512.9 | 768.700 | 1.513 |
| GIST | RaBitQ8 | 284.974 | 322.760 | 3511.5 | 1425.615 | 1.058 |
| Cohere | FP32 | 148.588 | 183.853 | 6710.4 | 744.459 | 1.955 |
| Cohere | FP16 | 139.797 | 164.065 | 7169.6 | 697.620 | 1.747 |
| Cohere | RaBitQ8 | 255.827 | 283.944 | 3912.3 | 1277.359 | 2.034 |

## Remove 单次调用

| 数据集 | 存储 | P50（μs） | P99（μs） | 调用/s | 累计user CPU（ms） | 累计system CPU（ms） |
| --- | --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 11.899 | 18.148 | 81085.2 | 62.553 | 0.294 |
| SIFT | FP16 | 11.949 | 18.378 | 80361.3 | 62.976 | 0.225 |
| SIFT | RaBitQ8 | 21.558 | 32.987 | 46302.3 | 108.977 | 0.316 |
| GIST | FP32 | 18.808 | 47.615 | 48209.4 | 104.795 | 0.270 |
| GIST | FP16 | 14.769 | 37.847 | 60507.4 | 83.752 | 0.117 |
| GIST | RaBitQ8 | 33.487 | 97.871 | 25861.8 | 194.576 | 0.104 |
| Cohere | FP32 | 17.407 | 38.097 | 52495.1 | 96.238 | 0.244 |
| Cohere | FP16 | 14.877 | 30.278 | 61513.4 | 82.379 | 0.164 |
| Cohere | RaBitQ8 | 38.477 | 95.521 | 23499.6 | 213.662 | 0.317 |

## Update 单次调用

| 数据集 | 存储 | P50（μs） | P99（μs） | 调用/s | 累计user CPU（ms） | 累计system CPU（ms） |
| --- | --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 73.542 | 100.501 | 13554.9 | 368.170 | 1.814 |
| SIFT | FP16 | 72.783 | 100.800 | 13622.3 | 365.241 | 1.419 |
| SIFT | RaBitQ8 | 106.620 | 172.964 | 9120.3 | 548.273 | 1.602 |
| GIST | FP32 | 219.220 | 404.963 | 4402.5 | 1136.042 | 2.539 |
| GIST | FP16 | 178.844 | 353.198 | 5341.1 | 936.857 | 2.011 |
| GIST | RaBitQ8 | 325.981 | 1167.964 | 2575.2 | 1939.204 | 3.966 |
| Cohere | FP32 | 166.595 | 361.436 | 5683.0 | 878.489 | 2.180 |
| Cohere | FP16 | 149.487 | 323.091 | 6355.4 | 786.921 | 1.823 |
| Cohere | RaBitQ8 | 273.083 | 1080.401 | 3102.8 | 1610.159 | 2.879 |

吞吐为5000次API调用数/累计API wall时间，不包含查询、真值、持久化、验证与记账。
CPU总和是该操作5000次调用的累计值，不是单次调用耗时；getrusage采样有微秒分辨率及额外开销。

## 保存、加载、索引文件和构建

warm uncontrolled-cache的同进程Save/Load，不是冷启动、drop-cache或fsync耐久IO。
构建每进程只有一个样本，直接报告wall时间，不赋予其有统计意义的P99。

### Save

| 数据集 | 存储 | P50（ms） | P99（ms） | 调用/s |
| --- | --- | --- | --- | --- |
| SIFT | FP32 | 3.671 | 3.905 | 271.5 |
| SIFT | FP16 | 3.118 | 3.267 | 318.6 |
| SIFT | RaBitQ8 | 4.203 | 4.777 | 240.8 |
| GIST | FP32 | 16.620 | 17.244 | 60.5 |
| GIST | FP16 | 10.136 | 10.589 | 99.1 |
| GIST | RaBitQ8 | 7.021 | 7.930 | 142.3 |
| Cohere | FP32 | 13.835 | 14.681 | 71.5 |
| Cohere | FP16 | 8.606 | 9.531 | 115.1 |
| Cohere | RaBitQ8 | 6.238 | 6.963 | 162.8 |

### Load

| 数据集 | 存储 | P50（ms） | P99（ms） | 调用/s |
| --- | --- | --- | --- | --- |
| SIFT | FP32 | 2.683 | 4.149 | 358.2 |
| SIFT | FP16 | 2.596 | 3.224 | 381.8 |
| SIFT | RaBitQ8 | 3.435 | 3.864 | 288.4 |
| GIST | FP32 | 18.289 | 19.181 | 54.6 |
| GIST | FP16 | 6.840 | 10.628 | 139.9 |
| GIST | RaBitQ8 | 5.657 | 5.962 | 177.8 |
| Cohere | FP32 | 8.218 | 14.300 | 113.1 |
| Cohere | FP16 | 6.138 | 9.409 | 156.1 |
| Cohere | RaBitQ8 | 4.759 | 5.492 | 207.0 |

| 数据集 | 存储 | 最终索引（MiB） | Build wall（s） |
| --- | --- | --- | --- |
| SIFT | FP32 | 6.256 | 0.539 |
| SIFT | FP16 | 3.815 | 0.532 |
| SIFT | RaBitQ8 | 2.824 | 0.602 |
| GIST | FP32 | 37.994 | 1.447 |
| GIST | FP16 | 19.684 | 1.333 |
| GIST | RaBitQ8 | 10.762 | 1.591 |
| Cohere | FP32 | 30.670 | 1.234 |
| Cohere | FP16 | 16.022 | 1.216 |
| Cohere | RaBitQ8 | 8.930 | 1.490 |

## 进程CPU和内存

这里的进程wall/user/system包含真值、Save/Load比对与记账，不能用来代替上面的API延迟。
CPU使用率按(user+system)/wall计算，100%表示约占满一个逻辑核，不是整台多核机器100%。
未采集硬件cycles、cache misses或能耗计数，不能把CPU时间直接解释为已证明的微架构热点。

| 数据集 | 存储 | wall（s） | user（s） | system（s） | 单核CPU（%） |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 16.94 | 16.89 | 0.05 | 100.0 |
| SIFT | FP16 | 16.86 | 16.81 | 0.04 | 100.0 |
| SIFT | RaBitQ8 | 19.20 | 19.15 | 0.04 | 100.0 |
| GIST | FP32 | 98.11 | 97.55 | 0.57 | 100.0 |
| GIST | FP16 | 89.53 | 89.30 | 0.23 | 100.0 |
| GIST | RaBitQ8 | 94.22 | 94.07 | 0.13 | 100.0 |
| Cohere | FP32 | 81.12 | 80.74 | 0.34 | 100.0 |
| Cohere | FP16 | 73.65 | 73.43 | 0.21 | 100.0 |
| Cohere | RaBitQ8 | 76.42 | 76.29 | 0.11 | 100.0 |

| 数据集 | 存储 | 进程峰值（MiB） | 初始RSS（MiB） | 最终RSS（MiB） | RSS变化（KiB） |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 39.922 | 27.234 | 39.922 | 12992 |
| SIFT | FP16 | 32.547 | 24.758 | 32.547 | 7976 |
| SIFT | RaBitQ8 | 30.355 | 24.574 | 30.355 | 5920 |
| GIST | FP32 | 165.801 | 125.621 | 129.195 | 3652 |
| GIST | FP16 | 146.793 | 107.293 | 146.793 | 40448 |
| GIST | RaBitQ8 | 144.477 | 99.203 | 112.652 | 13836 |
| Cohere | FP32 | 135.633 | 102.930 | 135.633 | 33476 |
| Cohere | FP16 | 120.820 | 88.281 | 120.820 | 33320 |
| Cohere | RaBitQ8 | 117.031 | 82.027 | 93.625 | 11876 |

RSS取持久化前的resident观测点；峰值包含输入、FP64真值、原始向量副本和加载副本，
不是索引自身的精确占用。增长可能包含分配器保留和缓存，本组不能单独证明泄漏、零碎片或长期稳定。
表中最终RSS是第20轮Save前的观测值，不是进程退出时的RSS。

| 数据集 | 存储 | minor faults | major faults | voluntary切换 | involuntary切换 |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 12761 | 0 | 0 | 61 |
| SIFT | FP16 | 10886 | 0 | 0 | 59 |
| SIFT | RaBitQ8 | 10311 | 0 | 0 | 71 |
| GIST | FP32 | 244567 | 0 | 1 | 330 |
| GIST | FP16 | 61587 | 0 | 0 | 277 |
| GIST | RaBitQ8 | 52823 | 0 | 0 | 297 |
| Cohere | FP32 | 61315 | 0 | 0 | 308 |
| Cohere | FP16 | 49986 | 0 | 0 | 272 |
| Cohere | RaBitQ8 | 43029 | 0 | 0 | 291 |

## 原始结果与复验

[raw.tar.gz](raw.tar.gz) 包含冻结协议、构建记录、输入manifest/hash、全部27进程结果、
命令、三次进程指标、聚合JSON和三套已独立校验的内部归档；[archive.json](archive.json)给出hash/体积。
只打包结果与必要源码记录，不打包数据集、临时快照、库或构建目录；服务器原文件仍保留。

在安装好benchmark依赖的仓库根目录，用现有安全解包器解开外层，再依次解开各数据集内部归档：

```bash
BENCH_EVIDENCE=/absolute/path/to/new-evidence-directory
test ! -e "$BENCH_EVIDENCE"
PYTHONPATH=lite/benchmark python3 -c \
  'from pathlib import Path; from pack_mixed_cpu_acceptance import unpack; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
  lite/benchmark/results/resident-three-datasets-20261019/raw.tar.gz "$BENCH_EVIDENCE"
for BENCH_DATASET in SIFT GIST Cohere; do
  PYTHONPATH=lite/benchmark python3 -c \
    'from pathlib import Path; from pack_mixed_cpu_acceptance import unpack; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
    "$BENCH_EVIDENCE/$BENCH_DATASET/raw.tar.gz" "$BENCH_EVIDENCE/$BENCH_DATASET/raw"
  python3 lite/benchmark/verify_mixed_cpu_acceptance.py "$BENCH_EVIDENCE/$BENCH_DATASET/raw"
done
```

外层manifest中的每个文件也须与SHA256对应。重新测量使用[固定协议](../../ACCEPTANCE.md#current-version-resident-protocol)
和[当前版本复现入口](../../README.md#current-version-graph-crud-and-archive-round-trip)，
每次只传一个数据集且保留3个trial，避免所有数据集临时快照同时占满RAM文件系统。

## 尚未完成的项目

当前Full公平对照、完整部署包重测、100k统一矩阵、小时级读写/内存耐久均不在本组内。
动态质量下降仍需按确定的质量标准判断和处理。协议与原始数据审计完成，不代表项目全部验收完成。
