# 三数据集统一 resident CRUD 验收数据

这是评审修复后 Lite（cba3406）的独立复验 E10，不是旧 Lite/Full 数字的拼接，也不是小规模工具冒烟。
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
| SIFT | FP32 | 171.094 | 202.771 | 5806.6 | 0.9971 → 0.9960 |
| SIFT | FP16 | 166.704 | 196.113 | 5960.2 | 0.9971 → 0.9959 |
| SIFT | RaBitQ8 | 203.062 | 236.389 | 4900.4 | 0.9907 → 0.9900 |
| GIST | FP32 | 431.371 | 598.705 | 2305.7 | 0.9646 → 0.9583 |
| GIST | FP16 | 346.488 | 446.060 | 2897.5 | 0.9645 → 0.9581 |
| GIST | RaBitQ8 | 435.230 | 502.573 | 2326.0 | 0.9607 → 0.9648 |
| Cohere | FP32 | 382.025 | 505.465 | 2579.5 | 0.9865 → 0.9775 |
| Cohere | FP16 | 336.659 | 397.455 | 2951.6 | 0.9862 → 0.9776 |
| Cohere | RaBitQ8 | 424.622 | 474.208 | 2352.5 | 0.9831 → 0.9792 |

| 数据集 | RaBitQ P50相对FP32 | RaBitQ P50相对FP16 |
| --- | --- | --- |
| SIFT | +18.68% | +21.81% |
| GIST | +0.89% | +25.61% |
| Cohere | +11.15% | +26.13% |

正值表示RaBitQ更慢，负值表示更快。还需同时看质量、索引大小和内存，不能只按一项选“最优”。

## Add 单次调用

| 数据集 | 存储 | P50（μs） | P99（μs） | 调用/s | 累计user CPU（ms） | 累计system CPU（ms） |
| --- | --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 69.264 | 85.022 | 14564.9 | 343.927 | 1.106 |
| SIFT | FP16 | 66.964 | 82.053 | 15006.1 | 332.537 | 1.679 |
| SIFT | RaBitQ8 | 102.920 | 119.878 | 9795.0 | 508.432 | 1.417 |
| GIST | FP32 | 156.786 | 209.341 | 6362.2 | 786.944 | 1.500 |
| GIST | FP16 | 147.098 | 182.594 | 6841.4 | 730.956 | 1.626 |
| GIST | RaBitQ8 | 285.714 | 324.961 | 3506.2 | 1425.748 | 1.944 |
| Cohere | FP32 | 145.958 | 183.413 | 6820.7 | 733.121 | 1.118 |
| Cohere | FP16 | 142.317 | 166.635 | 7045.7 | 710.719 | 1.292 |
| Cohere | RaBitQ8 | 260.606 | 291.464 | 3839.3 | 1301.504 | 2.384 |

## Remove 单次调用

| 数据集 | 存储 | P50（μs） | P99（μs） | 调用/s | 累计user CPU（ms） | 累计system CPU（ms） |
| --- | --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 11.990 | 18.039 | 80590.2 | 63.256 | 0.234 |
| SIFT | FP16 | 11.639 | 17.669 | 82582.4 | 61.438 | 0.293 |
| SIFT | RaBitQ8 | 22.748 | 34.687 | 44019.5 | 114.446 | 0.287 |
| GIST | FP32 | 16.349 | 43.275 | 54845.8 | 92.329 | 0.213 |
| GIST | FP16 | 12.989 | 34.217 | 68381.9 | 74.282 | 0.159 |
| GIST | RaBitQ8 | 35.268 | 105.852 | 24431.4 | 205.591 | 0.348 |
| Cohere | FP32 | 17.358 | 38.317 | 52690.3 | 95.958 | 0.135 |
| Cohere | FP16 | 14.758 | 30.236 | 62206.9 | 81.490 | 0.142 |
| Cohere | RaBitQ8 | 41.106 | 103.050 | 21947.9 | 228.573 | 0.450 |

## Update 单次调用

| 数据集 | 存储 | P50（μs） | P99（μs） | 调用/s | 累计user CPU（ms） | 累计system CPU（ms） |
| --- | --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 74.704 | 101.491 | 13331.0 | 375.671 | 1.418 |
| SIFT | FP16 | 72.194 | 100.110 | 13739.5 | 363.231 | 1.716 |
| SIFT | RaBitQ8 | 109.250 | 182.043 | 8874.3 | 561.652 | 1.655 |
| GIST | FP32 | 194.293 | 383.716 | 4916.2 | 1017.991 | 1.645 |
| GIST | FP16 | 165.046 | 328.949 | 5786.3 | 863.815 | 1.981 |
| GIST | RaBitQ8 | 325.009 | 1231.007 | 2556.1 | 1952.014 | 6.798 |
| Cohere | FP32 | 163.404 | 358.987 | 5788.7 | 863.221 | 1.332 |
| Cohere | FP16 | 151.147 | 316.011 | 6303.8 | 794.534 | 1.503 |
| Cohere | RaBitQ8 | 279.854 | 1136.766 | 3018.9 | 1651.895 | 7.370 |

吞吐为5000次API调用数/累计API wall时间，不包含查询、真值、持久化、验证与记账。
CPU总和是该操作5000次调用的累计值，不是单次调用耗时；getrusage采样有微秒分辨率及额外开销。

## 保存、加载、索引文件和构建

warm uncontrolled-cache的同进程Save/Load，不是冷启动、drop-cache或fsync耐久IO。
构建每进程只有一个样本，直接报告wall时间，不赋予其有统计意义的P99。

### Save

| 数据集 | 存储 | P50（ms） | P99（ms） | 调用/s |
| --- | --- | --- | --- | --- |
| SIFT | FP32 | 3.792 | 3.859 | 263.2 |
| SIFT | FP16 | 3.141 | 3.221 | 316.8 |
| SIFT | RaBitQ8 | 4.229 | 4.859 | 241.2 |
| GIST | FP32 | 16.009 | 16.717 | 62.2 |
| GIST | FP16 | 9.579 | 10.250 | 104.9 |
| GIST | RaBitQ8 | 6.543 | 7.435 | 154.9 |
| Cohere | FP32 | 13.392 | 14.032 | 74.1 |
| Cohere | FP16 | 8.659 | 9.043 | 114.5 |
| Cohere | RaBitQ8 | 6.063 | 6.797 | 166.3 |

### Load

| 数据集 | 存储 | P50（ms） | P99（ms） | 调用/s |
| --- | --- | --- | --- | --- |
| SIFT | FP32 | 2.871 | 4.288 | 335.2 |
| SIFT | FP16 | 2.494 | 3.462 | 386.9 |
| SIFT | RaBitQ8 | 3.382 | 3.874 | 294.1 |
| GIST | FP32 | 16.959 | 18.400 | 58.4 |
| GIST | FP16 | 7.033 | 11.388 | 134.5 |
| GIST | RaBitQ8 | 5.438 | 5.858 | 185.3 |
| Cohere | FP32 | 8.148 | 14.012 | 115.6 |
| Cohere | FP16 | 6.275 | 9.880 | 151.1 |
| Cohere | RaBitQ8 | 4.680 | 5.513 | 210.5 |

| 数据集 | 存储 | 最终索引（MiB） | Build wall（s） |
| --- | --- | --- | --- |
| SIFT | FP32 | 6.256 | 0.546 |
| SIFT | FP16 | 3.815 | 0.532 |
| SIFT | RaBitQ8 | 2.824 | 0.603 |
| GIST | FP32 | 37.994 | 1.288 |
| GIST | FP16 | 19.684 | 1.295 |
| GIST | RaBitQ8 | 10.762 | 1.561 |
| Cohere | FP32 | 30.670 | 1.244 |
| Cohere | FP16 | 16.022 | 1.235 |
| Cohere | RaBitQ8 | 8.930 | 1.486 |

## 查询与持久化的 API CPU

下表是每进程该操作全部样本的累计CPU时间，再对三个进程取中位数，单位ms；
不是单次耗时，也不包含该API之外的FP64真值和核对搜索。Add/Remove/Update的CPU在各自表中。
getrusage分辨率有限；system接近0不能解释为无系统开销。

| 数据集 | 存储 | 操作 | 调用次数 | 累计user CPU（ms） | 累计system CPU（ms） |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | query | 15000 | 2582.151 | 9.207 |
| SIFT | FP32 | save | 20 | 52.032 | 22.359 |
| SIFT | FP32 | load | 20 | 46.159 | 13.974 |
| SIFT | FP16 | query | 15000 | 2507.725 | 12.257 |
| SIFT | FP16 | save | 20 | 41.139 | 21.999 |
| SIFT | FP16 | load | 20 | 38.509 | 12.051 |
| SIFT | RaBitQ8 | query | 15000 | 3045.015 | 8.598 |
| SIFT | RaBitQ8 | save | 20 | 59.817 | 23.156 |
| SIFT | RaBitQ8 | load | 20 | 64.014 | 3.991 |
| GIST | FP32 | query | 15000 | 6505.032 | 12.817 |
| GIST | FP32 | save | 20 | 73.305 | 243.002 |
| GIST | FP32 | load | 20 | 123.666 | 223.954 |
| GIST | FP16 | query | 15000 | 5169.922 | 12.172 |
| GIST | FP16 | save | 20 | 66.109 | 125.305 |
| GIST | FP16 | load | 20 | 103.859 | 50.547 |
| GIST | RaBitQ8 | query | 15000 | 6445.952 | 9.659 |
| GIST | RaBitQ8 | save | 20 | 83.392 | 43.971 |
| GIST | RaBitQ8 | load | 20 | 89.544 | 20.009 |
| Cohere | FP32 | query | 15000 | 5814.342 | 9.453 |
| Cohere | FP32 | save | 20 | 78.099 | 191.692 |
| Cohere | FP32 | load | 20 | 108.132 | 62.017 |
| Cohere | FP16 | query | 15000 | 5083.835 | 9.084 |
| Cohere | FP16 | save | 20 | 58.670 | 115.940 |
| Cohere | FP16 | load | 20 | 90.198 | 42.150 |
| Cohere | RaBitQ8 | query | 15000 | 6372.915 | 13.487 |
| Cohere | RaBitQ8 | save | 20 | 84.342 | 35.952 |
| Cohere | RaBitQ8 | load | 20 | 70.698 | 19.989 |

## 进程CPU和内存

这里的进程wall/user/system包含真值、Save/Load比对与记账，不能用来代替上面的API延迟。
CPU使用率按(user+system)/wall计算，100%表示约占满一个逻辑核，不是整台多核机器100%。
未采集硬件cycles、cache misses或能耗计数，不能把CPU时间直接解释为已证明的微架构热点。

| 数据集 | 存储 | wall（s） | user（s） | system（s） | 单核CPU（%） |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 17.14 | 17.07 | 0.05 | 99.9 |
| SIFT | FP16 | 16.82 | 16.77 | 0.05 | 100.0 |
| SIFT | RaBitQ8 | 19.44 | 19.39 | 0.05 | 100.0 |
| GIST | FP32 | 96.16 | 95.61 | 0.54 | 100.0 |
| GIST | FP16 | 87.56 | 87.34 | 0.25 | 100.0 |
| GIST | RaBitQ8 | 90.78 | 90.62 | 0.11 | 100.0 |
| Cohere | FP32 | 79.92 | 79.60 | 0.31 | 100.0 |
| Cohere | FP16 | 72.58 | 72.37 | 0.21 | 100.0 |
| Cohere | RaBitQ8 | 77.08 | 76.95 | 0.11 | 100.0 |

| 数据集 | 存储 | 进程峰值（MiB） | 初始RSS（MiB） | 最终RSS（MiB） | RSS变化（KiB） |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 39.922 | 27.234 | 39.922 | 12992 |
| SIFT | FP16 | 32.629 | 24.820 | 32.629 | 7976 |
| SIFT | RaBitQ8 | 30.430 | 24.688 | 30.430 | 5864 |
| GIST | FP32 | 165.801 | 125.602 | 129.195 | 3660 |
| GIST | FP16 | 146.793 | 107.293 | 146.793 | 40448 |
| GIST | RaBitQ8 | 144.383 | 99.117 | 112.633 | 13840 |
| Cohere | FP32 | 135.621 | 102.930 | 135.621 | 33476 |
| Cohere | FP16 | 120.785 | 88.246 | 120.785 | 33320 |
| Cohere | RaBitQ8 | 117.066 | 82.070 | 93.727 | 11936 |

RSS取持久化前的resident观测点；峰值包含输入、FP64真值、原始向量副本和加载副本，
不是索引自身的精确占用。增长可能包含分配器保留和缓存，本组不能单独证明泄漏、零碎片或长期稳定。
表中最终RSS是第20轮Save前的观测值，不是进程退出时的RSS。

| 数据集 | 存储 | minor faults | major faults | voluntary切换 | involuntary切换 |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 12761 | 0 | 1 | 49 |
| SIFT | FP16 | 10887 | 0 | 1 | 44 |
| SIFT | RaBitQ8 | 10314 | 0 | 0 | 59 |
| GIST | FP32 | 244565 | 0 | 0 | 363 |
| GIST | FP16 | 61586 | 0 | 0 | 326 |
| GIST | RaBitQ8 | 52827 | 0 | 0 | 349 |
| Cohere | FP32 | 61315 | 0 | 0 | 297 |
| Cohere | FP16 | 49987 | 0 | 1 | 272 |
| Cohere | RaBitQ8 | 43028 | 0 | 1 | 284 |

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
  lite/benchmark/results/review-fixed-resident-20261020/raw.tar.gz "$BENCH_EVIDENCE"
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

## 与修复前 E9 的关系

本次固定输入、参数、轮数、trial数量和模式轮换，重新从干净的cba3406构建并测量。
E9与E10在不同时间窗口执行，未交替配对；下表只说明版本复验的观测差异，
不能把时延变化归因为某项修复、稳定加速或硬件噪声。无失败后改参/筛选trial。
数值边界修复和回调错误契约属于功能正确性，不要求普通数据延迟一定降低。

| 数据集 | 存储 | E9 P50（μs） | E10 P50（μs） | 变化 |
| --- | --- | --- | --- | --- |
| SIFT | FP32 | 168.985 | 171.094 | +1.25% |
| SIFT | FP16 | 166.545 | 166.704 | +0.10% |
| SIFT | RaBitQ8 | 198.982 | 203.062 | +2.05% |
| GIST | FP32 | 493.855 | 431.371 | -12.65% |
| GIST | FP16 | 383.664 | 346.488 | -9.69% |
| GIST | RaBitQ8 | 441.859 | 435.230 | -1.50% |
| Cohere | FP32 | 389.155 | 382.025 | -1.83% |
| Cohere | FP16 | 333.340 | 336.659 | +1.00% |
| Cohere | RaBitQ8 | 411.492 | 424.622 | +3.19% |

逐进程初始/最终质量CSV完全一致：是；全部持久化快照hash一致：否，可核对两组原始归档。
这不等同于记录了跨版本全部混合查询有序ID/距离；本轮每个版本内部的Save/Load比较仍逐查询执行。

分模式核对：FP32/FP16的全部快照hash跨版本相同，RaBitQ的不同。
源码修复包括normalize从FP32先平方改为FP64平方累计，以及保留真实零半径，
因此不能承诺重新编码后仍与旧版逐字节相同；本组没有隔离证明每一个差异字节的唯一成因。
这与版本内部的Save/Load一致性不同，后者均通过。旧快照读取兼容性及零半径边界见
[RaBitQ设计说明](../../../../docs/docs/zh/src/development/lite_rabitq_design.md)。

## 尚未完成的项目

当前Full公平对照、完整部署包重测、100k统一矩阵、小时级读写/内存耐久均不在本组内。
动态质量下降仍需按确定的质量标准判断和处理。协议与原始数据审计完成，不代表项目全部验收完成。

## 被测版本与可复现边界

功能源码为#2904的cba3406；benchmark协议提交和新编库hash在归档的
frozen-protocol.json、build-source-receipt.json及identity.json中绑定。
输入逐项hash与E9相同，复用已准备好的数据，不重复下载或改变数据预处理。
归档input-reuse-note.json说明：沿用模板receipt中的“before preparation”字段不用于
断言旧数据的准备时间；本轮冻结发生在新测量之前，数据本身在E9阶段已准备好。
计时期间没有C++构建、测试套件或数据准备；Cohere末段做过一次私有Python驱动语法检查，
不声称严格无任何后台工作，也不据小幅时延差异作因果结论。
只把本轮结果作为修复后版本数据；原始E9以及历史Full/包体结果不随合并自动更新。
