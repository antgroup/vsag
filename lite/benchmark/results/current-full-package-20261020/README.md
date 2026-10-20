# 当前版本 Lite / 原生 Full 与运行时包验收

这组新测量使用同一功能源码cba3406，覆盖SIFT128、GIST960、Cohere768各10k向量、1000查询行，
FP32/FP16/RaBitQ8各三次Lite/Full交替配对；不是历史数字拼接，也不是把旧Lite当作Full。
所有54次构建测量、162次独立进程加载与54000次查询结果审计通过。输入与E10逐项hash一致。

## 先看结论与限制

- 有明确的限定运行时包体收益：未压缩载荷约减少93.66%，gzip归档约减少90.56%。
- 本组Lite九个配置的查询P50均低于Full，但GIST/Cohere召回较低，不能称同质量全面胜利。
- 八个配置的暖加载更快；SIFT RaBitQ更慢。所有Lite索引文件都更大。
- 进程总loaded RSS全部更低，但扣除加载前baseline后的增量RSS全部更高。不能把低总RSS当作索引布局全面省内存。
- GIST各模式及Cohere FP32/FP16的builder进程峰值更高；Save也有多项劣势，全部保留。
- 按用户选择只报告实测召回及差值，不设项目质量硬门槛；功能和原始数据审计通过不等于项目全部验收完成。

## 协议与比较对象

Lite是单层图，Full是官方HGraph，内存IO、compressed graph、store_raw_vector=false；
degree16、construction ef128、query ef512、k10、build_thread_count1，CPU0，OMP/OpenBLAS/MKL1。
Full RaBitQ使用3+5、精排fp32、已有默认transform和随机训练过程；Lite固定模型与图维护不同。
各trial实际参数JSON保留；相同参数名不等于相同图、量化器、随机性或召回率。

Full从空目录编译，263个core对象，无旧对象/ccache；使用独立源码副本把官方Release的Ofast
改为O3，与Lite Release O3比较。原始源码未改，overlay、完整compile_commands和receipt在归档中。
这是“原生Full O3实验配置”，不是原生默认Ofast配置、最新版main、全Full测试或全工具链可复现证明。
两边使用相同系统编译器和BLAS/fmt环境，外部依赖源码复用，不复用Full编译产物。

这是初始索引、无CRUD的静态1000行查询；没有warmup轮，查询集合已被观察过，不是盲测。
每个进程的P50/P99再对三个进程取中位数；加载是九个loader样本中位数，不池化查询延迟。
同一查询集合被不同模式和trial重复使用，54000次结果不是54000个独立query。
GIST1000行只有998个不同向量。Cohere初始归一化，比较平方L2；动态不归一化的E10另列。

## 查询与召回

| 数据集 | 存储 | Lite P50（μs） | Full P50（μs） | Lite P99（μs） | Full P99（μs） | Lite Recall@10 | Full Recall@10 | 召回差（百分点） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 167.604 | 276.485 | 196.952 | 312.242 | 0.9971 | 0.9815 | 1.56 |
| SIFT | FP16 | 163.695 | 272.246 | 190.393 | 309.592 | 0.9971 | 0.9815 | 1.56 |
| SIFT | RaBitQ8 | 199.652 | 337.929 | 238.148 | 396.834 | 0.9907 | 0.9742 | 1.65 |
| GIST | FP32 | 461.157 | 556.789 | 603.594 | 656.331 | 0.9646 | 0.9928 | -2.82 |
| GIST | FP16 | 357.358 | 439.899 | 422.842 | 493.956 | 0.9645 | 0.9924 | -2.79 |
| GIST | RaBitQ8 | 439.979 | 526.951 | 512.843 | 591.905 | 0.9607 | 0.9880 | -2.73 |
| Cohere | FP32 | 431.370 | 534.601 | 534.362 | 644.011 | 0.9865 | 0.9942 | -0.77 |
| Cohere | FP16 | 347.198 | 458.438 | 395.035 | 506.805 | 0.9862 | 0.9937 | -0.75 |
| Cohere | RaBitQ8 | 428.140 | 553.959 | 539.700 | 622.875 | 0.9831 | 0.9895 | -0.64 |

召回差为Lite−Full，负数表示Lite较低；完整三次召回和时延均在原始归档，不只保留中位数。

| 数据集 | 存储 | Lite P50变化 | Lite Build（ms） | Full Build（ms） |
| --- | --- | --- | --- | --- |
| SIFT | FP32 | -39.38% | 546.073 | 796.745 |
| SIFT | FP16 | -39.87% | 531.025 | 808.693 |
| SIFT | RaBitQ8 | -40.92% | 601.793 | 920.337 |
| GIST | FP32 | -17.18% | 1320.912 | 1528.813 |
| GIST | FP16 | -18.76% | 1309.563 | 1501.838 |
| GIST | RaBitQ8 | -16.50% | 1599.775 | 1375.620 |
| Cohere | FP32 | -19.31% | 1243.219 | 1558.633 |
| Cohere | FP16 | -24.27% | 1235.229 | 1596.865 |
| Cohere | RaBitQ8 | -22.71% | 1488.367 | 1462.577 |

## 暖加载与保存

加载是新进程、非drop-cache的warm uncontrolled-cache，非严格冷启动；Save无fsync持久性承诺。
API Load计时不含进程启动，不能把它等同于整应用冷启动耗时。每builder只测一次Save。

| 数据集 | 存储 | Lite Load（ms） | Full Load（ms） | Lite Save（ms） | Full Save（ms） |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 3.862 | 5.529 | 4.014 | 3.398 |
| SIFT | FP16 | 2.996 | 3.576 | 3.315 | 2.570 |
| SIFT | RaBitQ8 | 3.966 | 2.784 | 2.646 | 2.001 |
| GIST | FP32 | 18.660 | 32.147 | 17.427 | 15.380 |
| GIST | FP16 | 11.205 | 16.585 | 10.591 | 8.367 |
| GIST | RaBitQ8 | 7.285 | 9.456 | 4.906 | 5.225 |
| Cohere | FP32 | 15.038 | 26.055 | 14.898 | 12.148 |
| Cohere | FP16 | 9.230 | 13.500 | 9.081 | 6.838 |
| Cohere | RaBitQ8 | 6.407 | 8.027 | 4.298 | 4.275 |

## 索引文件与进程峰值

峰值含输入、查询、构建和验证缓冲，不是索引独占内存；MiB=1048576 bytes。

| 数据集 | 存储 | Lite索引（MiB） | Full索引（MiB） | Lite builder峰值（MiB） | Full builder峰值（MiB） |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 6.256 | 5.499 | 27.668 | 44.391 |
| SIFT | FP16 | 3.815 | 2.999 | 22.996 | 39.523 |
| SIFT | RaBitQ8 | 2.824 | 2.102 | 21.719 | 41.871 |
| GIST | FP32 | 37.994 | 37.965 | 126.098 | 112.457 |
| GIST | FP16 | 19.684 | 19.215 | 104.832 | 90.504 |
| GIST | RaBitQ8 | 10.762 | 10.196 | 104.801 | 104.551 |
| Cohere | FP32 | 30.670 | 30.477 | 103.133 | 96.895 |
| Cohere | FP16 | 16.022 | 15.477 | 85.180 | 78.121 |
| Cohere | RaBitQ8 | 8.930 | 8.332 | 84.734 | 89.148 |

## 加载进程的总RSS与增量RSS

增量先逐loader计算loaded RSS−before_load RSS，再对九次取中位数；不是直接相减两个中位数。
Full进程baseline约25–28MiB，Lite约3.4–3.6MiB。总RSS收益与增量成本是不同口径。

| 数据集 | 存储 | Lite总RSS（MiB） | Full总RSS（MiB） | Lite增量RSS（MiB） | Full增量RSS（MiB） |
| --- | --- | --- | --- | --- | --- |
| SIFT | FP32 | 12.289 | 34.590 | 8.719 | 7.594 |
| SIFT | FP16 | 9.832 | 32.078 | 6.273 | 5.148 |
| SIFT | RaBitQ8 | 8.262 | 31.562 | 4.734 | 4.191 |
| GIST | FP32 | 44.074 | 67.051 | 40.484 | 40.078 |
| GIST | FP16 | 25.742 | 48.277 | 22.219 | 21.328 |
| GIST | RaBitQ8 | 16.289 | 39.742 | 12.703 | 12.242 |
| Cohere | FP32 | 36.730 | 59.551 | 33.164 | 32.586 |
| Cohere | FP16 | 22.062 | 44.551 | 18.512 | 17.586 |
| Cohere | RaBitQ8 | 14.453 | 37.883 | 10.875 | 10.379 |

## 实际SIMD分发证据

独立探针读取并调用实际选中的函数指针，用dladdr解析符号和库路径，而不是只看CPUID或编译宏：

| 项目 | Lite | 原生Full |
| --- | --- | --- |
| FP32 L2 | avx512_fp32_distance | avx512::FP32ComputeL2Sqr |
| FP16 L2 | avx512_fp16_distance | avx512::FP16ComputeL2Sqr |
| RaBitQ补充码IP | rabitq_supplement_ip_avx512 | avx512::RaBitQFloatSupplementCodeIP |

128/960/768及129尾部维度的简单可精确表示fixture通过；源码和原始stdout在归档。
这证明探针进程的上述dispatch，并非所有搜索路径的硬件计数/逐调用跟踪；不能据此宣布全部SIMD逻辑相同。
内核代码、累加次序、编码、图和transform仍有差异。没有采集cache misses、能耗或全Full CPU时间。
查询/CRUD逐API user/system CPU属于独立E10数据，不补造成本组Full CPU对照。

## 当前运行时包体与迁移

范围固定为native C++索引应用+递归动态ELF依赖+loader+launcher+license文本，
在新目录解包后通过包内loader运行，并检查实际映射库都来自迁移目录。
只strip副本，原库不变。以下不是单独.so、SDK、wheel、容器或整个OS大小。

| 口径 | Lite（bytes） | Full（bytes） | 减少 |
| --- | --- | --- | --- |
| 未压缩载荷 | 6693492 | 105581610 | 93.66% |
| gzip归档 | 2666743 | 28248144 | 90.56% |

排除SDK头/静态库、Python、容器/OS/kernel/shell、数据/快照和dlopen插件。
系统license通知及通用license文本已复制，但这是测量原型，不提供法律合规认证或完整SDK发行包。
六次迁移smoke覆盖三存储的两边，检查功能及记录召回，不作为三库性能重复实验。
gzip包文件仅作中间测量，归档保留其hash/字节、inventory和实际mapped ELF记录，复现工具可重新生成。
不能把归档中结果证据压缩包的大小误当作产品包大小。

## 原始结果与复核

[raw.tar.gz](raw.tar.gz)：5747575 bytes、45个成员；[archive.json](archive.json)包含SHA256。

包含内层54builder/162loader原始结果、全三次指标、来源绑定、Full编译记录/补丁、参数、
runtime inventory/迁移证明、SIMD探针源码和输出。无原始向量、构建对象、库或可执行文件。
需按输入manifest准备相同数据，解包后可指定新路径复验；原服务器source/data/plain raw不删除。

```bash
BENCH_FULL_EVIDENCE=/absolute/path/to/new-full-evidence
test ! -e "$BENCH_FULL_EVIDENCE"
PYTHONPATH=lite/benchmark python3 -c \
  'from pathlib import Path; from pack_mixed_cpu_acceptance import unpack; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
  lite/benchmark/results/current-full-package-20261020/raw.tar.gz "$BENCH_FULL_EVIDENCE"
PYTHONPATH=lite/benchmark python3 -c \
  'from pathlib import Path; from pack_mixed_cpu_acceptance import unpack; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
  "$BENCH_FULL_EVIDENCE/full/raw.tar.gz" "$BENCH_FULL_EVIDENCE/full/raw"
python3 lite/benchmark/verify_pinned_full_acceptance.py "$BENCH_FULL_EVIDENCE/full/raw" \
  --dataset SIFT=/absolute/path/to/SIFT-10k-1000-query-inputs \
  --dataset GIST=/absolute/path/to/GIST-10k-1000-query-inputs \
  --dataset Cohere=/absolute/path/to/Cohere-10k-1000-query-inputs
python3 lite/benchmark/verify_runtime_package.py "$BENCH_FULL_EVIDENCE/package"
```

外层manifest也须逐文件核对；不更改hash去适配另一份数据。
重跑使用[当前Full协议](../../ACCEPTANCE.md#current-source-native-full-protocol)与
[新编/比较/打包命令](../../README.md#guarded-historical-full-package-and-quality-replay)。
两个Full配置（默认Ofast与O3实验配置）不能混为一组最优值；旧历史结果保持原版本。

## 仍未闭环

本组不包含同质量参数搜索、当前Full长期CRUD吞吐、统一100k、小时级耐久或并发读写支持。
同一Lite实例仍须外部串行。项目的完整验收需结合E10功能/CRUD和此组包体/对照；
社区Actions需维护者批准及kind标签问题仍独立存在，不能写成CI全通过或项目全部完成。
