# 数据集特征分析工具：技术报告

本报告说明 [`profile_dataset`](profile_dataset.md) 如何选择与计算指标、采样与可复现性如何实现、可选 AI 层的隐私边界在哪里、AI 提示与响应如何构造和校验、
工具如何被验证，以及它不做什么。

## 1. 目标与非目标

目标：

- 基于 `eval_performance` 已在使用的 HDF5 文件，测量影响 VSAG 决策（索引类型、度量、量化器、变换、检索参数）的数据集特征。
- 让每个数值可复现：同一文件、选项和种子，在任意线程数或 I/O 块大小下得到逐位相同的指标，并在所有平台上得到相同的采样行。
- 内存有界：流式读取完整划分，从不构建 `N x N` 的两两矩阵，并用显式选项约束每一项采样计算。
- 离线解释结果（HTML 和基于规则的观察结论），并可选地借助只看到聚合指标、输出经过校验后才展示的 AI 服务。

非目标：训练或微调模型、修改或构建索引、启动生产任务、上传原始向量、替代可复现的 VSAG 基准测试。

## 2. 架构

```text
HDF5 文件
  |  inspect_dataset（结构、属性、与 eval_performance 的兼容性）
  v
逐个划分（先处理 test，使其样本能在同一遍扫描中对 train 检索）
  第 1 遍：以 4096 行为块的全量流式扫描
          非法值、范数、逐维矩、零值、重复哈希，捕获采样行
  第 2 遍：全量流式扫描
          取值直方图、6-sigma 离群值、采样查询（以及另一划分采样查询）的精确 k-NN
  样本分析：两两距离、LID、TwoNN、相对对比度、枢纽度、PCA、k-means、二维投影、量化模拟
  v
JSON 报告：input、config、splits、cross_split、ground_truth、labels、
          signals（规则）、确定性摘要、run
  |-- HTML 渲染（内联 SVG、CSP）---------------------------> report.html
  '-- 可选 AI：白名单证据 -> 提示 -> 服务 -> 校验
               -> ai_analysis -> 重写 JSON 与 HTML
```

| 文件 | 职责 |
| --- | --- |
| `tools/profile_dataset/hdf5_source.cpp` | 文件检查、稠密数据流式读取、稀疏字节流解析 |
| `tools/profile_dataset/dense_profiler.cpp`、`tools/profile_dataset/sparse_profiler.cpp` | 两遍扫描的划分分析器 |
| `tools/profile_dataset/dataset_profiler.cpp`、`tools/profile_dataset/profile_signals.cpp` | 流程编排、跨划分对比、真值、标签、规则、摘要 |
| `tools/profile_dataset/neighbor_search.cpp` | 精确 k-NN（稠密分块、稀疏倒排）、LID、TwoNN |
| `tools/profile_dataset/statistics.cpp`、`tools/profile_dataset/deterministic_random.cpp` | 矩、直方图、分位数；可移植的随机数与采样 |
| `tools/profile_dataset/linear_algebra.cpp`、`tools/profile_dataset/kmeans.cpp`、`tools/profile_dataset/quantization_simulation.cpp` | PCA、聚类、量化器模型 |
| `tools/profile_dataset/html_report.cpp`、`tools/profile_dataset/svg_chart.cpp` | 自包含 HTML 报告 |
| `tools/profile_dataset/ai_analysis.cpp`、`tools/profile_dataset/http_transport.cpp` | 证据、提示、校验、服务适配、传输 |
| `tools/profile_dataset/profile_cli.cpp` | 命令行、输出顺序、退出码 |

工具不链接 `libvsag`；它与 VSAG 共享 HDF5 布局、度量约定和量化器规则，但所有计算都自行完成，因此不可能修改任何索引。

## 3. 指标选择

每个指标回答 VSAG 用户面临的一个问题；基于规则的观察结论把对应阈值固化下来。

| 问题 | 指标 | 背景 |
| --- | --- | --- |
| 数据对该度量是否有效？ | 依据位模式统计的 NaN/Inf、零向量、稀疏 id 未排序或重复 | 距离无定义；零向量的余弦无意义 |
| `l2`、`ip`、`cosine` 能否互换？ | 单位范数比例、范数变异系数 | 归一化后三者排序一致；范数分散时 `ip` 变成 MIPS 问题 |
| 数据是否浪费内存？ | 精确重复（全量）、近似重复（k-NN）、常数维度、稀疏度 | 重复项占据 top-k 结果和图的边 |
| 近邻检索有多难？ | 相对对比度、LID（MLE）、TwoNN、距离集中度、枢纽度 | He、Kumar、Chang，ICML 2012；Levina、Bickel，NeurIPS 2004；Facco 等，Sci. Rep. 2017；Aumuller、Ceccarello，2019（LID 与 ANN 难度）；Radovanovic 等，JMLR 2010（枢纽度） |
| 适合哪种量化与变换？ | 逐维范围、6-sigma 离群值、峰度、PCA 谱、模拟 fp16/bf16/sq8/sq4/sq8_uniform/sq4_uniform 的召回率 | 复现 `ScalarQuantizationTrainer`：逐维截断边界（8 位截断率 0.001，4 位 0.05），`sq8_uniform` 用全局 min/max，`sq4_uniform` 用最窄的截断边界，向下取整编码并按 0.999 截断；fp16/bf16 与 uniform 量化器也编码查询，uniform 的 ip/cosine 得分由编码乘积加存储的原始值之和得到，编码大小包含元数据与对齐 |
| 类 IVF 分区是否均衡？ | k-means 簇大小变异系数、最大/平均簇大小、轮廓系数 | 粗略代理；IVF 实际使用的分桶数多得多 |
| 查询是否与底库相似？ | 范数 KS、标准化均值偏移、分布外比例、二维投影叠加 | 分布外查询需要用真实查询调参 |
| 真值是否可信？ | 与重新计算近邻对比的 id 召回率；基于存储 id 重新计算距离得到的距离召回率与距离约定 | 错误的真值会悄无声息地污染所有召回率数字 |
| 稀疏 / SINDI 参数 | `max_term_id`（决定 `term_id_limit`）、id 密度（`remap_term_ids`）、文档频率与 Zipf/Gini、SINDI `doc_prune_ratio` 质量规则下保留的词项、每个查询的倒排条目数 | 倒排链长度主导 SINDI 的计算量 |

距离遵循 VSAG 约定（`l2` 为平方、`ip` 为 `1 - <x, y>`、`cosine` 为 `1 - cos`）。需要度量空间的估计量（LID、TwoNN、相对对比度、近似重复）
使用欧氏距离，`cosine` 使用归一化向量之间的弦距离；对 `ip`，同一遍扫描会额外维护一份欧氏近邻列表。

## 4. 采样设计

- **行。** 每个划分用种子为 `derive_seed(seed, "sample/<split>")` 的生成器，以 Floyd 算法无放回均匀抽取 `--sample-size` 个行号并排序，
  以便在流式扫描中捕获。没有 offsets 的稀疏划分会先计数。
- **顺序。** 用 `order/<split>` 流打乱采样位置；跳过不适合计算距离的行（非有限值、`cosine` 下的零向量）。每项分析都使用该顺序的*前缀*：
  前 `--knn-queries` 行是 k-NN 查询，前 `--hubness-sample` 行构建样本内近邻图，以此类推。均匀样本的随机排列的前缀本身也是均匀样本，
  因此每项分析都是无偏的，且各子集相互嵌套，使不同选项下的结果可比。
- **独立的流。** 点对（`pairs/<split>`）与 k-means 初始化（`kmeans/<split>`）使用各自的种子，修改一个选项不会改变另一项分析使用的行。
- **对完整划分做 k-NN。** 近邻距离在完整划分中检索，而不是在样本中：样本内的近邻总是比真实近邻更远。代价是一遍完整的流式扫描
  （时间 `O(N * D * queries)`），内存为 `O(queries * k)`。
- **标注。** 每个采样部分都记录方法、总体大小、采样行数与有效行数，以及实际使用的行数；HTML 在每张图上展示这些信息。采样误差随样本增大而减小；
  处于阈值附近的采样指标可以用更大的 `--sample-size` 或 `--knn-queries` 确认。

## 5. 可复现性工程

| 风险 | 措施 |
| --- | --- |
| 随机分布依赖标准库实现 | 自实现 xoshiro256** 与 SplitMix64 播种、基于拒绝采样的有界抽取、Floyd 采样、Fisher-Yates 洗牌（已与独立的 Python 实现对照） |
| 线程数和块大小改变浮点求和顺序 | 固定 4096 行分块，按块顺序用 Pebay 公式合并矩；逐对计算的核函数使用固定的结合顺序 |
| k-NN 顺序依赖调度 | 严格的（距离，行号）全序；最小的 k 个元素构成唯一集合 |
| 编译器优化 | 目标以 `-fno-fast-math -ffp-contract=off` 编译（否则 release 构建使用 `-Ofast`）；依据位模式识别 NaN/Inf |
| 线性代数库 | 串行、双精度的 Householder + 隐式 QL 特征分解；固定特征向量符号 |
| 聚类 | 由该随机数生成器播种的 k-means++，平局取编号最小的中心，按行顺序累加中心 |
| 证明 | `determinism.metrics_sha256` 覆盖除 `run` 与 `ai_analysis` 外的所有部分 |

实测：稠密与稀疏测试数据在 1、7、24 线程以及 1 MB、64 MB 块大小下摘要完全相同；下文三个公开数据集在 24 线程、64 MB 块与 3 线程、8 MB 块下摘要完全相同；单元测试中亦然。GCC 11 与 GCC 13 的 release 构建在这些数据集上得到完全相同的指标。采样行来自整数运算，在所有平台上一致。浮点指标还用到 C 库的 `log`、`exp`、`hypot`，因此不同数学库或不同架构生成的报告可能在最后几位上有差异。

## 6. 复杂度、内存与实测性能

| 阶段 | 时间 | 内存 |
| --- | --- | --- |
| 第 1 遍 | `O(N * D)` | 稠密数据每行 13 字节（范数、非零计数、有效标志），每行 16 字节重复哈希（分组时 24 字节，受 `--duplicate-memory-mb` 约束），一个 I/O 块 |
| 第 2 遍 | `O(N * D)`，精确 k-NN 额外 `O(N * D * Q)`（`Q` 为本划分与跨划分查询数之和） | `O(Q * k)` 近邻列表，一个 I/O 块 |
| 两两距离 | `O(P * D)` | `O(P)` |
| 枢纽度 | `O(H^2 * D)` | `O(H * k)` |
| PCA | `O(S * D^2 + D^3)`；维度超过 `--pca-max-dim` 时跳过 | `O(S * D + D^2)` |
| k-means 与轮廓系数 | `O(I * S * K * D)` 与 `O(2000^2 * D)` | `O(S * D)` |
| 量化模拟 | `O(方案数 * 查询数 * 底库 * D)` | `O(底库 * D)` |

默认选项（采样 10,000 行，每个划分 1,000 个 k-NN 查询、k = 20，100,000 个点对），GCC 13 release 构建，24 线程（Intel Core Ultra 9 275HX，WSL2），
公开 ann-benchmarks 文件。测量时机器上还有其他构建在运行（平均负载约 25），因此耗时是上限：

| 数据集 | train 行数 x 维度 | 耗时 | 峰值 RSS |
| --- | --- | --- | --- |
| `fashion-mnist-784-euclidean` | 60,000 x 784 | 13.3 s | 191 MB |
| `sift-128-euclidean` | 1,000,000 x 128 | 17.2 s | 166 MB |
| `glove-100-angular` | 1,183,514 x 100 | 15.9 s | 178 MB |

结果与这些数据集的已知特点一致：SIFT1M 含 14,538 条精确重复行（1.45%）；GloVe 在三个数据集中对比度最低、LID 最高
（train 划分的相对对比度中位数为 1.62、LID 中位数为 32.7），与其“难数据集”的名声相符。

两种维度估计可能不一致。SIFT1M train 划分的 TwoNN 为 3.9，LID 中位数为 14.2。TwoNN 只使用每个查询的前两个近邻，LID 使用前 20 个，
因此报告同时给出两者。并列和重复不能解释这一差距：它们只排除了 2.4% 的查询，去掉第一近邻最近的 10% 查询后 TwoNN 也只升到约 5.0。

## 7. 验证策略

- **单元测试**（`profile_dataset_test`）：随机数输出与独立 Python 参考实现对照、FIPS 180-4 SHA-256 测试向量、矩与直接公式及任意合并切分对照、
  直方图边界、分位数与 NumPy 定义对照、距离核函数与直接循环对照、半精度舍入、top-k 与插入顺序无关、各度量下 k-NN 与暴力检索及任意分块一致、
  已知维度数据上的 LID/TwoNN、特征分解残差与正交性、平面数据的 PCA、可分簇上的 k-means、量化器边界规则。
- **植入缺陷的测试数据**，在测试时确定性生成（仓库中没有二进制文件）：NaN/+Inf/-Inf 行、重复行、零向量、范数离群行、哨兵行、含有底库中不存在标签的查询，
  以及含空记录与重复记录的稀疏数据。测试检查精确计数、真值约定、标签选择率和观察结论。
- **读取器健壮性**：多种块大小下跨块边界的记录、未排序与重复 id、截断的字节流和损坏的长度（在分配内存前即被拒绝）、错误的秩、空的或不匹配的数据集、
  未知类型、非 HDF5 文件。
- **确定性与有界性**：不同线程数、块大小和多次运行得到相同摘要；修改种子只改变采样部分；按配置采样的 10 万行画像峰值内存低于 2 GB
  （两两矩阵需要 80 GB）。
- **HTML**：存在内容安全策略且固定了脚本哈希，没有外部 `src`/`href`/`@import`/`url()`，恰好两个 script 元素，文件名、划分名与模型输出均被转义，每张图表都带有指标字段、定义、
  范围以及总体或样本大小。
- **AI 层**：请求中不含哨兵值、文件名、逐行数组或认证信息；密钥只出现在请求头中，从不出现在报告、stdout 或 stderr 中；拒绝违反结构、
  引用未知证据、过度承诺以及格式错误的回复；服务不可达、HTTP 401、429/503 重试、dry run，以及针对本地模拟服务的端到端运行（以及停止服务后的再次运行）
  都会留下有效的确定性报告。
- **独立参考实现**：`tools/profile_dataset/validate_profile.py` 用 Python 复现采样，选出相同的行，并用 NumPy 重新计算全量指标与基于精确 k-NN 的指标。
  在稠密测试数据（两个划分）、Fashion-MNIST、SIFT1M 和 GloVe 上全部检查通过：采样行完全一致，k-NN 距离分位数、LID 和 TwoNN 约 10 位有效数字一致，
  逐维均值与标准差的相对差异在 1e-11 以内。
- **与 VSAG 本身对照量化器**：`tools/profile_dataset/validate_quantization.py` 用 pyvsag 的 `brute_force` 索引对相同的采样基向量逐个构建模拟的量化器，
  并把 recall@k 与报告比较。在 SIFT1M、GloVe 和 Fashion-MNIST 上，全部 18 个召回率（每个数据集 6 种量化器）都与 pyvsag 0.18.5 的结果相同。
- **CI**：测试程序与其他工具测试一起在社区与团队流水线中运行，包括 AddressSanitizer 构建。

## 8. 隐私边界

| 资产 | 边界 |
| --- | --- |
| 原始向量与采样行 | 从不离开进程。AI 请求由聚合报告字段的白名单构建；逐行数据（投影坐标、重复与离群示例）、逐维数组、直方图、term id 与标签值在构造上即被排除。证据大小与 N 无关。 |
| 文件路径与文件名 | 路径从不发送；只有使用 `--ai-include-file-name` 时才发送文件名。 |
| 认证信息 | 只从环境变量读取，只放在请求头中，并从所有消息中脱敏；拒绝 URL 中内嵌的认证信息；报告中保存的端点不含查询字符串。 |
| 不可信的模型输出 | 依据请求中的证据字段和结构进行校验，在 HTML 中转义，并与确定性部分分开存放；它从不改变指标或观察结论。 |
| 报告文件 | 包含派生逐行数据（投影、示例行号）的本地产物；HTML 页面用 CSP 阻止所有网络请求。请像对待数据集本身一样分享它们。 |
| 网络 | 未显式配置 AI 服务时不访问网络；失败时回退到离线报告。 |

## 9. AI 提示与响应结构

请求（提示版本 `1`，温度 0）包含两条消息：

- **System**：角色；规则（只使用 EVIDENCE；准确引用 EVIDENCE 字段；建议是需要用 `eval_performance` 验证的假设，不得使用 “guaranteed”、
  “always optimal”、“best possible”、“definitely” 等措辞；采样与模拟结果存在不确定性；使用 VSAG 的索引、量化器、`tq_chain`、SINDI 与检索参数术语；
  覆盖特征、影响、建议、不确定性和缺失的测量；只返回一个 JSON 对象）；`zh` 时要求以简体中文撰写文本；随后是响应结构。
- **User**：证据对象，即从报告字段到取值的扁平映射。

响应结构：

```json
{
  "summary": "string, at most 800 characters",
  "findings": [{"title": "", "severity": "info | notice | warning | critical", "detail": "",
                "evidence": ["EVIDENCE key"]}],
  "recommendations": [{"title": "", "category": "index | metric | quantization | transformation | evaluation | data_quality",
                       "hypothesis": "", "rationale": "", "validation": "", "confidence": "low | medium | high",
                       "evidence": ["EVIDENCE key"]}],
  "uncertainties": [""],
  "additional_measurements": [""]
}
```

校验：回复可以包裹在代码块或文字中；工具提取并解析其中的 JSON 对象（最多 256 KB）。`summary` 与四个数组必须存在。每个条目必须在长度限制内提供必填字段，
并使用合法的枚举值；不在请求中的证据字段会被移除，失去全部证据的条目被拒绝；包含保证或最优之类说法（中英文措辞）的建议被拒绝；每类最多保留 20 条。
若没有任何条目通过，状态为 `invalid_response`。被拒绝的条目及原因保存在 `ai_analysis.rejected_items` 中供审阅。状态包括：`ok`、`failed`
（配置、传输或 HTTP 错误）、`invalid_response`、`dry_run`。

## 10. 局限性

- 不分析多向量数据集。
- 精确 k-NN 的代价为 `O(N * D * queries)`；对非常大的划分可减小 `--knn-queries`。
- 采样指标存在采样误差；LID 与 TwoNN 假设局部均匀，且由于 TwoNN 只使用前两个近邻，两者可能不一致（`two_nn_excluded_ratio` 报告被跳过的查询比例）。
- 量化部分对 VSAG 标量量化器（范围、编码、查询编码）建模，不覆盖 PQ、RaBitQ 或变换链，也不能替代基准测试。
- 较小 k 的 k-means 只是分区结构的粗略视图。
- 图相关的难度指标（例如 Steiner-hardness）需要构建索引，不在范围内。
- 即使在所引用证据的范围内，AI 层仍可能生成看似合理但错误的文字，因此其输出被表述为假设。
