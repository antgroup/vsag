# 数据集特征分析工具（`profile_dataset`）

`profile_dataset` 用于测量向量数据集中影响索引选择、量化效果、召回率和检索性能的特征，并分三层给出结果：

1. **确定性数据画像**：基于 HDF5 文件计算、带版本号的 JSON 报告；
2. **自包含的 HTML 报告**：可视化全部指标，可离线打开；
3. **可选的 AI 辅助解读**：把聚合指标整理为发现和待基准测试验证的假设。默认关闭，且永远不会接收原始向量。

工具位于 `tools/profile_dataset/`。设计、采样和隐私模型见[技术报告](profile_dataset_design.md)。

## 构建

工具默认不构建，需要通过 `VSAG_ENABLE_TOOLS=ON` 开启（对应 CMake 选项 `ENABLE_TOOLS=ON` 与
`ENABLE_CXX11_ABI=ON`）：

```bash
VSAG_ENABLE_TOOLS=ON make release
# 产物：./build-release/tools/profile_dataset/profile_dataset
```

与 [`eval_performance`](eval.md) 一样，HDF5 由项目构建。该可执行文件不链接 `libvsag`，运行时无需设置
`LD_LIBRARY_PATH`。访问 HTTPS 的 AI 服务需要在构建时提供 OpenSSL 3（`apt install libssl-dev`，安装了
`libcurl4-openssl-dev` 的环境已自带）。缺少 OpenSSL 时工具仍可构建，此时 AI 请求只能发往 `http://`
端点（例如本地网关）；若配置了 `https://` 端点，工具会给出明确提示。

## 支持的输入

工具读取 VSAG 评测工具使用的 [HDF5 数据集格式](dataset_format.md)。

| 布局 | 判定方式 | 分析的数据 |
| --- | --- | --- |
| 稠密 | `type` 属性为 `dense` 或缺失 | `/train` 与 `/test`，形状为 `(N, D)`。`float32` 与 `int8` 与 `eval_performance` 一致；`uint8`、`float64` 和更宽的整数类型可以读取，但会被标记。 |
| 稀疏 | `type` 属性为 `sparse` | `/train` 与 `/test` 字节流，记录格式为 `uint32 len, uint32 ids[len], float32 vals[len]`；存在 `/train_offsets`、`/test_offsets` 时由它们确定记录数，字节流中的记录多于或少于该数目时报错。 |
| 多向量 | `type` 属性为 `multi_vector` | 暂不支持，工具会给出明确错误并退出。 |

`/neighbors`、`/distances`（真值）以及 `/train_labels`、`/test_labels`、`/valid_ratios` 都是可选的；存在时也会被检查。
距离度量取自 `distance` 属性（`euclidean` → `l2`，`ip` → `ip`，`angular` → `cosine`），可用 `--metric` 覆盖。稀疏数据支持
`ip` 和 `cosine`。与 `eval_performance` 要求不一致之处（例如缺少真值、`uint8` 存储、缺少 `distance` 属性）会列在
`input.eval_compatibility` 中并作为警告展示。

## 快速开始（离线）

```bash
./build-release/tools/profile_dataset/profile_dataset \
    --datapath /tmp/data/sift-128-euclidean.hdf5
```

该命令分析文件中存在的所有划分（`train`、`test`），并在当前目录写出 `sift-128-euclidean.profile.json` 和
`sift-128-euclidean.profile.html`，全程无需网络。进度输出到 stderr；stdout 上的摘要列出报告路径、指标摘要值和基于规则的观察结论：

```text
profile_dataset: sift-128-euclidean.hdf5 (dense, metric l2)
  JSON report: sift-128-euclidean.profile.json
  HTML report: sift-128-euclidean.profile.html
  metrics SHA-256: 3c1f...
  observations: 0 critical 1 warning 3 notice 2 info
  AI analysis: disabled
```

常用选项：

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `--datapath`、`-d` | | 待分析的 HDF5 文件 |
| `--output`、`-o` / `--html` | `<stem>.profile.json` / `<stem>.profile.html` | 报告路径；`--no-html` 不生成 HTML 报告 |
| `--splits` | 全部存在的划分 | `train`、`test` 或 `train,test` |
| `--metric` | `auto` | `auto`、`l2`、`ip` 或 `cosine` |
| `--seed` | `42` | 所有采样使用的随机种子 |
| `--sample-size` | `10000` | 每个划分用于采样类指标的行数 |
| `--knn-queries`、`--knn-k` | `1000`、`20` | 在完整划分上执行精确 k-NN 的采样行数与近邻数 |
| `--pair-count` | `100000` | 计算两两距离分布的随机采样点对数 |
| `--clusters` | `16` | 采样 k-means 摘要的 k（`0` 表示关闭） |
| `--pca-max-dim` | `2048` | 维度超过该值时跳过 PCA |
| `--disable` | | 可关闭 `duplicates,knn,pca,clustering,hubness,quantization,ground_truth,labels` 中的任意项 |
| `--duplicate-memory-mb` | `4096` | 精确重复检测的内存预算（每行 24 字节）；超出时改为在样本上检测 |
| `--threads`、`--chunk-mb` | 全部核心、`64` | 只影响执行方式，不影响结果 |
| `--from-profile` | | 基于已有 JSON 报告重新生成 HTML 或运行 AI 分析 |

完整选项（包括下文的 AI 选项）见 `profile_dataset --help`。

## 可复现示例

公开的 [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) 数据集已经采用该格式（`scripts/download_annbench_datasets.sh`
可下载其中几个）：

```bash
mkdir -p /tmp/data
wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5 -O /tmp/data/fashion-mnist-784-euclidean.hdf5
./build-release/tools/profile_dataset/profile_dataset \
    -d /tmp/data/fashion-mnist-784-euclidean.hdf5 --seed 42 -o run1.json --html run1.html
./build-release/tools/profile_dataset/profile_dataset \
    -d /tmp/data/fashion-mnist-784-euclidean.hdf5 --seed 42 -o run2.json --no-html --threads 3 --chunk-mb 8
```

两次运行输出相同的 `metrics SHA-256`，`run1.json` 与 `run2.json` 只在 `run` 部分（时间戳、耗时、线程数、峰值内存）不同。修改
`--seed` 会改变采样类部分，而 `norms`、`dimensions`、`duplicates` 等全量部分保持不变。

## 报告结构

| 部分 | 内容 |
| --- | --- |
| `schema`、`schema_version`、`generator` | `vsag.dataset_profile`、布局版本 `1.0`、构建对应的 VSAG 版本 |
| `input` | 文件名和大小、向量类型、`type`/`distance` 属性、HDF5 数据集、`eval_compatibility`、警告 |
| `config` | 所有可能改变指标值的选项，以及最终使用的度量及其来源 |
| `splits.<train/test>` | 各划分的指标部分（见下文） |
| `cross_split` | 查询（test）与底库（train）的对比 |
| `ground_truth`、`labels` | `/neighbors` 与 `/distances` 的一致性；标签与过滤统计 |
| `signals` | 基于固定阈值的规则观察结论及其引用的报告字段 |
| `determinism` | 覆盖除 `run` 和 `ai_analysis` 外所有部分的 `metrics_sha256` |
| `run` | 开始时间、耗时、线程数、块大小、峰值常驻内存（非确定性） |
| `ai_analysis` | 仅在启用 AI 分析时出现（非确定性） |

每个指标部分都带有 `scope`（`full` 表示完整划分，`sample` 表示样本，`mixed` 表示二者兼有并为每个字段单独标注）和
`definition`。采样部分还带有 `sample`：采样方法、划分大小（`population`）、采样行数与有效采样行数，以及该分析实际使用的行数
（`used_rows`）。直方图记录分箱以及落在绘图范围之外的 `below`、`above` 数量；统计摘要始终使用全部数值。

## 指标定义

距离遵循 VSAG 约定（见[度量语义](metric_semantics.md)）：`l2` 为平方 L2，`ip` 为 `1 - <x, y>`，`cosine` 为 `1 - cos(x, y)`。
需要真正度量的估计量（内在维度、相对对比度、近似重复）使用*几何*距离：欧氏距离；对 `cosine` 使用归一化向量之间的弦距离。

### 稠密划分

| 字段 | 范围 | 定义 | 意义 |
| --- | --- | --- | --- |
| `basic` | 全量 | 行数、维度、元素类型 | 内存与数据类型选择 |
| `invalid_values` | 全量 | 依据 IEEE-754 位模式统计的 NaN、+Inf、-Inf 坐标及包含它们的行 | 距离无定义；这些行不参与范数与距离类指标 |
| `norms` | 全量 | 每个有效行的 L2 范数：摘要（min、p01-p99、max、mean、std）、直方图、变异系数、`unit_norm_ratio`（\|范数 - 1\| <= 1e-3）、零范数行、Tukey 远离群值 | 归一化数据下 `l2`、`ip`、`cosine` 排序一致；范数分散会使 `ip` 偏向大范数向量 |
| `values` | 全量 | 全部有限坐标的 min、max、mean、std、偏度、超额峰度、直方图，以及偏离所在维度 6 个标准差以上的比例 | 标量量化的取值范围利用率 |
| `dimensions` | 全量 | 每个维度的均值、标准差、最小值、最大值、零值比例、偏度、超额峰度和 6-sigma 比例；以及常数维度数、标准差与取值范围比值、前 10% 维度的方差占比、重尾维度比例、均值向量范数与平均行范数之比 | 无效维度、维度间范围不均（逐维 `sq8` 与全局范围 `sq8_uniform` 的取舍） |
| `sparsity` | 全量 | 有限坐标中精确为零的比例、全零行、每行非零元个数 | 是否更适合稀疏表示 |
| `duplicates` | 全量 | 规范化取值（`-0.0 == 0.0`、NaN 统一）完全相同的行，按 128 位哈希分组；示例列出行号 | 浪费内存，top-k 被重复项占据 |
| `pairwise_distances` | 样本 | 随机采样的不同行对之间的度量距离与几何距离；`concentration_ratio` = 几何距离的标准差 / 均值；两两余弦相似度 | 距离集中使近邻难以区分 |
| `nearest_neighbors` | 样本对全量 | 采样行在**完整**划分中的精确 k-NN（排除自身）：第一近邻与第 k 近邻距离、精确重复近邻、近似重复（几何距离 <= `near_duplicate_threshold` x 两两距离中位数）、孤立行（第一近邻距离超过 Tukey 远离群栅栏） | 范围检索半径、重复数据、连接性差的行 |
| `intrinsic_dimensionality` | 样本 | 基于 k 个几何近邻距离的 Levina-Bickel 最大似然局部内在维度；基于第一/第二近邻距离比的 TwoNN 估计 | LID 越高，每个查询需要越多候选 |
| `relative_contrast` | 样本 | 到参考采样行的平均几何距离除以第一近邻距离（He、Kumar、Chang，2012） | 对比度低意味着检索难，需要更大的 `ef_search` |
| `hubness` | 样本 | 每个采样行出现在其他采样行 k-NN 列表中的次数：偏度、反枢纽比例、前 1% 占比（Radovanovic 等，2010） | 枢纽点主导图遍历，反枢纽点难以被访问到 |
| `anisotropy` | 混合 | 两两余弦均值（样本）、均值向量范数比（全量）、第一主成分占比（样本） | 向量集中在窄锥内会压缩余弦距离 |
| `pca` | 样本 | 样本协方差谱：解释方差（前 32 个）、累计曲线、解释 50/80/90/95/99% 方差所需主成分数、有效秩、参与比；维度超过 `--pca-max-dim` 时跳过 | 是否适合在 `tq_chain` 中加入 `pca` |
| `clustering` | 样本 | k-means（k-means++ 初始化）：排序后的簇大小、大小变异系数、最大簇与平均簇之比、归一化熵、轮廓系数 | 粗粒度分区是否均衡，例如 IVF 分桶 |
| `projection` | 样本 | 采样行的二维 PCA 坐标及其所属簇 | 仅用于 HTML 散点图，永远不会发送给 AI 服务 |
| `quantization` | 样本 | 模拟 `fp16`、`bf16`、`sq8`、`sq4`、`sq8_uniform`、`sq4_uniform`：相对精确检索的 recall@k、平均相对重建误差、编码大小（含元数据与对齐）。取值范围、向下取整编码与 0.999 上限截断遵循 `ScalarQuantizationTrainer` 与各量化器；与 VSAG 一致，fp16/bf16 与 uniform 量化器也会编码查询，uniform 量化器在 ip/cosine 下的得分使用存储的原始值之和 | 哪些编码需要 `use_reorder`；属于模型估计，需用 `eval_performance` 确认 |

### 稀疏划分

| 字段 | 范围 | 定义 |
| --- | --- | --- |
| `basic` | 全量 | 记录数、存储条目数、字节流大小、`max_nonzeros`（即 `eval_performance` 推导出的 `dim`）、是否有 offsets、id 未排序或重复的记录数 |
| `invalid_values`、`norms`、`values` | 全量 | 与稠密数据相同，基于存储的取值；`values` 额外统计负值和显式零值 |
| `sparsity` | 全量 | 空记录数、每条记录的条目数、在 id 空间上的密度 |
| `terms` | 全量 | 不同词项数、`max_term_id`、`min_term_id_limit`（SINDI 的 `term_id_limit` 至少应为该值）、term id 密度、倒排条目总数、文档频率摘要与 log10 直方图、Zipf 指数、Gini 系数、前 1% 词项的倒排占比、最高频词项，以及 `doc_prune_retained_term_fraction`：在 SINDI 质量规则下 `doc_prune_ratio` 取 0.1/0.2/0.3/0.5 时平均保留的词项比例 |
| `duplicates` | 全量 | id 与取值完全相同的记录 |
| `pairwise_distances` | 样本 | 随机采样点对的度量距离与余弦相似度；没有共同词项的点对比例 |
| `nearest_neighbors` | 样本对全量 | 基于采样查询倒排索引的精确 k-NN；无共同词项的记录距离恰为 1，按行号进入 k-NN 列表（排在内积为负的记录之前）；含非有限值的记录被排除；重叠记录少于 k 个的查询数 |
| `hubness` | 样本 | 与稠密数据相同 |

### 跨划分、真值与标签

| 字段 | 范围 | 定义 |
| --- | --- | --- |
| `cross_split.norm_ks_statistic` | 全量 | train 与 test 范数分布的双样本 Kolmogorov-Smirnov 统计量 |
| `cross_split.standardized_mean_shift` | 全量 | 每个维度的 \|mean_test - mean_train\| / std_train（稠密） |
| `cross_split.out_of_distribution_ratio` | 样本 | 到 train 最近距离超过 train 行自身最近邻距离第 95 百分位的采样查询比例 |
| `cross_split.query_term_coverage`、`postings_per_query`、`nonzeros_ks_statistic` | 样本 / 全量 | 仅稀疏数据：查询词项在 train 中出现的比例、穷举倒排扫描需要访问的条目数、条目数分布的 KS 统计量 |
| `ground_truth` | 样本 | 采样查询的 `/neighbors`、`/distances` 与重新计算的精确近邻对比：id 召回率、距离召回率（重新计算距离后落在精确第 k 近邻距离之内的存储 id 比例，因此容忍并列）、存储距离与存储 id 重新计算的距离之间最匹配的距离约定（`euclidean` 为 `sqrt(L2Sqr)` 还是平方；`ip`/`angular` 为 `1 - 相似度` 还是相似度本身）、越界 id、未排序的行 |
| `labels` | 全量 | 标签数量、最大标签占比、归一化熵、每个查询的过滤选择率、标签在底库中不存在的查询数、`valid_ratios` 摘要 |

### 基于规则的观察结论

`signals` 对上述指标应用固定阈值，例如 `invalid_values.total > 0`（严重）、`duplicates.duplicate_ratio > 0.01`（警告）、
`relative_contrast.summary.p50 < 2`、`hubness.skewness > 1`、`pca.components_for_variance.90 <= 0.25 * dim`、模拟 `sq8`
recall@k 低于 0.95、查询分布偏移（范数 KS > 0.1 或分布外比例 > 0.1），以及存储的真值与重新计算的近邻不一致。每条结论都记录了规则和所引用的报告字段。
它们无需任何 AI 服务即可使用，也是校验 AI 输出的参照。

## 确定性、采样与内存

- 所有采样都使用 xoshiro256** 生成器，其种子由 `--seed` 与流名称（`sample/<split>`、`order/<split>`、`pairs/<split>`、
  `kmeans/<split>`）派生。行通过 Floyd 算法无放回均匀抽取，再按带种子的随机顺序排列；每项分析使用该顺序的一个前缀，因此每个子集本身也是均匀样本。
  所有随机抽取都有完全确定的定义，同一种子在任何编译器、标准库和平台上都得到相同的行。
- 全量统计按固定的 4096 行分块计算，并按块顺序合并；k-NN 列表使用严格的（距离，行号）全序；k-means 和特征分解串行、双精度执行。
  该目标以 `-fno-fast-math -ffp-contract=off` 编译，并通过位模式识别 NaN/Inf。因此结果与 `--threads`、`--chunk-mb` 无关，
  可以用 `determinism.metrics_sha256` 验证。
- 内存有界：每行保存一个范数、一个非零元计数和一个有效标志（稠密数据为 13 字节），外加 16 字节的重复检测哈希（分组时为 24 字节，受
  `--duplicate-memory-mb` 限制）。采样类计算受选项约束：样本为 `sample_size x dim` 个浮点数，k-NN 状态为 `knn_queries x k`，样本内分析最多使用
  `sample_size` 行。工具从不为完整划分构建 `N x N` 的两两矩阵，完整划分只被流式读取两遍。

## HTML 报告

HTML 报告是单个文件，包含内联 CSS、内联 SVG 图表和少量内联脚本。其内容安全策略（`default-src 'none'`）禁止任何网络请求，并且只允许页面自身的脚本运行（以其 SHA-256 哈希固定），因此可以直接从磁盘打开，
不会访问任何服务。报告包括：

- 总览：关键指标与基于规则的观察结论；
- 每个划分一个标签页；稠密与稀疏划分使用不同的图表（范数、取值、逐维带状图、非零元、两两距离与近邻距离、LID、相对对比度、枢纽度、PCA 谱、
  叠加了测试查询的二维投影、簇大小、模拟量化召回率；或者文档频率、高频词项与 SINDI 剪枝曲线）；
- 查询与底库对比、真值、标签、AI 分析等标签页，以及配置页：可筛选的全部指标索引和下载内嵌 JSON 的按钮。

每张图表都标注指标字段、指标定义，以及覆盖的是完整划分（`N = ...`）还是样本（`n = ... of ...`、种子）。直方图可切换为对数坐标，证据链接可跳转到支撑某条结论的图表。

## AI 辅助解读

只有显式配置服务时才会启用 AI 分析：

| 选项 | 环境变量 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--ai-provider` | `VSAG_PROFILE_AI_PROVIDER` | `none` | `openai`（任何兼容 OpenAI chat completions 的 API）或 `anthropic`（Messages API） |
| `--ai-endpoint` | `VSAG_PROFILE_AI_ENDPOINT` | 服务默认值 | 基础 URL，例如 `https://api.openai.com/v1`；以 `/chat/completions` 结尾的 URL 会被原样使用 |
| `--ai-model` | `VSAG_PROFILE_AI_MODEL` | | 模型名称（必填） |
| `--ai-api-key-env` | | `VSAG_PROFILE_AI_API_KEY` | 保存密钥的环境变量名 |
| `--ai-auth-header` | | `Authorization` | 携带密钥的请求头；`Authorization` 发送 `Bearer <key>`，其他名称（例如 Azure OpenAI 的 `api-key`）直接发送密钥 |
| `--ai-language` | `VSAG_PROFILE_AI_LANGUAGE` | `en` | 生成文本的语言：`en` 或 `zh` |
| `--ai-timeout`、`--ai-retries` | | `60`、`1` | 单次请求超时（秒）；对传输错误、HTTP 429 与 5xx 的重试次数 |
| `--ai-dry-run`、`--ai-request-out` | | | 只构建（并保存）完整请求体，不发送 |
| `--ai-output` | | | 另外把 `ai_analysis` 部分写入单独文件 |
| `--ai-required` | | 关闭 | AI 分析未成功时以退出码 3 结束（报告仍会写出） |

API Key 只从 `--ai-api-key-env` 指定的环境变量读取；工具没有任何接收密钥的命令行选项，因此密钥不会进入 shell 历史或进程列表。密钥只出现在请求头中，
并会从所有消息、报告和日志中脱敏。支持 `HTTPS_PROXY`、`HTTP_PROXY` 和 `NO_PROXY`。

```bash
# OpenAI 或任意兼容 OpenAI 的服务（DeepSeek、通义千问 DashScope 兼容模式、vLLM 等）
export VSAG_PROFILE_AI_API_KEY=...        # 不要把密钥作为参数传入
./build-release/tools/profile_dataset/profile_dataset -d /tmp/data/sift-128-euclidean.hdf5 \
    --ai-provider openai --ai-model gpt-4o-mini

# 通过 Ollama 部署的本地模型：无需密钥，数据不离开本机
./build-release/tools/profile_dataset/profile_dataset -d /tmp/data/sift-128-euclidean.hdf5 \
    --ai-provider openai --ai-endpoint http://127.0.0.1:11434/v1 --ai-model qwen2.5:7b

# 查看将要发送的完整内容，但不真正发送
./build-release/tools/profile_dataset/profile_dataset -d /tmp/data/sift-128-euclidean.hdf5 \
    --ai-provider openai --ai-dry-run --ai-request-out request.json

# 不重新分析数据，基于已有报告用中文重新解读
./build-release/tools/profile_dataset/profile_dataset --from-profile sift-128-euclidean.profile.json \
    --ai-provider anthropic --ai-model claude-sonnet-4-5 --ai-language zh
```

### 隐私行为

请求只包含一个扁平的证据对象，其中是白名单内的**聚合**指标（摘要、比例、计数和较短的谱）、基于规则的观察结论、最终使用的度量以及采样配置。
请求永远不包含向量、采样行、行号、PCA 坐标、逐维数组、直方图、term id、标签值、文件路径或认证信息。只有指定 `--ai-include-file-name`
时才会包含文件名。证据大小与数据行数无关。报告会记录 `request_sha256`、证据字段数量和 `raw_vectors_sent: false`；`--ai-dry-run`
可以查看完整请求内容。

### 响应约定与校验

模型必须返回一个 JSON 对象：`summary`、`findings`（title、severity、detail、evidence）、`recommendations`（title、category、
hypothesis、rationale、validation、confidence、evidence）、`uncertainties` 和 `additional_measurements`。工具会拒绝证据字段缺失或不在请求中的条目、
字段非法的条目，以及把配置表述为必然最优或有保证的建议；被拒绝的条目列在 `ai_analysis.rejected_items` 中。格式错误的回复得到
`status: invalid_response`，服务不可达得到 `status: failed`。无论哪种情况，确定性的 JSON 与 HTML 报告都已在请求之前写出；工具会输出警告，
并以退出码 0 结束，除非设置了 `--ai-required`。模型输出在 HTML 报告中会被转义。

### 验证 AI 建议

请把每条建议都当作假设：

1. 在 HTML 报告中核对所引用的证据（证据链接可跳转到图表），确认现象真实存在而不是采样误差（若采样指标处于阈值附近，可增大 `--sample-size`）。
2. 把假设转化为带基线的 [`eval_performance`](eval.md) 实验：例如比较 `base_quantization_type` 为 `sq8` 时开启与不开启
   `use_reorder`，或在相同召回目标下比较 `sq8` 与 `tq_chain` 中的 `"rom, sq8_uniform"`。
3. 为每个候选方案分别调优检索参数（`ef_search`、`scan_buckets_count`），使用数据集的真实查询，在固定召回率下比较延迟与 QPS。
4. 只有基准测试证实后才采纳建议，并把画像的 `metrics_sha256`、工具版本和基准测试配置与结果一起记录。

## 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 报告已写出（如有 AI 警告会输出） |
| `1` | 选项或输入文件非法 |
| `3` | 设置了 `--ai-required` 且 AI 分析未成功；报告仍已写出 |

## 局限性

- 暂不支持多向量数据集。
- 采样类指标存在采样误差，其 `sample` 字段给出了样本规模。
- LID 与 TwoNN 假设局部数据近似均匀；相对对比度和枢纽度依赖 k 的取值。
- 量化部分是对 VSAG 量化器的建模，并不调用量化器本身，也不能替代 `eval_performance`。
- 重复检测基于哈希，误判概率约为 `N^2 / 2^129`。
- 稀疏数据不计算内在维度、对比度、PCA 和聚类类指标。
- 即使通过了校验，AI 输出仍可能有误；它只是需要基准测试验证的建议，而不是决策。
