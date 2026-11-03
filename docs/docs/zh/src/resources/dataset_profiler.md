# 数据集特征可视化（`dataset_profiler`）

VSAG 已提供索引性能评测工具（`eval_performance`）与索引序列化结构可视化工具
（`visualize_index`）。**数据集特征分析工具**从另一个角度补全了信息：它分析并可视化*输入向量
数据集本身的特征*——向量范数、各维度数值分布、稀疏度、重复数据、采样距离分布、异常值与各向异性，
这些因素会显著影响索引选择、量化质量、召回率与检索性能。

该工具是纯 Python 实现（位于 `tools/dataset_profiler`），直接读取 `eval_performance` 使用的稠密
与稀疏 HDF5 文件，输出两类产物：

- **带版本号的 JSON 报告**：每项指标都注明定义、统计范围（全量或采样）以及样本数量；
- **自包含的交互式 HTML 报告**：CSS/JS/SVG 全部内联，可离线打开；

另有**可选的 AI 辅助解读**：只解释*聚合指标*（绝不接触原始向量），并把每条建议表述为附有指标
依据的待验证假设。

## 安装

工具需要 Python 3.10+ 以及 NumPy、h5py，不参与 C++ 构建。

```bash
cd tools/dataset_profiler
pip install -r requirements.txt
```

## 快速开始

```bash
# 使用默认确定性配置分析 train 分片
python -m dataset_profiler /path/to/my_dataset.hdf5
```

默认在输入文件旁生成 `my_dataset.profile.json` 与 `my_dataset.profile.html`；用
`--out-json` / `--out-html` 指定输出位置（`--out-json -` 表示将 JSON 打印到标准输出）。

更完整的调用示例：

```bash
python -m dataset_profiler my_dataset.hdf5 \
    --split train --seed 42 \
    --sample-size 512 --pairwise-sample-size 256 \
    --metric euclidean --clusters 8 \
    --out-json my_dataset.profile.json --out-html my_dataset.profile.html
```

`examples/make_toy_datasets.py` 可端到端重新生成自带的演示数据与报告；`examples/output/`
目录中保存了示例 JSON/HTML 文件。

## 支持的数据格式

工具读取[HDF5 数据集格式](dataset_format.md)中定义的两种布局：

- **稠密**（无 `type` 属性或 `type="dense"`）：`/<split>` 是 `(N, D)` 的 `INT8` 或 `FLOAT32`
  矩阵；
- **稀疏**（`type="sparse"`）：`/<split>` 是扁平字节流，每条记录为
  `uint32 len | uint32 ids[len] | float32 vals[len]`，可选地由 `/<split>_offsets` 索引。稀疏
  维度按 `max(特征 id) + 1` 推断。

`multi_vector` 文件会被拒绝并给出说明。数据按有界分块流式读取：工具不会把完整数据集载入内存，
两两距离计算受采样规模约束，绝不会为完整数据集分配 `O(N²)` 矩阵。

如果数据只存在于内存中（例如准备传给 `pyvsag` 的数组），可先用 h5py 按上述布局导出为 HDF5，
参见 `tools/dataset_profiler/examples/README.md`。

## 指标

JSON 报告中每项指标都带有人类可读的 `definition`、`scope`（`full`、`sample` 或 `mixed`）以及
所用 `sample_size`；HTML 报告在每一节中重复这些溯源信息。

| 指标 | 范围 | 说明 |
| --- | --- | --- |
| 数据集概况 | 全量 | 向量数、维度、元素类型、文件属性。 |
| `norms` | 全量 | 每个向量的 L2 范数分布。 |
| `dimensions` | 全量 | 每个坐标维度的均值、标准差、最小/最大值、零值比例；单独列出方差最高与零值比例最高的维度；超长序列会按桶平均并标注。 |
| `sparsity` | 全量 | 零值/非零值计数与密度；稀疏文件还统计存储条目数与每向量非零元素（nnz）分布。 |
| `invalid_values` | 全量 | NaN/Inf 的数量（按值与受影响向量）。非法值不参与数值统计，在其余计算中按 0 处理。 |
| `outliers` | 全量 | 基于稳健修正 z 分数（median/MAD，IQR 兜底）的范数异常向量。 |
| `duplicates` | 混合 | 精确重复（全量，字节级一致），外加基于采样配对近重复指示指标（`距离 ≤ epsilon × 采样距离中位数`）。 |
| `pairwise_distances` | 采样 | 有界随机样本内两两距离的分布。 |
| `nearest_neighbor_distances` | 采样 | 每个采样向量到样本内最近邻的距离。 |
| `anisotropy` | 采样 | 中心化协方差特征值集中度：最大特征值占比、前 10 占比、有效秩、平均两两余弦相似度。 |
| `clustering` | 采样 | 确定性 PCA-2 投影散点图；可选的带种子 k-means 摘要（`--clusters K`）。 |

## 采样与可复现性

- 所有采样均使用显式 `--seed`（默认 42）与独立 PCG64 生成器；采样方法字符串记录在报告中。
- 样本下标用 Floyd 算法抽取（内存/时间受样本量约束，适配流式扫描），两两距离矩阵取该均匀样本
  的前缀。
- `--sample-size` 约束分布与降维类计算，`--pairwise-sample-size` 约束距离矩阵；报告中同时记录
  请求值与生效值。
- 相同输入、配置、采样规模与随机种子两次运行生成的 JSON 报告逐字节一致（键排序、固定缩进）。
  稀疏文件的分析结果与是否提供 `*_offsets` 无关。

## 隐私与离线行为

- 确定性 JSON/HTML 报告完全离线生成；HTML 内联全部资源，打开时不访问任何外部服务。
- 原始向量既不写入报告，也绝不发送给 AI 服务：AI 请求只包含聚合指标、采样配置、警告与有限的
  非敏感元数据。
- AI 凭据只来自环境变量，仅通过 https 的 `Authorization` 头传输，且绝不写入报告、命令输出或
  错误信息（错误消息会被脱敏）。

## 可选的 AI 辅助解读

AI 分析默认关闭。使用 `--ai`（或设置 `VSAG_DATASET_PROFILER_AI=1`）显式启用，并通过环境变量
配置：

```bash
export VSAG_AI_API_KEY=...      # 访问令牌（兜底：OPENAI_API_KEY）
export VSAG_AI_MODEL=...        # 必填的模型名
export VSAG_AI_BASE_URL=...     # 可选，https 的 OpenAI 兼容端点
python -m dataset_profiler my_dataset.hdf5 --ai
```

AI 收到聚合画像后必须返回结构化分析：`summary`、`findings`、`recommendations`（每条都要引用
报告中真实存在的指标字段，以“假设 + `validation` 验证步骤”的形式表述）、`limitations` 与
`additional_measurements`。分析器在把结果并入报告的 `ai_analysis` 字段前会做校验；格式错误的
响应、虚构的指标引用以及服务失败都不会阻碍报告生成——它们在 JSON 中呈现为
`ai_analysis.status = "error"` 条目，在 HTML 中呈现为醒目警告框，同时在终端打印警告。

### 如何验证 AI 建议

请把每条建议当作假设而非结论：

1. 自行复核指标证据——打开 HTML 报告，核对其引用的指标键及统计范围/样本量。
2. 用 `eval_performance`（见[性能评估工具](eval.md)）配合真实查询集与 ground truth 复现实验；
   AI 解读永远不能替代可复现的基准测试。
3. 任何数据变换（归一化、旋转、维度剪枝）之后重新画像，并对比建议所引用的前后指标键。

## 测试与参考

运行工具的单元测试与集成测试：

```bash
cd tools/dataset_profiler && python -m pytest tests -q
```

设计说明（指标选择、采样内部机制、隐私边界、AI 提示与响应结构）见
`tools/dataset_profiler/TECHNICAL_REPORT.md`。

## 相关文档

- [HDF5 数据集格式](dataset_format.md)
- [性能评估工具](eval.md)
- [索引分析工具](analyze_index.md)
- [FAQ：数据分布如何影响向量检索](faq.md)
