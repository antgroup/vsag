# VSAG 数据集特征分析工具（`dataset_profiler`）

一个可复现的数据集特征分析与可视化工具：分析并可视化输入向量数据集的**特征**——各维度数值分布、
数值统计、向量范数、稀疏度、重复数据、采样向量间距离、异常值、各向异性以及可选的聚类摘要。
支持 VSAG 评测工具（以及 `pyvsag` 工作流）现有的稠密和稀疏 HDF5 数据格式。

规范文档位于网站文档源码树中。

- 本地中文文档：
  [docs/docs/zh/src/resources/dataset_profiler.md](../../docs/docs/zh/src/resources/dataset_profiler.md)
- 本地英文文档：
  [docs/docs/en/src/resources/dataset_profiler.md](../../docs/docs/en/src/resources/dataset_profiler.md)
- HDF5 数据格式：
  [docs/docs/zh/src/resources/dataset_format.md](../../docs/docs/zh/src/resources/dataset_format.md)
- 中文网站：<https://vsag.io/docs/zh/resources/dataset_profiler.html>
- 英文网站：<https://vsag.io/docs/resources/dataset_profiler.html>
- 技术报告（指标、采样、隐私与 AI 结构说明）：[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- 示例数据与示例报告：[examples/](examples/README.md)

## 快速开始

需要 Python 3.10+，并安装 `numpy` 与 `h5py`（见 [requirements.txt](requirements.txt)），无需 C++
构建。

```bash
cd tools/dataset_profiler
pip install -r requirements.txt

# 生成确定性 JSON 与自包含 HTML（默认写到输入文件旁边）
python -m dataset_profiler /path/to/my_dataset.hdf5 --split train --seed 42

# 限定采样规模、指定距离度量、并对 PCA-2 投影做 k-means
python -m dataset_profiler my_dataset.hdf5 \
    --sample-size 512 --pairwise-sample-size 256 \
    --metric euclidean --clusters 8 \
    --out-json my_dataset.profile.json --out-html my_dataset.profile.html
```

重新生成本目录自带的演示数据与报告：

```bash
python examples/make_toy_datasets.py
```

## AI 辅助解读（可选）

确定性报告的生成完全不需要网络。可选的 AI 层只基于**聚合指标**（绝不发送原始向量）生成解读，
并引用指标字段作为依据。通过 `--ai` 显式启用，并且完全通过环境变量配置：

```bash
export VSAG_AI_API_KEY=...        # 访问令牌；绝不写入报告或日志
export VSAG_AI_MODEL=...          # 模型名（OpenAI 兼容接口）
export VSAG_AI_BASE_URL=...       # 可选，仅允许 https
python -m dataset_profiler my_dataset.hdf5 --ai
```

AI 服务不可用时，JSON/HTML 报告仍会正常生成，并在终端给出明确警告。AI 输出会经过校验：每项
发现/建议必须引用报告中真实存在的指标字段，且建议以“待通过 `tools/eval` 基准验证的假设”的形式
呈现。提示与响应结构见 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)。

## 测试

```bash
pip install pytest
python -m pytest tests -q
```
