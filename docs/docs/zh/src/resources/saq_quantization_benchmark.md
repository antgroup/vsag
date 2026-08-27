# SAQ 量化基准工具

`saq_quantization_benchmark` 使用 ann-benchmarks HDF5 数据集直接比较 SAQ 与 multi-bit
RaBitQ，测量模型训练、编码、查询准备加单条距离、四条批量距离、码对距离和重建误差。工具将
完整机器可读结果写入一个 JSON 文件，进度信息写入标准错误。

如需完整、可复现的 SIFT1M/GIST1M 流程，包括 Release 配置、HGraph 构建与检索、环境信息、
校验和及汇总 CSV，请使用
[`benchs/saq` 流程](https://github.com/antgroup/vsag/blob/main/benchs/saq/README.md)。

## 构建

启用工具并构建目标：

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release --target saq_quantization_benchmark -j2
```

## 用法

```text
saq_quantization_benchmark DATASET.hdf5 OUTPUT.json \
    [AVG_BITS=4] [TRAIN_COUNT=10000] [ENCODE_COUNT=100000] \
    [--ablations] [--exact-rabitq]
```

| 参数 | 默认值 | 约束 | 含义 |
| --- | --- | --- | --- |
| `DATASET.hdf5` | 必填 | ann-benchmarks 格式的稠密 FP32 欧氏距离数据集 | 输入基础向量、查询和真值。 |
| `OUTPUT.json` | 必填 | 可写路径 | 完整结构化结果。 |
| `AVG_BITS` | `4` | 整数 `[1, 8]` | SAQ 与 RaBitQ 完整记录等长比较点。 |
| `TRAIN_COUNT` | `10000` | 正整数，不超过基础向量数 | 每个量化器使用的训练前缀；SAQ 内部最多采样 65,536 条。 |
| `ENCODE_COUNT` | `100000` | 正整数，不超过基础向量数 | 编码并参与直接测量的基础向量前缀。 |
| `--ablations` | 关闭 | 可选标志 | 增加 PCA、旋转、调整轮数和固定分段 SAQ 变体。 |
| `--exact-rabitq` | 关闭 | 可选标志 | 增加精确 RaBitQ 编码诊断对照；普通 RaBitQ 行使用生产快速编码器。 |

示例：

```bash
./build-release/tools/saq_quantization_benchmark/saq_quantization_benchmark \
  /data/sift-128-euclidean.hdf5 /tmp/sift1m-saq.json \
  4 65536 1000000 --ablations --exact-rabitq
```

该工具目前只评估 L2。直接吞吐结果属于微基准，不能替代相同召回率下的端到端索引测量；不同
召回率对应的 QPS 不应直接比较。
