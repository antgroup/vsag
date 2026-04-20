# 标准环境性能参考

本页作为官方性能数据的入口与说明。具体数值建议以 GitHub 最新发布的 benchmark 结果为准，
并通过 [性能评估工具](eval.md) 在目标环境中复测。

## 参考机型

官方基准测试默认在以下量级的机型上进行（具体 SKU 以 Release Notes 为准）：

- **CPU**：主流 x86_64 服务器 CPU（支持 AVX2 / AVX-512）
- **内存**：足够覆盖索引 + 操作系统 page cache 的 DDR4/DDR5
- **磁盘**：NVMe SSD（DiskANN 场景）
- **操作系统**：Ubuntu 20.04 / 22.04 或 CentOS 7 / 8
- **编译**：`make release`，默认启用 Intel MKL（若可用）

## 参考数据集

官方对比常用以下数据集（HDF5 格式，兼容 [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)）：

| 数据集 | 维度 | 距离 | 规模 |
|-------|------|------|------|
| SIFT-1M | 128 | L2 | 1,000,000 |
| GIST-1M | 960 | L2 | 1,000,000 |
| Deep-10M | 96 | L2 | 10,000,000 |
| Text-to-Image-1M | 200 | IP | 1,000,000 |

## 关键指标

- QPS（单线程 / 多线程）
- 平均召回率（Recall@k）
- P50 / P95 / P99 延迟
- 峰值内存、索引体积
- 构建时间

## 如何复现

```bash
make release
./build-release/tools/eval/eval_performance --config tools/eval/eval_template.yaml
```

将输出的 JSON / Markdown 结果与官方对比，可定位性能回归或量化退化。

## 如何贡献你的数据

欢迎通过 PR 向本页面补充"其他机型下的结果"章节；提交时请附：

- 详细 CPU / 内存 / 磁盘信息；
- VSAG 版本（`git rev-parse HEAD`）；
- `eval_performance` 输出（建议使用 JSON + Markdown 两种格式）；
- 构建命令与环境变量（如 `VSAG_ENABLE_INTEL_MKL` 等）。
