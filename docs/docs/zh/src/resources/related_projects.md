# 关联项目

本页收录与 VSAG 相关或集成了 VSAG 的上下游项目，便于用户快速搭建完整方案。

## 使用 VSAG 的项目

- **[OceanBase](https://github.com/oceanbase/oceanbase)**：蚂蚁集团开源的分布式关系数据库，向量检索能力基于 VSAG。
- **[MyScale / OpenSearch / 其他向量数据库](https://github.com/antgroup/vsag/issues)**：如有集成 PR 或 issue，欢迎补充到本页面。

## VSAG 的依赖与灵感来源

- **[hnswlib](https://github.com/nmslib/hnswlib)**：HNSW 的经典实现，VSAG 中的 HNSW 索引在接口与算法上受其影响。
- **[DiskANN](https://github.com/microsoft/DiskANN)**：微软研究院的大规模磁盘向量检索工作，VSAG 的 `diskann` 索引基于该思路实现。
- **[Faiss](https://github.com/facebookresearch/faiss)**：Meta 的向量检索库；VSAG 在 IVF / 量化思路上有所借鉴。
- **[SPANN / SPTAG](https://github.com/microsoft/SPTAG)**：微软的大规模向量检索工程，提供了混合索引的思路。

## 生态工具

- **[ann-benchmarks](https://github.com/erikbern/ann-benchmarks)**：行业通用的 ANN 基准测试工具，VSAG 自带 [性能评估工具](eval.md) 与其数据集格式兼容。
- **[pybind11](https://github.com/pybind/pybind11)**：`pyvsag` Python 绑定基于此实现。
- **[napi-rs](https://napi.rs/)**：`typescript/` 下的 Node.js 绑定基于此实现。

## 绑定与语言支持

- **C++**（原生）
- **Python**：`pyvsag`，源码位于 `python_bindings/` 与 `python/`。
- **Node.js / TypeScript**：源码位于 `typescript/`，npm 包名 `vsag`。

欢迎提交 PR 完善本列表。
