# 版本日志

本页是 VSAG 1.x 系列的主变更日志。1.0 之前（0.15 / 0.16 / 0.18 系列）
的历史请见 [GitHub Releases](https://github.com/antgroup/vsag/releases)。

VSAG 遵循 [Semantic Versioning 2.0](https://semver.org/)：

- `MAJOR.MINOR.PATCH`
- `MAJOR` 通常伴随 API 或序列化格式的不兼容修改；
- `MINOR` 新增功能但保持向后兼容；
- `PATCH` 仅包含缺陷修复与性能改进。

1.x 系列将遵守的兼容性合约会在独立的 *API 稳定性* 页面中描述（作为后续
PR 跟踪于 [#2069](https://github.com/antgroup/vsag/issues/2069)）。
从 0.18 升级，请先阅读 [升级到 VSAG 1.0](migration_to_1_0.md)，所有不
兼容变更都集中说明在那里。

---

## VSAG 1.0.0 — *目标发布：2026 年，具体日期 TBD*

VSAG 1.0 是首个稳定大版本。它锁定了对外公开的 C++/Python/Node.js API、
索引序列化格式，以及支持的索引族，让 1.x 后续版本可以在不破坏已有代码
的前提下持续迭代。

### 亮点

- **两类生产可用的索引族** —— `hgraph` 用于图检索，`ivf` 用于倒排索引。
  两者均覆盖纯内存与内存+磁盘混合检索模式。旧的 `hnsw` 和 `diskann`
  索引已弃用，详见 [升级到 VSAG 1.0](migration_to_1_0.md)。
- **完整量化方案** —— RabitQ（BQ）用于极致压缩、PQ 用于灵活压缩比、
  SQ4 / SQ8 提供标准量化与小幅召回损失。所有量化器都可与 HGraph 或
  IVF 组合。
- **原生支持非 FP32 输入** —— INT8、BF16、FP16 与稀疏向量作为一级输入
  类型，不再依赖 FP32 仿真。
- **多平台 SIMD** —— x86_64（SSE / AVX / AVX2 / AVX-512 / AMX）与 ARM
  （NEON / SVE）后端，以及可选的 Intel MKL、OpenBLAS 矩阵核。
- **租户级资源隔离** —— per-index allocator 与可注入的线程池，使得在
  同一进程内承载多租户成为现实可行的方案。
- **统一的检索 API** —— `Index::SearchWithRequest(SearchRequest)` 取代
  弃用的 `KnnSearch(query, k, SearchParam&)`，原生支持 per-search
  allocator 与 reasoning。
- **稳定的对外头文件** —— `include/vsag/` 下每个头文件保证自包含；
  1.x 系列内的小版本不会悄悄改变对外 ABI。

### 索引

- **HGraph** —— 大多数场景下推荐使用的图索引。
  - 支持反向边、可选的重复距离阈值，以及用于精细延迟控制的
    `hops_limit` 检索参数；
  - 图索引支持 `Remove`（mark-remove 配合带超时的 `ShrinkAndRepair`
    回收）;
  - 内置 `Train` API 与 ODescent 离线构图，详见
    [Build and Train](../advanced/build_and_train.md)；
  - Reasoning 诊断：通过 `QueryContext` 收集每次检索的访问节点、跳数、
    距离计算次数等，不影响检索结果格式。
- **IVF** —— 推荐用于批量 / 大 K 检索的倒排索引。支持与 HGraph 相同的
  量化器集合，并与 per-search allocator 集成。
- **SINDI** —— 稀疏倒排索引，内置稀疏词表的 term ID 重映射、向量更新
  与 analyzer 钩子。
- **Pyramid** —— 分层倒排索引，支持去重、静态优化、基类
  `IndexSearchParameter` 上的 `topk_factor` 参数，以及
  `PyramidAnalyzer` 统计工具。
- **BruteForce** —— 精确基线，支持并行 range search。
- **WARP** —— 多向量暴力检索后端，已迁移到新的 MultiVectors API。

### 量化

- **RabitQ（BQ）** 支持 extend-bit 与 split-base reorder，并配套独立的
  SIMD 实现;
- **PQ / SQ4 / SQ8** 作为标准的内存 / 召回权衡;
- **Scalar quantizer** 加固了 NaN 编码场景;
- **Quantization Transform** 高级页面完整描述了量化流水线，详见
  [Quantization Transform](../advanced/quantization_transform.md)。

### 数据类型与数据集

- **FP32 / INT8 / BF16 / FP16** 向量输入作为一级类型;
- **稀疏向量** 端到端支持（SINDI + `pyvsag` 的稀疏 HDF5 helper）;
- **MultiVector 数据集** 作为一级类型；评测工具和 WARP 均直接消费新的
  MultiVectors API;
- **`extra_info`** 可与向量一并存储，详见 HGraph 的 `extra_info`
  使用指南。

### 检索 API

- 新的 `SearchRequest` / `Index::SearchWithRequest` 作为主检索入口。
  query 数据集、k、可选 filter、reasoning 钩子、per-search allocator
  统一封装在一个结构体里，热路径不再混用位置参数与 out 参数。
- `SearchParam` 与旧的 `KnnSearch(query, k, SearchParam&)` 仍然可用，
  但已标记 `[[deprecated]]`。完整对照见
  [升级到 VSAG 1.0](migration_to_1_0.md)。
- `CalDistanceById`（批量接口）正在改名为 `CalcDistancesById`，返回值
  语义统一；旧名作为 wrapper 保留。见
  [按 ID 计算距离](../advanced/calc_distance_by_id.md) 与
  Issue [#2068](https://github.com/antgroup/vsag/issues/2068)。
- Range search 变体（带半径语义的 `SearchWithRequest`）在 HGraph、
  IVF、BruteForce 上均可用。

### 平台与打包

- **x86_64 SIMD：** SSE、AVX、AVX2、AVX-512，并新增 AMX 后端（SQ8U
  INT8 IP，以及 KMeans 用的 BF16 GEMM）;
- **ARM SIMD：** NEON 与 SVE;
- **macOS（Darwin）** 作为受支持的构建平台;
- **Intel MKL** 改为可选（`VSAG_ENABLE_INTEL_MKL=OFF` / CMake
  `ENABLE_INTEL_MKL=OFF` 默认关闭）;
- **OpenBLAS** 可从系统链接，而非使用内置副本
  （`VSAG_ENABLE_SYSTEM_OPENBLAS=ON`）;
- 第三方依赖下载支持自定义镜像地址，方便无法直连 GitHub 的环境。

### 资源隔离与可观测性

- **Per-index allocator** —— 通过 `IndexCommonParam` 注入自定义
  `Allocator`，该索引下所有容器都会沿用；
- **可注入线程池** —— 构建与检索都可以使用业务自带的线程池；
- **Per-search allocator** —— 详见
  [Per-Search Allocator](../advanced/search_allocator.md);
- **检索统计** —— `io_cnt`、`io_time_ms` 等计数器通过 `SearchRequest`
  reasoning 暴露;
- **内存与诊断** —— 详见 [Memory](../advanced/memory.md) 与
  [Index Introspection](../advanced/introspection.md);
- **索引生命周期** —— [Index Lifecycle Management](../advanced/index_lifecycle.md)
  描述了在线情况下
  add / remove / mark-remove / rebuild 的安全做法。

### 工具与生态

- **`pyvsag`** Python 绑定已扩展到完整的索引接口，包含稀疏 HDF5
  helper 与 pyramid 导出;
- **Node.js / TypeScript 绑定** —— `vsag` npm 包，配套
  `examples/typescript/` 快速上手示例;
- **`eval_performance`** 工具支持多向量数据集与可配置的 query 数量;
- **HTTP monitor 服务** 基于 `cpp-httplib`，对外暴露在线索引指标。

### 不兼容变更（相对 0.18）

完整列表与代码 diff 见 [升级到 VSAG 1.0](migration_to_1_0.md)。这里
只列要点：

1. `hnsw`、`diskann` 索引类型已弃用，分别迁移到 `hgraph`（或内存+磁盘
   混合配置）与 `ivf`。
2. `SearchParam` 与 `Index::KnnSearch(query, k, SearchParam&)` 已弃用，
   请改用 `SearchRequest` / `Index::SearchWithRequest(SearchRequest)`。
3. `CalDistanceById`（批量）对非法 ID 返回 `-1`，并正在改名为
   `CalcDistancesById`，旧名再保留一个小版本周期。
4. `VSAG_ENABLE_INTEL_MKL` 默认 `OFF`，如果之前依赖 MKL，请显式开启。
5. 多个 HGraph 默认值发生变化（`max_degree=64`、`ef_construction=400`、
   `graph_type="nsw"`）;`support_remove`、`support_duplicate`、
   `store_raw_vector` 默认关闭。

序列化方面：0.18 序列化文件**不保证**能在 1.0 上反序列化；建议在新版本
重建索引。详见 *Migration*。

### 已知问题

- *将在 1.0 RC 阶段补充。*

### 致谢

VSAG 1.0 是蚂蚁集团 VSAG 团队与开源社区共同贡献的成果。逐版本完整
贡献者名单仍维护在
[GitHub Releases](https://github.com/antgroup/vsag/releases) 页面。

---

## 如何获取对应版本

### C++ / 源码

```bash
git checkout v1.0.0
make release
```

### Python

```bash
pip install pyvsag==1.0.0
```

### Node.js / TypeScript

```bash
npm install vsag@1.0.0
```

## 升级建议

- 从任意 0.x 版本升级前，请先阅读
  [升级到 VSAG 1.0](migration_to_1_0.md);
- 跨大版本升级前，请阅读对应 Release 的 **Breaking Changes** 部分;
- 涉及序列化格式变更时，建议先在测试环境验证反序列化兼容性;
- 生产环境灰度升级，结合 [性能评估工具](eval.md) 对比召回与延迟。

## 参考

- [升级到 VSAG 1.0](migration_to_1_0.md)
- [路线图](roadmap_2025.md)
- [最佳实践](best_practices.md)
- [性能](performance.md)
