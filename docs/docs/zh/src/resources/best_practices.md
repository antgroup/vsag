# 最佳实践

本页整理在生产环境使用 VSAG 的经验性建议，作为参数手册与性能调优的补充。

## 索引选型

| 场景 | 推荐索引 | 理由 |
|------|---------|------|
| 中等规模（≤ 1000 万）纯内存、对召回/延迟要求极高 | `hgraph` | 统一的高质量图索引，支持多种量化与 Tune |
| 兼容既有 HNSW 部署 | `hnsw` | 接口与参数最贴近 hnswlib |
| 10 亿级向量、内存受限 | `diskann` | 内存只放 PQ，完整向量在磁盘 |
| 候选召回层 / 粗排 | `ivf` | 训练后即可大规模并行 |
| 小规模、需要 100% 精度 | `brute_force` | 暴力搜索，作为召回率 baseline |
| 多租户 / 分区数据 | `pyramid` | 一个索引内部多棵子图，支持按 tag 检索 |
| 稀疏向量（BM25 / SPLADE 类） | `sindi` | 专为稀疏向量设计 |

详细参数参见 [索引参数](index_parameters.md)。

## 构建阶段

- **先确定 metric**：`l2` / `ip` / `cosine` 不可在构建后变更。
- **`ef_construction`**：典型 200~500。过小召回不足；过大构建显著变慢。
- **`max_degree` / `M`**：典型 16~48。越大召回越高、内存也越高。
- **量化策略**：延迟敏感场景建议 `sq8` 或 `pq`；精度敏感建议 `fp32` 或 `fp16`。
- **并行构建**：使用自定义 `ThreadPool`（见 `examples/cpp/203_custom_thread_pool.cpp`）以控制并发度。

## 搜索阶段

- **`ef_search`**：典型 `topk ~ topk * 10`，可按 QPS / 召回率做 grid search。
- **批量搜索**：多查询合并可提升缓存命中；参考 `examples/cpp/205_*`（若提供）或业务侧批量化。
- **Filter**：使用内置 `Filter`（`examples/cpp/301_feature_filter.cpp`），不要在结果侧二次过滤。
- **临时 Allocator**：高并发在线服务建议每线程一份 arena allocator，见 [内存管理](../advanced/memory.md)。

## 调优

- 使用 [`Tune`](../advanced/optimizer.md) 对真实查询分布进行调参；
- 对尾部困难查询，启用 [共轭图](../advanced/enhance_graph.md)；
- 使用 [`eval_performance`](eval.md) 做持续回归测试。

## 部署

- 推荐使用官方 Docker 镜像，详见 [安装](../guide/installation.md)。
- 生产二进制建议选择对应 ABI 的发布包：`dist-pre-cxx11-abi`、`dist-cxx11-abi`、`dist-libcxx`（见 [编译构建](../development/building.md)）。
- 开启 `VSAG_ENABLE_INTEL_MKL=ON` 可在 Intel CPU 上获得额外加速；
- DiskANN 建议使用 NVMe SSD，并配合 `VSAG_ENABLE_LIBAIO=ON`。

## 可观测

- `Index::GetMemoryUsage()` 暴露运行时内存；
- 搜索路径上可用自定义 `Logger`（`examples/cpp/202_custom_logger.cpp`）接入业务日志；
- 结合 `eval_performance` 将关键指标写入 InfluxDB 进行长期监控。
