# VSAG 介绍

**VSAG** 全称 Vector Search Algorithm Group，是一个用于相似性检索的向量索引库。VSAG 允许用户在各种规模的向量集合中进行高效搜索，包括无法完全放入内存的集合，同时提供基于向量维度和数据规模自动生成参数的能力，使开发者无需深入了解底层算法原理即可快速上手。

VSAG 使用 C++ 编写，并提供：

- Python 包装 [`pyvsag`](https://pypi.org/project/pyvsag/)
- Node.js / TypeScript 绑定 `vsag`（由 `napi-rs` 生成）

该项目由蚂蚁集团发起并主导开发，目前以开源社区的方式维护。

## 主要特性

- **低内存占用**：通过量化（RaBitQ、PQ、SQ4/SQ8）与内存-磁盘混合索引降低使用成本；
- **高性能检索**：针对 x86_64（SSE/AVX/AVX2/AVX512/AMX）和 ARM（Neon/SVE）做了指令集适配；
- **丰富的索引类型**：HGraph、HNSW、DiskANN、IVF、Pyramid、BruteForce、SINDI（稀疏）等；
- **灵活的过滤与混合搜索**：支持 bitmap 与 callback 两种过滤方式，以及混合 `(data vector, attribute)` 查询；
- **易于集成**：提供基于 CMake 的集成方式，详见 [README](https://github.com/antgroup/vsag/blob/main/README.md#integrate-with-cmake)。

## Contributing

VSAG 是免费和开源的。你可以在 [GitHub](https://github.com/antgroup/vsag) 上获取到源代码，以及提交错误报告和功能请求到 [GitHub问题跟踪器](https://github.com/antgroup/vsag/issues) 上。VSAG 依靠社区来修复错误和增加功能：如果你想做出贡献，请阅读 [贡献指南](https://github.com/antgroup/vsag/blob/main/CONTRIBUTING.md) 并考虑 [创建合并请求](https://github.com/antgroup/vsag/pulls)。

## License

VSAG 源代码和文档在 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可证下发布。
