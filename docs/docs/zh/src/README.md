# VSAG介绍

**VSAG** 全称（Vector Search Algorithm Group）是一个用于相似性检索的库。允许用户在各种规模的向量集合中进行高效搜索，尤其是那些无法完全加载到内存中的集合。同时，VSAG 还提供了基于向量维度和数据规模自动生成参数的方法，使开发者无需了解算法原理即可快速上手使用。VSAG 使用 C++ 编写，并提供一个简单的 Python 包装叫 pyvsag。该项目主要由蚂蚁集团开发。

- 索引 **低内存用量** 可以降低使用成本
- 多平台指令集适配提供 **高性能** 检索
- 支持 bitmap 和 callback 两种 **混合搜索** 方式
- 使用 C++ 编写，并提供 [基于 CMake 的集成方法](https://github.com/antgroup/vsag/blob/main/README.md#integrate-with-cmake)

## Contributing

VSAG 是免费和开源的。你可以在 [GitHub](https://github.com/antgroup/vsag) 上获取到源代码，以及提交错误报告和功能请求到 [GitHub问题跟踪器](https://github.com/antgroup/vsag/issues) 上。VSAG 依靠社区来修复错误和增加功能：如果你想做出贡献，请阅读 [贡献指南](https://github.com/antgroup/vsag/blob/main/CONTRIBUTING.md) 并考虑 [创建合并请求](https://github.com/antgroup/vsag/pulls)。

## License

VSAG 源代码和文档在 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可证下发布。