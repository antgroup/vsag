# VSAG 数据集特征分析工具

`profile_dataset` 将 VSAG 兼容的 HDF5 数据集（稠密或稀疏）分析为带版本号的 JSON 报告和自包含的 HTML 报告，并可对聚合指标进行可选的
AI 辅助解读。规范文档位于网站源码目录。

```bash
VSAG_ENABLE_TOOLS=ON make release
./build-release/tools/profile_dataset/profile_dataset --datapath /tmp/data/sift-128-euclidean.hdf5
```

- 本地英文文档：[docs/docs/en/src/resources/profile_dataset.md](../../docs/docs/en/src/resources/profile_dataset.md)
- 本地中文文档：[docs/docs/zh/src/resources/profile_dataset.md](../../docs/docs/zh/src/resources/profile_dataset.md)
- 技术报告：[docs/docs/zh/src/resources/profile_dataset_design.md](../../docs/docs/zh/src/resources/profile_dataset_design.md)
- 网站：<https://vsag.io/docs/resources/profile_dataset.html>
- 中文网站：<https://vsag.io/docs/zh/resources/profile_dataset.html>

测试：使用 `ENABLE_TESTS=ON` 构建后运行 `./build/tools/profile_dataset/profile_dataset_test`。
