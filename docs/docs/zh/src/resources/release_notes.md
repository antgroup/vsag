# 版本日志

VSAG 的正式发布历史与变更说明维护在 GitHub Releases 页面：

- [Releases on GitHub](https://github.com/antgroup/vsag/releases)

每个发布版本包含：

- **新增功能**（Features）
- **改进**（Improvements）
- **缺陷修复**（Bug Fixes）
- **不兼容变更**（Breaking Changes，如有）
- **贡献者名单**

## 版本号规范

VSAG 遵循 [Semantic Versioning 2.0](https://semver.org/)：

- `MAJOR.MINOR.PATCH`
- `MAJOR` 通常伴随 API / 序列化格式的不兼容修改；
- `MINOR` 新增功能但保持向后兼容；
- `PATCH` 仅包含缺陷修复与性能改进。

## 如何获取对应版本

### C++ / 源码

```bash
git checkout vX.Y.Z
make release
```

### Python

```bash
pip install pyvsag==X.Y.Z
```

### Node.js / TypeScript

```bash
npm install vsag@X.Y.Z
```

## 升级建议

- 跨大版本升级前，请先阅读对应 Release 的 **Breaking Changes** 部分；
- 涉及序列化格式变更时，建议先在测试环境验证反序列化兼容性；
- 对于 SINDI，旧版 `use_term_lists_heap_insert` 检索参数会被忽略。SINDI 现在根据
  `doc_prune_ratio` 和 `query_prune_ratio` 自动推导堆插入策略；如果配置曾依赖直接指定
  该路径，请改用剪枝比例。
- 生产环境灰度升级，结合 [性能评估工具](eval.md) 对比召回与延迟。
