# 版本日志

VSAG 网站的版本日志按 `MAJOR.MINOR` 系列维护。
每个系列页面覆盖首个发布版本以及该系列后续的全部补丁版本。
GitHub Releases 仍是逐补丁 PR 清单、发布产物和贡献者名单的完整来源。

## 版本系列

- [VSAG 1.0](release_notes/v1.0.md)
  - 首个版本：[v1.0.0](https://github.com/antgroup/vsag/releases/tag/v1.0.0)，
    2026 年 7 月 12 日
  - 最新补丁版本：`v1.0.0`
  - 状态：稳定版本

后续版本沿用相同结构，例如 `v1.1`、`v1.2`、`v2.0`。
补丁版本直接更新所属系列页面，不再为每个补丁单独创建网站页面。

## 版本与日志归档方式

Release tag 使用 `vMAJOR.MINOR.PATCH` 形式。
网站按 `MAJOR.MINOR` 聚合，便于在一个页面说明整个版本系列；
GitHub Releases 则记录每个 tag 的准确内容。

## 如何获取特定版本

### C++ / 源码

```bash
git checkout vX.Y.Z
make release
```

### Python

先在 [PyPI](https://pypi.org/project/pyvsag/) 查看可用的绑定版本，再安装对应的精确版本：

```bash
pip install pyvsag==X.Y.Z
```

绑定版本不一定与每个 C++ core tag 对应。
仓库还包含 C 和 Node.js/TypeScript 绑定。
各绑定的具体支持范围与打包状态，
请以对应版本系列页面和仓库示例为准。

## 升级建议

- 升级前先阅读目标版本系列的兼容性说明；
- 序列化格式发生变化时，先在测试环境使用
  [兼容性检查工具](check_compatibility.md)验证旧索引产物；
- 生产环境灰度升级，并使用
  [性能评估工具](eval.md)对比召回、延迟和资源消耗。

完整的逐补丁发布历史请查看
[GitHub 上的全部 VSAG Releases](https://github.com/antgroup/vsag/releases)。
