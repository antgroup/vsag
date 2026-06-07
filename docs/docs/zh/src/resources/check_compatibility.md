# 兼容性检查工具（`check_compatibility`）

`check_compatibility` 用于验证当前 VSAG 构建是否能够加载并搜索旧版本 VSAG 生成的索引文件。
它主要用于 CI，帮助发现序列化格式和向后兼容性回归。

## 构建

使用项目 Makefile 时，通过 `VSAG_ENABLE_TOOLS=ON` 开启工具构建；底层对应的 CMake
选项是 `ENABLE_TOOLS=ON` 与 `ENABLE_CXX11_ABI=ON`：

```bash
VSAG_ENABLE_TOOLS=ON make release
# 产物：./build-release/tools/check_compatibility/check_compatibility
```

如果直接调用 CMake，需要显式传入两个选项：

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TOOLS=ON -DENABLE_CXX11_ABI=ON
cmake --build build-release -j
```

## 输入

命令接收一个形如 `<tag>_<algo_name>` 的位置参数，例如 `v1.0.0_hnsw`。对于该标识，工具会在
`/tmp/` 下查找以下文件：

| 文件 | 用途 |
| --- | --- |
| `/tmp/<tag>_<algo_name>.index` | 旧版本 VSAG 生成的序列化索引 |
| `/tmp/<tag>_<algo_name>_build.json` | 构建该索引时使用的参数 |
| `/tmp/<tag>_<algo_name>_search.json` | 搜索校验使用的参数 |
| `/tmp/random_512d_10K.bin` | 搜索校验使用的测试向量 |

这些文件通常由旧版本兼容性 fixture 生成。

## 使用

```bash
./build-release/tools/check_compatibility/check_compatibility v1.0.0_hnsw
```

工具会创建当前版本的索引实例，反序列化旧索引文件，然后执行一次小规模 KNN 搜索。加载和搜索都成功
时输出 `<identifier> success`；否则输出 `<identifier> failed` 并以非零状态退出。

## 本地入口

`tools/check_compatibility/README_zh.md` 保留为浏览工具目录时的简短入口。
