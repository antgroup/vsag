# 运行测试

VSAG 采用 [Catch2](https://github.com/catchorg/Catch2) 作为测试框架，测试分为两类：

- **单元测试**：与源码同目录，位于 `src/` 下，聚焦单个类/函数的行为。
- **功能测试**：位于 `tests/` 目录，覆盖跨模块、端到端的索引行为。典型用例包括 `test_hnsw.cpp`、
  `test_hgraph.cpp`、`test_diskann.cpp`、`test_ivf.cpp`、`test_pyramid.cpp`、`test_sindi.cpp`、
  `test_brute_force.cpp`、`test_multi_thread.cpp`、`test_memleak.cpp` 等。

## 构建并运行全部测试

推荐先使用 `make debug` 或 `make dev` 构建调试版本，再执行：

```bash
make test
```

该命令会：

1. 运行 `src/` 下的单元测试；
2. 运行 `tests/` 下的功能测试；
3. 收集覆盖率数据（配合 `make cov` 生成 HTML 报告）。

## 仅运行单个测试二进制

构建完成后，可直接运行单个测试：

```bash
./build-debug/tests/functional_tests "[hgraph]"
./build-debug/tests/functional_tests "[hnsw][concurrent]"
```

Catch2 支持按名字、tag、通配符等方式筛选用例，详见 `--help`。

## 覆盖率

贡献时应保持 `src/` 与 `include/` 下代码的行覆盖率不低于 **90%**。在本地执行：

```bash
make cov
```

报告会输出到 `build-debug/coverage/` 下，可用浏览器打开 `index.html` 查看未覆盖的分支。

## 内存泄漏与多线程

- `test_memleak.cpp`：基于 AddressSanitizer / LeakSanitizer，对索引的构造/销毁路径进行验证。
- `test_multi_thread.cpp`：验证并发 `Build` / `KnnSearch` / `RangeSearch` 下的正确性。

## Python 测试

`tests/python/` 包含 `pyvsag` 的 pytest 用例。构建好 `pyvsag` 后：

```bash
make pyvsag PY_VERSION=3.10
cd tests/python && pytest -q
```

## 参考

- 功能测试源代码目录：`tests/`
- 脚本入口：`Makefile` 中的 `test`、`cov`、`asan` 目标
