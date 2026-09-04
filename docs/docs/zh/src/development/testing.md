# 运行测试

VSAG 采用 [Catch2](https://github.com/catchorg/Catch2) 作为测试框架，测试分为两类：

- **单元测试**：与源码同目录，位于 `src/` 下，聚焦单个类/函数的行为。
- **功能测试**：位于 `tests/` 目录，覆盖跨模块、端到端的索引行为。典型用例包括
  `test_hgraph.cpp`、`test_ivf.cpp`、`test_pyramid.cpp`、`test_sindi.cpp`、
  `test_brute_force.cpp`、`test_memleak.cpp` 等。

## 构建并运行全部测试

`make test` 会以 Debug 配置重新编译（启用 `ENABLE_TESTS=ON`）并运行单元与功能测试：

```bash
make test
```

## 构建并运行单个单元测试模块

为了缩短编辑、构建和测试周期，可以只构建并运行一个仍在维护的单元测试子系统：

```bash
make test-module MODULE=datacell
make test-module MODULE=algorithm CASE='[ut][sample_train_data]'
```

可用模块名为 `simd`、`common`、`algorithm`、`factory`、`attr`、`datacell`、`layout`、
`quantization`、`storage`、`io`、`utils` 和 `impl`。每条命令都会构建共享的生产代码与测试夹具依赖，
但只编译所选模块的测试源码。对应的 CMake 目标和可执行文件名为 `unittests_<module>`：

```bash
cmake -S . -B build -DENABLE_TESTS=ON
cmake --build build --target unittests_datacell
./build/tests/unittests_datacell '[ut][AttributeInvertedInterfaceParameter]'
```

模块目标仅用于加速本地开发。完整验证仍应运行 `make test`；其中聚合的 `unittests` 可执行文件包含所有模块。

说明：

1. 运行 `src/` 下的单元测试；
2. 运行 `tests/` 下的功能测试；
3. `make test` 并未开启覆盖率（`ENABLE_COVERAGE=ON`）。需要覆盖率报告时请使用
   `make cov`：该目标仅完成带覆盖率插桩的编译，随后需要手动运行测试二进制以生成报告。

## 仅运行单个测试二进制

构建完成后，可直接运行单个测试：

```bash
./build-debug/tests/functional_tests "[hgraph]"
./build-debug/tests/functional_tests "[hgraph][concurrent]"
```

Catch2 支持按名字、tag、通配符等方式筛选用例，详见 `--help`。

## 覆盖率

贡献时应保持 `src/` 与 `include/` 下代码的行覆盖率不低于 **90%**。在本地执行：

```bash
make cov
# 然后运行测试二进制以采集覆盖率，例如：
./build-debug/tests/functional_tests
```

报告会输出到 `build-debug/coverage/` 下，可用浏览器打开 `index.html` 查看未覆盖的分支。

## 内存泄漏与多线程

- `test_memleak.cpp`：基于 AddressSanitizer / LeakSanitizer，对索引的构造/销毁路径进行验证。
- `test_index/test_index_concurrent.cpp`：对声明了相应 feature flag 的在维护索引，
  验证共享的并发添加与搜索行为。

## Python 测试

`tests/python/` 包含 `pyvsag` 的 pytest 用例。构建好 `pyvsag` 后：

```bash
make pyvsag PY_VERSION=3.10
cd tests/python && pytest -q
```

## 参考

- 功能测试源代码目录：`tests/`
- 脚本入口：`Makefile` 中的 `test`、`cov`、`asan` 目标
