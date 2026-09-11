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

说明：

1. 运行 `src/` 下的单元测试；
2. 运行 `tests/` 下的功能测试；
3. `make test` 并未开启覆盖率（`ENABLE_COVERAGE=ON`）。需要覆盖率报告时请使用
   `make cov`：该目标仅完成带覆盖率插桩的编译，随后按下文命令以固定随机种子运行
   与覆盖率 CI 相同的非 daily 单元测试和功能测试并生成报告。

## 仅运行单个测试二进制

构建完成后，可直接运行单个测试：

```bash
./build/tests/functests "[hgraph]"
./build/tests/functests "[hgraph][concurrent]"
```

Catch2 支持按名字、tag、通配符等方式筛选用例，详见 `--help`。

## 覆盖率

贡献时应保持 `src/` 与 `include/` 下代码的行覆盖率不低于 **90%**。在本地执行：

```bash
make cov
VSAG_TEST_SEED=424242 bash scripts/testing/test_parallel_bg.sh
bash scripts/coverage/collect_cpp_coverage.sh
bash scripts/coverage/check_cov.sh
```

采集脚本会生成包含分支数据和仓库相对路径的 `coverage/coverage.info`。统计范围仅包括
`src/` 下的维护中生产代码和 `include/` 下的公共头文件，并排除构建时生成的
`src/version.h` 与引入的兼容头文件 `include/vsag/expected.hpp`。仅在其他平台编译的路径
明确不属于 Linux x86 报告的统计范围，必须由平台专用覆盖率任务统计，不能视为已覆盖。

### 合并前的 SIMD 覆盖率阶段

`PR SIMD Coverage` 在 SIMD 源码、测试及其覆盖率基础设施发生变更时运行。
任务在同一台 Linux x86 运行器上依次构建 PR 的基准版本和 GitHub 合并候选版本，
使用四个并行构建任务、OpenBLAS、`[simd]~[!benchmark]` 测试筛选条件、
Catch2 随机种子 `424242` 和字典序执行顺序。每次测试限时 15 分钟，整个任务限时 90 分钟。
此阶段仍需构建聚合测试可执行文件。合并后及手动触发的完整单元测试与功能测试覆盖率流程保持不变。

Actions 摘要和产物**仅报告维护中的 `src/simd/` 代码行覆盖率**。
纳入测量的新增可执行代码行覆盖率必须达到 80%；该范围的行覆盖率不得低于本次重新测量的基准，
比较时使用未经四舍五入的比例。新文件计入候选版本及补丁的总量，不虚构文件级基准。
覆盖率数据缺失、SHA 或测试配置不匹配、变更文件未被测量，或未删除源码的记录消失，都会导致失败。
没有测量到新增可执行代码行时，结果为 N/A，而不是 100%。

这是范围有限的第一阶段，尚未提供全项目覆盖率回退保护。任务运行时会列出其他已变更但不受支持的 C++ 路径；
未涉及触发路径的 PR 不会获得此项测量。条件编译路径仍受运行器平台限制。部分测试使用 `random_device`，
因此 Catch2 种子无法控制所有输入。局部覆盖率数据不会上传到 Codecov，也不会与完整测试套件的覆盖率比较。
任务仅使用 PR 的只读权限，不使用密钥、不持久化检出凭据，也不恢复或保存构建缓存。
依赖定义发生变更时会报告无法比较，因为依赖固定版本不同时，两次构建无法安全地共用一份预先准备的依赖源码。

如需比较已收集的数据，将 `base.info`、`head.info` 和 JSON 元数据文件
`base.json`、`head.json` 放入报告目录。每份元数据须包含 `sha` 和 `profile`，
其中 `profile` 必须与 `scripts/coverage/compare_simd_coverage.py` 中的 `PROFILE` 常量完全一致。然后运行：

```bash
python3 scripts/coverage/compare_simd_coverage.py --root . \
  --base "$BASE_SHA" --head "$MERGE_SHA" --reports simd-coverage
```

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
