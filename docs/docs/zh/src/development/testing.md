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
`src/` 下的维护中生产代码和 `include/` 下的公共头文件，并排除 `src/` 下的
`version.h`（构建时由 `cmake/GenerateVersionHeader.cmake` 从纳入版本控制的模板
`src/version.h.in` 生成）与引入的兼容头文件 `include/vsag/expected.hpp`。仅在其他平台编译的路径
明确不属于 Linux x86 报告的统计范围，必须由平台专用覆盖率任务统计，不能视为已覆盖。

### 合并前的 SIMD 覆盖率阶段

`PR SIMD Coverage` 仅在目标分支为 `main` 的 PR 修改 SIMD 源码、测试或其构建与覆盖率基础设施时运行。
本阶段有意仅在 `main` 推出。发布分支保留现有的 PR CI，但不包含这项新的 SIMD 覆盖率检查；扩展到发布分支需要单独推进。
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

## 统计 testcase 预算与耗时

`scripts/testing/test_budget_report.py` 只读取已保存的测试列表、结果和成本记录，不运行测试；
不依赖 Team CI 的库存生成工具：

```bash
build/tests/functests '~[daily]' --list-tests --reporter xml > functests-list.xml
build/tests/unittests '~[daily]' --list-tests --reporter xml > unittests-list.xml
python3 scripts/testing/test_budget_report.py \
  --list-tests tests/functests=functests-list.xml \
  --list-tests tests/unittests=unittests-list.xml \
  --source-root "$PWD" --output budget-report.md
```

`--list-tests` 接收 Catch2 列表 XML，而非执行结果 XML。上述列表命令只枚举，不执行测试。
保留所有列出的用例（包括模板展开），报告器不会额外排除 daily。
已有执行结果可加 `--xml tests/functests=report.xml`，同一源码版本的成本日志可加
`--cost tests/functests=cost.log`。列表 XML 的哈希只标识输入列表，不代表测试二进制哈希。

Catch2 XML reporter 会把预算日志放入 `StdOut`。同一份已保存的执行 XML 可以显式导入两类数据：
`--xml` 只读取耗时，`--cost` 先解码 XML 转义，再只读取 `StdOut` 中的 `[test-budget]` 行：

```bash
python3 scripts/testing/test_budget_report.py \
  --list-tests tests/functests=functests-list.xml \
  --xml tests/functests=report.xml --cost tests/functests=report.xml \
  --source-root "$PWD" --output budget-report.md
```

XML 没有预算记录时，成本仍标为 **unknown**；带前缀但格式错误的记录会报错，不会静默丢弃。

已有 Team CI 产物也可提供冻结库存和执行结果：

```bash
python3 scripts/testing/test_budget_report.py \
  --inventory team-plan-RUN-ATTEMPT.zip \
  --result team-results-0-RUN-ATTEMPT.zip \
  --result team-results-1-RUN-ATTEMPT.zip \
  --result team-results-2-RUN-ATTEMPT.zip \
  --result team-results-3-RUN-ATTEMPT.zip \
  --output budget-report.md
```

库存也可使用 schema 3 的 inventory/plan JSON，结果也可直接使用分片 `result.json`。
Catch2 XML 使用 `--xml tests/functests=report.xml`；不要再通过 `--result` 重复输入同一批用例。
`--binary tests/functests` 只统计功能测试，`--group-by family` 按索引族统计，
`--format json` 输出每个用例的分类明细。

计数保留二进制身份：同时编入单元与功能测试二进制的同一源码用例计为两项。
分类按固定优先级互斥地归入 Tune、恢复、序列化、修改/并发、搜索/过滤、构建/训练、
数据/量化/SIMD 或 Other。这是名称、标签和源码位置的启发式分类，不表示断言覆盖率。

可选参数 `--cost tests/functests=cost.jsonl` 接收 schema 1 的完整 testcase 成本记录。
每条包含 `model_version`、`unit`、`policy`、`case`（`name`、`tags`、`source_file`、
`source_line`）、`cap_points` 和 `total_points`。测试日志可为记录添加 `[test-budget]` 前缀；
此时忽略其他日志行。只统计顶层预算上限和估算成本，不重复累加内部 `runs`。Catch generator
重入产生的完全相同计划只计一次，另记录 `identical_plan_repetitions`；同一用例出现不同计划会报错。
桥接记录若标签为空且 `source_line: 0`，必须按二进制、名称、完整源码身份唯一匹配库存。
Catch 原始的绝对路径或含父目录的相对路径只作为身份字符串，报告器不会打开这些源码路径。
优先精确匹配完整路径；只有需要把绝对路径映射为仓库相对身份时，才使用
`--source-root /absolute/checkout`。不允许只按文件名匹配，冻结库存的 ID 和摘要保持不变。

用例数量、预算上限、估算成本、实测 case 秒数分别使用独立分母；不同模型、单位和策略不会混合。
未提供的数据标为 **unknown**，不会当作零；缺少成本记录时仍能生成耗时基线。
case 秒数之和**既不是 CI 墙钟耗时，也不是 CPU 时间**。失败与超预算记录仍会显示；
报告命令不替代 CI 的完整执行检查。

本地脚本回归命令：`python3 tests/scripts/test_test_budget_report.py`。

## 预算策略

`tests/fixtures/core/test_budget.h` 根据 N、D、Q，以及数据准备、训练、构建、搜索、参考答案计算、
序列化和修改操作的声明次数估算逻辑成本。一个 testcase 的完整配置矩阵共用一个预算上限。
每个候选都保留所需配置和合法最小规模；没有合法候选能满足上限时，计划失败。
Catch generator 重入复用同一完整计划，不会重新获得一份预算。`CurrentPolicy` 使用 Catch
`--rng-seed` 的种子，因此同一 testcase 和种子的资源选择可复现。

第一批接入 HGraph、IVF、BruteForce、Warp 和 Dataset 测试中的随机资源族，并未覆盖每个静态用例；
未接入者在报告中仍显示未建模。PR HGraph Tune 保留四类代表转换，Q=20、N≤400；
PiPNN 使用一类转换，Q=10、N=256。Daily 保留全部十类转换，Q=100。

逻辑点数和声明上界不等于墙钟秒数，也不是通用运行时间保证。预算不替代断言或超时；
预算不足不能静默跳过 testcase，也不能把未完成的运行判为通过。

## 参考

- 功能测试源代码目录：`tests/`
- 脚本入口：`Makefile` 中的 `test`、`cov`、`asan` 目标
