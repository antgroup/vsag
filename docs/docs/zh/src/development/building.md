# 编译构建

VSAG 是一个 C++ 项目，使用 CMake 构建。项目源码使用 C++17 标准编写，请确保你使用的编译器支持 C++17 的语法。我们建议你使用 GCC 9.4.0 或者 Clang 13.0.0 以后的版本，因为这些版本在我们的开发中工作良好。

支持的平台包括 Ubuntu 20.04+、CentOS 7+，以及 Apple Silicon 上的 macOS 14+。
macOS 使用 Xcode Command Line Tools 提供的 Apple Clang 和 Homebrew 依赖；当前支持范围是
arm64 核心 C++ 构建，预编译 C++ 包与 Python wheel 仍以 Linux 为主。首次构建前运行：

```bash
./scripts/deps/install_deps.sh
```

脚本会自动选择 Linux 发行版或 macOS 对应的依赖安装流程。

如果需要启用 Linux `io_uring` 后端，请安装 liburing，并在直接配置 CMake 时添加
`-DENABLE_LIBURING=ON`。默认值为 `OFF`；在非 Linux 平台或未找到 liburing 时，
请求 `uring_io` 的配置会打印一次性告警并回退到 `buffer_io`。

如果需要构建可选的 CUDA 后端，请安装 NVIDIA 工具链（`nvcc` 在 `PATH` 中或设置
`CMAKE_CUDA_COMPILER`），并在直接配置 CMake 时添加 `-DENABLE_CUDA=ON`。默认值为 `OFF`；
关闭时会编译一份导出相同符号的空实现，调用方无条件链接、在运行时判断是否有设备。

在 CMake 配置中，有许多参数和编译目标。为了方便使用，我们将常用的编译目标（或命令）写到了 Makefile 中，以避免记忆各种配置或者从命令行输入大段参数。当存在可用的 `ninja` 且 CMake 支持 Ninja 生成器时，这些目标会优先使用 Ninja；否则会回退到 Unix Makefiles。显式设置的 `CMAKE_GENERATOR` 始终优先，例如可以使用 `make debug CMAKE_GENERATOR='Unix Makefiles'` 直接指定回退生成器。由于 CMake 构建目录与生成器绑定，更换现有构建目录的生成器前应先运行对应的清理目标。这些编译目标（或命令）可以通过在项目根目录运行 `make help` 查看：

```bash
Usage: make <target>

Targets:
help:                    ## Show the help.
##
## ================ development ================
debug:                   ## Build vsag with debug options.
dev:                     ## Build full developer configuration.
test:                    ## Build and run unit tests.
asan:                    ## Build with AddressSanitizer option.
test_asan: asan          ## Run unit tests with AddressSanitizer option.
tsan:                    ## Build with ThreadSanitizer option.
test_tsan: tsan          ## Run unit tests with ThreadSanitizer option.
clean:                   ## Clear build/ directory.
##
## ================ integration ================
fmt:                     ## Format codes.
cov:                     ## Build unit tests with code coverage enabled.
lint:                    ## Check coding styles defined in `.clang-tidy`.
fix-lint:                ## Fix coding style issues in-place via clang-apply-replacements, use it be careful!!!
test_parallel:           ## Run all tests parallel (used in CI).
test_asan_parallel: asan ## Run unit tests parallel with AddressSanitizer option.
test_tsan_parallel: tsan ## Run unit tests parallel with ThreadSanitizer option.
##
## ================ distribution ================
release:                 ## Build vsag with release options.
dist-pre-cxx11-abi:      ## Build vsag with distribution options.
dist-cxx11-abi:          ## Build vsag with distribution options.
dist-libcxx:             ## Build vsag using libc++.
pyvsag:                  ## Build a specific Python version wheel. Usage: make pyvsag PY_VERSION=3.10
pyvsag-all:              ## Build wheels for all supported versions. Usage: make pyvsag-all
clean-release:           ## Clear build-release/ directory.
install:                 ## Build and install the release version of vsag.
```

## 构建阶段与缓存报告

采集器在干净构建、暖缓存重建和未改动的 no-op 阶段间保留 Ninja 历史。
只有命令成功且可用的 Ninja 数据记录零条构建边时，才验证为 no-op；缺少数据时明确标记为未验证。
JSON 和 Markdown 包含所有类别，包括 datacell 等嵌套生产模块、工具、示例及未分类工作。
多输出边只计一次。累计边耗时包含并行重叠时间，不等于墙钟时间；嵌套 ExternalProject
工作计入父边，而非逐个命令单独计时。

可缓存请求命中率为 `hits / (hits + misses)`。可用的不可缓存原因（包括
`could_not_use_precompiled_header`）单独报告；缺失计数及零分母在 JSON 中为 `null`，
在 Markdown 中为 `n/a`。由于未统计绕过 ccache 的编译器调用，整体缓存覆盖率不可用。
JSON 保留原始计数。报告不会修改 PCH 或缓存正确性设置。

依赖准备计时只覆盖 configure 前的源码准备和压缩包恢复，不代表全部依赖工作；后续下载、
配置、编译和安装位于配置或构建阶段。`/usr/bin/time -v` 的 Peak RSS 是被计时命令及其
等待子进程所报告的最大驻留集大小，并非所有同时运行构建进程的内存总和。

## 依赖大小指标

构建指标 JSON 使用 schema 版本 3。每个依赖的 `local_bytes`（Markdown 报告中的
**Local size**）是文件大小的合计，而不是实际分配的磁盘空间。它替代了 `source_bytes` 和
`build_bytes`：HDF5 和 OpenBLAS 在源码目录内构建，ANTLR4 的二进制目录也位于源码目录内，
因此无法可靠地将源码和构建产物分开统计。

对于 ExternalProject 依赖，合计值将整个依赖前缀目录统计一次，包含源码、构建产物、安装文件及
前缀目录内的元数据。共享 `BUILD_INFO_DIR`（默认为 `.vsag-build-info`）中的临时文件、
stamp、日志和元数据不计入 `local_bytes`，也不计入准备阶段表格的 `external_build_bytes`；
这两项均不代表 ExternalProject 的全部存储。对于 FetchContent 依赖，合计同级的 `<name>-src` 和 `<name>-build` 目录。
不统计符号链接项。单独缓存的 ExternalProject 归档在准备阶段表格中报告，不计入 `local_bytes`。
系统依赖显示为零，因为不测量主机上的安装文件；未分类或缺失的目录也贡献零字节。这是被测本地目录
的快照，既不是纯源码大小，也不是单独的构建产物大小。
## PR ASan 构建与独立构建基准

> **[ #2899 ](https://github.com/antgroup/vsag/pull/2899) 修复前，旧版 no-op 测量不可信。** 独立工作流会对缺少受支持 schema 3 `true_noop` 报告能力的 collector 明确预检失败。拆分 PR 构建本身并不修复 collector；前置 PR 尚未合并时，定时任务会失败，而不是发布误导性的旧版基线。

PR CI 的 **ASan Build X86** 只运行一次 `make asan COMPILE_JOBS=3`，保留 Ninja、compile commands、system OpenBLAS、examples 和 tools。Makefile 仍启用 tests、Sanitize 和 ASan/UBSan，changed-source lint、下游测试、兼容性检查及构建产物契约保持不变。`ccache --zero-stats` 只重置统计，**不删除缓存对象**。构建步骤记录耗时，always 步骤输出 verbose/兼容回退缓存诊断，不掩盖真实构建失败；ccache 仍启用并使用一次性 GitHub-hosted VM 上的默认目录，不跨 job 恢复或保存编译缓存。PR X86 的依赖归档缓存有意从恢复并保存改为**只恢复**（`actions/cache/restore@v4`），防止 PR 上下文将归档写回共享缓存；这项策略并不只适用于独立 benchmark。

独立的 **Build Performance Benchmark**（`.github/workflows/build_performance.yml`）每周一 UTC 03:23 测 upstream `main`，也可通过 Actions 手动运行。它不是 PR required check，使用独立 concurrency、Ubuntu 22.04 hosted runner、180 分钟超时和 14 天报告保留期。工作流进入默认分支后，可运行：

```bash
gh workflow run build_performance.yml --ref main -f ref=main
# 或指定本仓库 main 历史中可达的完整 40 位 commit SHA：
gh workflow run build_performance.yml --ref main -f ref=<full-main-history-sha>
```

工作流本身必须从 `main` dispatch（其他工作流分支会被跳过），并显式 checkout `antgroup/vsag` upstream `main`，只接受 `main` 或该已获取 upstream 历史中的完整 SHA，并将实际 checkout 的 SHA（不是 dispatch 事件 SHA）及 main 基线写入 `target.txt`。PR ref、任意分支和未合并功能提交均被有意拒绝，因此迁移草稿分支无法通过此入口测量自己；hosted dispatch 和端到端验收需等工作流及受支持 collector 进入 main 后完成。fork 的定时任务被禁用。

安全边界采取保守策略：contents 只读权限、不使用业务 secrets、不保留 checkout 凭据，通过环境变量传递并引用输入，校验 SHA 及祖先关系；Ubuntu 安装使用工作流自身固定的包列表，而不以特权运行目标 ref 的安装脚本。包列表与 PR 的 OpenBLAS-only host 准备一致，后续应保持同步。现有 composite action checkout 固定版本的依赖源码；下载归档缓存只恢复、**不保存**，编译缓存完全不跨运行恢复或保存。全新 benchmark job 独占 `build/` 和 `.benchmark-ccache`，`--clear-ccache` 不触碰普通 PR 缓存；保留默认 `build/` 以兼容现有 collector/Makefile API。

“Cold”表示全新构建和空的**编译缓存**，不意味着网络或依赖下载也是冷启动。保持 collector 原有 CLI，流程为 configure → cold build → clean → warm ccache rebuild → no-op，warm 与 no-op 之间不再 clean。报告保留 JSON、Markdown、分阶段日志、峰值 RSS、Ninja 统计和 ccache 细项；`toolchain.txt` 记录工具版本和工作流实际环境变量值，`collector-command.txt` 记录工作流执行的、经过 shell 转义的 collector 调用命令。底层构建命令/选项以 collector 的分阶段日志为准，其报告保留依赖准备耗时与缓存元数据。compiler-cache key 只是说明性元数据，不证明恢复过缓存。并行累计 edge 时间不是墙钟时间，命中率不覆盖 bypass/PCH/link 工作；缺失遥测不能当作零。

预检不执行 collector，只检查已知 schema 3 能力，不证明测量逻辑本身正确。采集结束后还要求成功状态、Ninja 遥测可用、`true_noop: true` 和零 build edges。#2899 在 no-op 为 false 时可以只报告而不返回失败，因此此工作流会额外使该运行失败，并保留原始报告供诊断。残余生成/检查 edge 需要结合日志和 collector 的 build-edge 分类解释。未来未知 schema 也会保守失败，等待审核后适配。这是以暂时的红灯/缺失基线换取避免错误标记的权衡，不应绕过预检恢复旧报告。#2899 合并后需重新核验最终 API 及测试，包括日志缺失、历史重写、多输出 edge、源文件/头文件修改后的重建。本迁移不修改 collector、PCH、编译选项或持久化缓存。

## 编译 VSAG 库

`make debug` 是我们开发中最常用的命令，它会以开发模式编译整个项目，禁用大多数优化（`-O0`）并生成调试信息（`-g`）。该目标默认关闭测试、示例、工具和 Python 绑定；如需同时启用它们，可使用 `make dev`。

在默认设置下，开发模式的编译产物会生成在 `./build/` 目录中。可以通过如下命令运行单元测试：

```bash
./build/tests/unittests
```

以及通过如下命令运行功能测试：

```bash
./build/tests/functests
```

## 运行测试用例

除了上面提到的方法——编译后手动运行测试用例，VSAG 还支持用一条命令完成编译和运行所有测试：

```bash
make test
```

在我们的开发工作流中，代码修改完成后需要使用上述命令通过所有测试后，才会提交到 GitHub 仓库中。

## 内存和多线程测试

VSAG 是一个索引库，有大量的内存分配和并行计算的代码。我们依赖 AddressSanitizer 和 ThreadSanitizer 来检查发现内存和多线程的问题。当你在开发过程中遇到可疑的内存问题或者多线程问题，可以使用 `make test_asan` 或者 `make test_tsan` 来帮助问题发现。

## 清除编译工作区

当你在调试第三方库引入，或者 CMake options 时，可能会遇到明明修改了 cmake 文件却没有变化的问题，不妨试试 `make clean` 指令。它会清除掉 `build/` 目录的所有内容，然后你就可以像刚下载的新项目一样从头编译了。

## 格式化代码

我们使用 clang-format 工具来保持代码风格的统一，对应的配置文件路径是 `vsag/.clang-format`。

`make fmt` 命令会自动将 VSAG 的源代码格式化。这个命令需要你的环境中安装有 [clang-format](https://clang.llvm.org/docs/ClangFormat.html)。GitHub CI 会在每一个 Pull Request 中运行代码风格检查，以保证合并进主分支的代码风格一致。

## 代码覆盖率统计

`make cov` 会使用 coverage 参数来编译 VSAG 项目，使得测试用例运行后能够得到代码覆盖率统计文件。

## 静态代码分析

VSAG 使用 [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) 工具来实现静态代码分析，旨在提前暴露一些编程规范上的问题，对应的配置文件路径是 `vsag/.clang-tidy`。

使用 `make lint` 可以在本地执行静态代码分析任务。同样地，可以使用 `make fix-lint` 来自动完成代码修复。

> 需要注意的是，fix-lint 命令会在源文件上直接修改，请确定你希望这样做！

## 编译发布模式

在生产环境中，我们需要使用发布模式的 VSAG 库。在此模式下，编译器会尽可能优化代码生成，以实现更好的运行性能。使用以下命令生成发布模式的 VSAG 库：

```bash
make release
```

为了和开发模式的产物区分开，发布模式的产物默认生成在 `./build-release/` 目录中。

如果你需要分发预编译产物，可使用以下目标以控制 ABI：

- `make dist-pre-cxx11-abi`：使用 `-D_GLIBCXX_USE_CXX11_ABI=0` 构建（pre-C++11 ABI）；
- `make dist-cxx11-abi`：使用 `-D_GLIBCXX_USE_CXX11_ABI=1` 构建（C++11 ABI）；
- `make dist-libcxx`：使用 `libc++` 代替 `libstdc++` 构建。

## 编译 pyvsag 包

pyvsag 是 VSAG 的 Python 版本。通过 `pip install pyvsag` 下载安装的 wheel 包就是通过 `make pyvsag` 命令构建出来的。

默认会使用 `PY_VERSION=3.10`，你可以显式指定目标 Python 版本：

```bash
make pyvsag PY_VERSION=3.11
```

或者一次构建所有受支持版本的 wheel：

```bash
make pyvsag-all
```

## 环境变量

在 Makefile 文件的开始可以看到一些 VSAG 编译系统定义的环境变量。这些变量可以通过命令行运行 `export` 命令，或者在 `.bashrc` / `.zshrc` 等 shell 配置文件中设置来修改。

环境变量说明如下：

- `CMAKE_GENERATOR`：显式指定 CMake 使用的生成器；未设置时优先选择可用的 Ninja，否则回退到 Unix Makefiles。其他可选值请参考 [CMake Generators](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html)；
- `CMAKE_INSTALL_PREFIX`：安装路径，即运行 `make install` 后头文件和库文件会被安装到哪里，一般不需要修改；
- `COMPILE_JOBS`：编译并行度，默认是 6 并行编译，建议设置成你的 CPU 核数以提高编译速度；
- `DEBUG_BUILD_DIR`：开发模式产物目录，非必要不修改；
- `RELEASE_BUILD_DIR`：发布模式产物目录，非必要不修改；
- `VSAG_ENABLE_INTEL_MKL`：是否启用 Intel MKL 作为 BLAS 后端，默认 `OFF`；关闭时使用 OpenBLAS；
- `VSAG_ENABLE_LIBAIO`：是否启用 `libaio`，默认 `ON`。

配置和构建目标的 `DEBUG_BUILD_DIR` 应传入普通路径，例如 `make asan DEBUG_BUILD_DIR="custom build"`。命令行引号由 shell 去除，Makefile 在调用 CMake 时会为路径加引号；不要在变量值中嵌入字面引号。

## 离线 / 内网环境构建

VSAG 会在配置 / 构建阶段下载第三方库。在离线或网络受限的环境中，可以设置按依赖的
`VSAG_THIRDPARTY_*` 环境变量，从本地路径或内网镜像（内网 HTTP 服务、OSS 存储桶等）获取每个
压缩包。完整的变量列表与示例见[离线 / 内网环境构建](offline_build.md)。

## 发布流程

如果要在 GitHub 上手动发布 Release，请到 GitHub Actions 页面运行 `Build and Publish Release` 工作流，并填写以下参数：

- `branch`：要发布的分支、tag 或 commit SHA
- `tag_name`：新的发布标签，例如 `v1.0.0`
- `prerelease`：是否标记为预发布版本

如果你想在本地手动执行同样的打包流程，可以运行：

```bash
COMPILE_JOBS=6 bash ./scripts/release/dist.sh
```

如果机器内存足够，可以适当调大 `COMPILE_JOBS`；默认值会比较保守，以避免 CI 里再次触发
内存不足。
