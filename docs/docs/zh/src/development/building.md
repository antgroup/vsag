# 编译构建

VSAG 是一个 C++ 项目，使用 CMake 构建。项目源码使用 C++17 标准编写，请确保你使用的编译器支持 C++17 的语法。我们建议你使用 GCC 9.4.0 或者 Clang 13.0.0 以后的版本，因为这些版本在我们的开发中工作良好。

在 CMake 配置中，有许多参数和编译目标。为了方便使用，我们将常用的编译目标（或命令）写到了 Makefile 中，使用 Unix Makefiles 进行管理，已避免记忆各种配置或者从命令行输入大段参数。这些编译目标（或命令）可以通过在项目根目录运行 `make help` 查看：

```bash
Usage: make <target>

Targets:
help:                    ## Show the help.
##
## ================ development ================
debug:                   ## Build vsag with debug options.
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
dist-old-abi:            ## Build vsag with distribution options.
dist-cxx11-abi:          ## Build vsag with distribution options.
dist-libcxx:             ## Build vsag using libc++.
pyvsag:                  ## Build pyvsag wheel.
clean-release:           ## Clear build-release/ directory.
install:                 ## Build and install the release version of vsag.
```

## 编译 VSAG 库

`make debug` 是我们开发中最常用的命令，它会以开发模式编译整个项目，禁用大多数优化（`-O0`），生成调试信息（`-g`）。并且，编译的内容包括了测试用例、pybinds 以及 VSAG 工具。

在默认的设置下，开发模式的编译产物会生成在 `./build/` 目录中。可以通过这样的命令运行单元测试：

```cpp
./build/tests/unittests
```

以及通过这样的命令运行功能测试：

```cpp
./build/tests/functests
```

## 运行测试用例

除了上面提到的方法，编译后手动运行测试用例，VSAG 还支持一条命令完成编译和运行所有测试：

```
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

在生产环境中，我们需要使用的是发布模式的 VSAG 库。在此模式下，编译器会尽可能地优化代码生成，以实现最好的运行性能。使用以下命令生成发布模式的 VSAG 库：

```
make relase
```

为了和开发模式的产物区分开，发布模式的产物默认生成在 `./build-release` 目录中。

## 编译 pyvsag 包

pyvsag 是 VSAG 的 Python 版本。通过 `pip install pyvsag` 下载安装的 wheel 包就是通过 `make pyvsag` 命令构建出来的。

## 环境变量

在 Makefile 文件的开始可以看到一些 VSAG 编译系统定义的环境变量。这些变量可以通过命令行运行 `export` 命令或者 `.bashrc` / `.zshrc` 等文件中的配置修改。

环境变量说明如下：

- `CMAKE_GENERATOR` ：CMake 内部使用什么来编译项目，默认是"Unix Makefiles"，其他可选值请参考：https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html；
- `CMAKE_INSTALL_PREFIX` ：安装路径，即运行 `make install` 后，头文件和库文件会被安装到哪里，一般不需要修改；
- `COMPILE_JOBS` ：编译并行度，默认是 6 并行编译，建议设置成你的 CPU 数以提高编译速度；
- `DEBUG_BUILD_DIR` ：开发模式产物目录，非必要不修改；
- `RELEASE_BUILD_DIR` ：发布模式产物目录，非必要不修改；

