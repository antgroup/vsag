# 安装

VSAG 是一个向量检索库，支持在 C++、Python 和 Node.js / TypeScript 程序中使用。核心 C++
库支持 Linux，也支持在 Apple Silicon 上使用 macOS 14 或更高版本从源码构建。完整 CI
仍以 Linux 为主；预编译 C++ 包和 `pyvsag` wheel 当前也主要面向 Linux。

如果使用的是 Python，可以从官方第三方包仓库 PyPI 下载，包名为 [`pyvsag`](https://pypi.org/project/pyvsag/)。`pyvsag` 的版本与源代码版本一一对应，版本功能可以直接参考 [GitHub 发布日志](https://github.com/antgroup/vsag/releases)。Python 包使用 manylinux2014 构建，可以在绝大部分 Linux 环境中运行。通过如下命令获得最新版本：

```bash
pip install pyvsag
```

如果使用的是 Node.js，可以从 npm 直接安装 [`vsag`](https://www.npmjs.com/package/vsag) 包：

```bash
npm install vsag
```

## 下载预编译二进制包

我们为 C++ 用户提供预编译的二进制包，可以在 [GitHub Releases](https://github.com/antgroup/vsag/releases) 中找到。

预编译二进制分为两个版本：

- **旧的 pre-C++11 ABI**：文件名为 `vsag-vX.Y.Z-pre-cxx11-abi.tar.gz`，使用 `-D_GLIBCXX_USE_CXX11_ABI=0` 编译；
- **C++11 ABI**：文件名为 `vsag-vX.Y.Z-cxx11-abi.tar.gz`，使用 `-D_GLIBCXX_USE_CXX11_ABI=1` 编译。

其中 `X.Y.Z` 是版本号。两个版本分别满足不同应用对 ABI 的需求。

## 使用 Docker 镜像

我们也提供了包含完整开发工具链的 Docker 镜像，推荐用于开发和 CI：

```bash
docker pull vsaglib/vsag:ubuntu
```

镜像内的工具版本（clang-format / clang-tidy 等）与项目要求保持一致。

## 从源代码构建

VSAG 可以使用 CMake 从源代码构建，支持 Linux `x86_64` / `aarch64`，以及 macOS
arm64。macOS 当前验证范围是核心 C++ 库的 `make debug`、`make release` 与
`make test`；Python wheel 打包仍以 Linux 为主。

构建依赖：

- 操作系统：
  - **Ubuntu 20.04** 或更高版本
  - 或 **CentOS 7** 或更高版本
  - 或 **macOS 14** 或更高版本（Apple Silicon）
- 编译器：
  - **GCC 9.4.0** 或更高版本
  - 或 **Clang 13.0.0** 或更高版本
  - macOS 使用 Xcode Command Line Tools 提供的 Apple Clang
- 构建工具：
  - **CMake 3.18.0** 或更高版本
  - **clang-format 15**（精确版本，用于代码格式化）
  - **clang-tidy 15**（精确版本，用于静态检查）
- 其他依赖项：
  - Linux：gfortran、OpenMP、libaio、Python 3.6+、curl
  - macOS：Xcode Command Line Tools、Homebrew

依赖项可以通过以下脚本安装：

```bash
# 自动识别当前平台
./scripts/deps/install_deps.sh

# 也可以直接调用平台脚本
./scripts/deps/install_deps_ubuntu.sh  # Debian / Ubuntu
./scripts/deps/install_deps_centos.sh  # CentOS / AliOS
./scripts/deps/install_deps_macos.sh   # macOS
```

如需 Linux `io_uring` 存储后端，请安装 liburing，并在直接配置 CMake 时添加
`-DENABLE_LIBURING=ON`。未启用时，`uring_io` 配置会回退到 `buffer_io`。用法与
限制见[磁盘索引](../resources/disk_index.md#启用-uring_io)。

VSAG 使用 CMake 管理工程，常用构建目标封装在项目根目录的 `Makefile` 中。运行以下命令可以在发布模式下编译并安装：

```bash
make release && make install
```

更多构建选项请参考 [编译构建](../development/building.md)。
