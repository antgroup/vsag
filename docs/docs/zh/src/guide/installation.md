# 安装

VSAG 是一个向量检索库，支持在 C++ 和 Python 程序中使用，从而实现向量检索功能。在使用 VSAG 之前，你需要在系统上安装或者构建 VSAG 库。虽然 VSAG 库本身完全使用 C++ 编写，存在一些第三方库依赖是 Linux 特有的，所以当前 VSAG 库只支持在 Linux 系统上运行。

如果你使用的是 Python，可以通过 Python 语言的官方第三方软件包仓库 PyPI 下载。VSAG 库的 Python 包名叫作 `pyvsag`，pyvsag 的版本与源代码版本对应，版本功能可以直接参考 GitHub 上的发布日志。pyvsag 包使用 manylinux2014 构建，可以在绝大部分 Linux 环境中运行。通过如下命令获得最新版本的 pyvsag 包：

```bash
pip install pyvsag
```

## 下载预编译二进制包

当前，VSAG 的用户大多是在 C++ 程序中使用，我们为此提供预编译的二进制包，可以在 GitHub 版本产物中找到（https://github.com/antgroup/vsag/releases）。

预编译二进制分成两个版本，”旧的 C++11 前 ABI“ 文件名叫 `vsag-vX.Y.Z-old-abi.tar.gz`，和 “新的 C++11 ABI“ 文件名叫 `vsag-vX.Y.Z-cxx11-abi.tar.gz`，其中 X/Y/Z 是版本号。这两个版本分别使用 `-D_GLIBCXX_USE_CXX11_ABI=0` 和 `-D_GLIBCXX_USE_CXX11_ABI=1` 编译构建得到，以满足不同应用程序对于 ABI 的需求。

## 从源代码构建

VSAG 可以使用 CMake 从源代码构建。

VSAG 支持在 x86-64 和 aarch64 架构的 Linux 环境中运行，包括运行在 Apple Silicon 的 Linux 容器。

构建依赖：
- 操作系统：
  - **Ubuntu 20.04** 或更高版本
  - 或 **CentOS 7** 或更高版本
- 编译器：
  - **GCC 9.4.0** 或更高版本
  - 或 **Clang 13.0.0** 或更高版本
- 构建工具：**CMake 3.18.0** 或更高版本
- 其他依赖项：
  - gfortran
  - openmp
  - libaio
  - python 3.6+
  - curl

依赖项可以通过以下脚本安装：

```bash
# for Debian/Ubuntu
./scripts/deps/install_deps_ubuntu.sh

# for CentOS/AliOS
./scripts/deps/install_deps_centos.sh
```

VSAG 使用 CMake 来构建项目，并将常用的构建目标放在 Unix Makefiles 中管理和提供。运行以下命令可以执行编译和安装：

```bash
make release && make install
```
