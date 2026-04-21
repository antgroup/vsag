# 安装

VSAG 是一个向量检索库，支持在 C++、Python 和 Node.js / TypeScript 程序中使用。VSAG 核心库使用 C++ 编写，由于依赖的部分第三方库是 Linux 特有的，当前 VSAG **仅支持在 Linux 系统上运行**。

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

VSAG 可以使用 CMake 从源代码构建，支持 `x86_64` 和 `aarch64` 架构的 Linux 环境，包括在 Apple Silicon 上运行的 Linux 容器。

构建依赖：

- 操作系统：
  - **Ubuntu 20.04** 或更高版本
  - 或 **CentOS 7** 或更高版本
- 编译器：
  - **GCC 9.4.0** 或更高版本
  - 或 **Clang 13.0.0** 或更高版本
- 构建工具：
  - **CMake 3.18.0** 或更高版本
  - **clang-format 15**（精确版本，用于代码格式化）
  - **clang-tidy 15**（精确版本，用于静态检查）
- 其他依赖项：
  - gfortran
  - openmp
  - libaio
  - Python 3.6+
  - curl

依赖项可以通过以下脚本安装：

```bash
# Debian / Ubuntu
./scripts/deps/install_deps_ubuntu.sh

# CentOS / AliOS
./scripts/deps/install_deps_centos.sh
```

VSAG 使用 CMake 管理工程，常用构建目标封装在项目根目录的 `Makefile` 中。运行以下命令可以在发布模式下编译并安装：

```bash
make release && make install
```

更多构建选项请参考 [编译构建](../development/building.md)。
