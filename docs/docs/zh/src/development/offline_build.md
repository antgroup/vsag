# 离线 / 内网环境构建

VSAG 在 CMake **配置 / 构建阶段**会下载一批第三方库（通过 `ExternalProject_Add`
与 `FetchContent`）。在没有外网访问、或网络较慢 / 受限的机器上，这些下载可能失败或
超时。本文介绍如何把每个依赖指向**本地路径**或**内网镜像**（内网 HTTP 服务、OSS
存储桶、Artifactory 等），从而在完全离线的环境中完成编译。

## 第三方下载的解析顺序

对每一个需要下载的依赖，VSAG 会构造一个候选 URL *列表*，由 CMake **按顺序**依次尝试，
命中第一个成功的即停止。以 `antlr4` 为代表
（[`extern/antlr4/antlr4.cmake`](https://github.com/antgroup/vsag/blob/main/extern/antlr4/antlr4.cmake)）：

```cmake
set (antlr4_urls
    https://github.com/antlr/antlr4/archive/refs/tags/4.13.2.tar.gz   # 1. 上游
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/antlr4/v4.13.2.tar.gz  # 2. 项目镜像
)
if (DEFINED ENV{VSAG_THIRDPARTY_ANTLR4})
    message (STATUS "Using local path for antlr4: $ENV{VSAG_THIRDPARTY_ANTLR4}")
    list (PREPEND antlr4_urls "$ENV{VSAG_THIRDPARTY_ANTLR4}")   # 0. 你的覆盖项（最先尝试）
endif ()

ExternalProject_Add (antlr4
    URL ${antlr4_urls}
    URL_HASH MD5=3b75610fc8a827119258cba09a068be5
    ...)
```

因此解析顺序为：

1. **`VSAG_THIRDPARTY_<LIB>`** —— 你设置的覆盖项（如果该环境变量已设置为**非空**值）。**最先尝试**。
2. **上游** URL（GitHub / 项目发布页）。
3. 项目维护的 **阿里云 OSS 镜像**
   （`vsagcache.oss-rg-china-mainland.aliyuncs.com`）。该兜底地址始终存在，在中国大陆 /
   弱网环境下很有帮助，但**不可由用户配置**——若要使用纯内网镜像，请使用环境变量。

> **可用版本：** `VSAG_THIRDPARTY_*` 覆盖能力在 `main` 分支以及 `0.15`、`0.16`、
> `0.17`、`0.18` 发布线上均可用——详见[版本可用性](#版本可用性)。

## 开始前的关键事项

- **取值可以是本地路径，也可以是 URL。** 支持绝对文件路径
  （`/data/deps/fmt-10.2.1.tar.gz`）、`file://` URL，或任意 `http(s)://` URL——
  包括内网 HTTP 服务或 OSS / S3 存储桶。
- **依然会校验压缩包哈希。** 每个依赖都声明了 `URL_HASH`（MD5 或 SHA256）。你镜像 /
  本地的压缩包必须与**上游压缩包逐字节一致**，否则 CMake 会因哈希不匹配而中止。最稳妥的
  做法是把上游原始文件下载一次，原封不动地重新托管。
- **覆盖项在配置阶段读取。** 如果你在上一次配置之后修改了变量，请重新执行 CMake 配置或
  先运行 `make clean`，新值才会生效。
- **请使用非空值，否则就不要设置。** CMake 把“已 export 但为空”的变量视为*已定义*，因此
  `export VSAG_THIRDPARTY_FMT=` 会把一个空项 prepend 到 URL 列表里，导致下载失败。若要停用
  某个覆盖项，请 `unset` 它，而不要把它设为空字符串。
- **每个依赖相互独立。** 没有单一的全局镜像变量；每个需要的依赖各自设置一个
  `VSAG_THIRDPARTY_<LIB>`。你只需为本次构建实际拉取的依赖设置变量（见
  [我需要哪些依赖？](#我需要哪些依赖)）。
- **日志中的确认信息。** 覆盖项生效时，CMake 会打印
  `-- Using local path for <lib>: <你的取值>`。

## 环境变量

| 环境变量 | 库 | 需镜像的上游压缩包 | 何时被拉取 |
| --- | --- | --- | --- |
| `VSAG_THIRDPARTY_JSON` | nlohmann/json 3.11.3 | `github.com/nlohmann/json/.../v3.11.3.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_ANTLR4` | ANTLR4 runtime 4.13.2 | `github.com/antlr/antlr4/.../4.13.2.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_BOOST` | Boost 1.67.0（头文件） | `archives.boost.io/.../boost_1_67_0.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_OPENBLAS` | OpenBLAS 0.3.23 | `github.com/OpenMathLib/OpenBLAS/.../OpenBLAS-0.3.23.tar.gz` | 默认 BLAS 后端（未使用系统库 / MKL 时） |
| `VSAG_THIRDPARTY_CPUINFO` | pytorch/cpuinfo | `github.com/pytorch/cpuinfo/archive/ca678952...tar.gz` | 始终 |
| `VSAG_THIRDPARTY_FMT` | fmt 10.2.1 | `github.com/fmtlib/fmt/.../10.2.1.tar.gz` | 始终（除非使用系统 fmt） |
| `VSAG_THIRDPARTY_THREAD_POOL` | log4cplus/ThreadPool | `github.com/log4cplus/ThreadPool/archive/3507796e...tar.gz` | 始终 |
| `VSAG_THIRDPARTY_TSL` | Tessil/robin-map 1.4.0 | `github.com/Tessil/robin-map/.../v1.4.0.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_ROARINGBITMAP` | CRoaring 3.0.1 | `github.com/RoaringBitmap/CRoaring/.../v3.0.1.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_CATCH2` | Catch2 3.7.1 | `github.com/catchorg/Catch2/.../v3.7.1.tar.gz` | `ENABLE_TESTS=ON` |
| `VSAG_THIRDPARTY_HDF5` | HDF5 1.14.4 | `github.com/HDFGroup/hdf5/.../hdf5_1.14.4.tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_ARGPARSE` | p-ranav/argparse 3.1 | `github.com/p-ranav/argparse/.../v3.1.tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_YAML_CPP` | yaml-cpp 0.9.0 | `github.com/jbeder/yaml-cpp/.../yaml-cpp-0.9.0.tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_TABULATE` | p-ranav/tabulate | `github.com/p-ranav/tabulate/archive/3a583010...tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_HTTPLIB` | cpp-httplib 0.35.0 | `github.com/yhirose/cpp-httplib/.../v0.35.0.tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_PYBIND11` | pybind11 2.11.1 | `github.com/pybind/pybind11/.../v2.11.1.tar.gz` | Python 绑定（`pyvsag` / `ENABLE_PYBINDS=ON`） |

> 每个依赖确切的上游 URL **以及**期望的 `URL_HASH`，其唯一权威来源是对应的
> [`extern/<lib>/<lib>.cmake`](https://github.com/antgroup/vsag/tree/main/extern)
> 文件。镜像时（尤其是版本升级后）请以该文件为准。

此处未列出的（不下载，因此无需覆盖）：**Intel MKL**（通过 `find_path` 在主机上查找）与
**DiskANN**（以源码内置于 `extern/diskann/`）。

## 我需要哪些依赖？

你只需镜像本次构建实际会下载的依赖：

- **核心库**（`make debug` / `make release`）：`JSON`、`ANTLR4`、`BOOST`、
  `OPENBLAS`、`CPUINFO`、`FMT`、`THREAD_POOL`、`TSL`、`ROARINGBITMAP`。
  其中两个是条件依赖：当 BLAS 由 Intel MKL（x86_64 且 `ENABLE_INTEL_MKL=ON`）或系统
  OpenBLAS 提供时，`OPENBLAS` **不会**下载；当找到系统 `fmt` 时，`FMT` 会被跳过。
- **+ 测试**（`make test`，`ENABLE_TESTS=ON`）：另加 `CATCH2`。
- **+ 工具**（`ENABLE_TOOLS=ON` **且** `ENABLE_CXX11_ABI=ON`）：另加 `HDF5`、
  `ARGPARSE`、`YAML_CPP`、`TABULATE`、`HTTPLIB`——仅当两个选项同时开启时才会下载
  （见 [`cmake/VSAGThirdParty.cmake`](https://github.com/antgroup/vsag/blob/main/cmake/VSAGThirdParty.cmake)）。
- **+ Python wheel**（`make pyvsag`）：另加 `PYBIND11`。

## 示例

### A. 内网 HTTP 服务或 OSS 存储桶（推荐）

将上游压缩包原封不动地重新托管到内网地址，再让每个变量指向它。用一个基础 URL 的 shell
变量可以让配置更简洁：

```bash
# 内网镜像，逐字节提供上游压缩包
export VSAG_MIRROR=https://mirror.corp.example.com/vsag-thirdparty

export VSAG_THIRDPARTY_JSON=$VSAG_MIRROR/v3.11.3.tar.gz
export VSAG_THIRDPARTY_ANTLR4=$VSAG_MIRROR/antlr4-4.13.2.tar.gz
export VSAG_THIRDPARTY_BOOST=$VSAG_MIRROR/boost_1_67_0.tar.gz
export VSAG_THIRDPARTY_OPENBLAS=$VSAG_MIRROR/OpenBLAS-0.3.23.tar.gz
export VSAG_THIRDPARTY_CPUINFO=$VSAG_MIRROR/cpuinfo-ca678952.tar.gz
export VSAG_THIRDPARTY_FMT=$VSAG_MIRROR/fmt-10.2.1.tar.gz
export VSAG_THIRDPARTY_THREAD_POOL=$VSAG_MIRROR/thread_pool-3507796e.tar.gz
export VSAG_THIRDPARTY_TSL=$VSAG_MIRROR/robin-map-1.4.0.tar.gz
export VSAG_THIRDPARTY_ROARINGBITMAP=$VSAG_MIRROR/CRoaring-3.0.1.tar.gz

make release
```

OSS / S3 存储桶用法完全相同——直接使用其公网（或网络可达）的对象 URL，例如
`https://my-bucket.oss-cn-hangzhou.aliyuncs.com/vsag/OpenBLAS-0.3.23.tar.gz`。

### B. 预先下载的本地文件（完全离线）

在完全没有网络的机器上，先把压缩包拷贝到本机（例如 `/data/vsag-deps`），再让变量指向本地
文件：

```bash
export VSAG_THIRDPARTY_JSON=/data/vsag-deps/v3.11.3.tar.gz
export VSAG_THIRDPARTY_ANTLR4=/data/vsag-deps/antlr4-4.13.2.tar.gz
export VSAG_THIRDPARTY_BOOST=/data/vsag-deps/boost_1_67_0.tar.gz
export VSAG_THIRDPARTY_OPENBLAS=/data/vsag-deps/OpenBLAS-0.3.23.tar.gz
export VSAG_THIRDPARTY_CPUINFO=/data/vsag-deps/cpuinfo-ca678952.tar.gz
export VSAG_THIRDPARTY_FMT=/data/vsag-deps/fmt-10.2.1.tar.gz
export VSAG_THIRDPARTY_THREAD_POOL=/data/vsag-deps/thread_pool-3507796e.tar.gz
export VSAG_THIRDPARTY_TSL=/data/vsag-deps/robin-map-1.4.0.tar.gz
export VSAG_THIRDPARTY_ROARINGBITMAP=/data/vsag-deps/CRoaring-3.0.1.tar.gz

make release
```

使用 `file://` URL（`export VSAG_THIRDPARTY_FMT=file:///data/vsag-deps/fmt-10.2.1.tar.gz`）
同样有效。

### C. 只覆盖单个依赖

如果只有某一个下载不稳定，只覆盖它即可，其余继续使用默认地址：

```bash
export VSAG_THIRDPARTY_OPENBLAS=https://mirror.corp.example.com/OpenBLAS-0.3.23.tar.gz
make release
```

## 备选方案：复用系统库

对于主机上已安装的依赖，你可以直接跳过下载，而不必镜像。设置
`VSAG_USE_SYSTEM_DEPS=ON`（或按依赖设置 `VSAG_USE_SYSTEM_<DEP>=ON`）。当前支持系统复用的
依赖列表见
[`DEVELOPMENT.md`](https://github.com/antgroup/vsag/blob/main/DEVELOPMENT.md#system-third-party-dependencies)。

## 常见问题

- **哈希不匹配 / 出现 “HASH mismatch” 错误** —— 你镜像或本地的压缩包与上游文件不是逐字节
  一致。请重新下载确切的上游压缩包并原样托管，或在 `extern/<lib>/<lib>.cmake` 中核对期望的
  `URL_HASH`。
- **覆盖项似乎未生效** —— 确认变量是在运行 `make` / `cmake` 的同一个 shell 中 `export` 的，
  然后重新执行配置（或 `make clean`），因为取值是在 CMake 配置阶段读取的。确认配置输出中出现
  `-- Using local path for <lib>: <你的取值>` 这一行。
- **仍然在访问网络** —— 多半是漏掉了本次构建会拉取的某个依赖。请对照
  [我需要哪些依赖？](#我需要哪些依赖) 与你启用的选项（`ENABLE_TESTS`、`ENABLE_TOOLS`、
  Python 绑定）逐项核对。

## 版本可用性

按依赖配置的 `VSAG_THIRDPARTY_*` 覆盖能力在 `main` 开发线以及 `0.15`、`0.16`、`0.17`、
`0.18` 发布线上均可用，因此本地路径与内网镜像覆盖在所有这些分支上的行为完全一致。该能力最初由
[#1606](https://github.com/antgroup/vsag/pull/1606) 在 `main` 引入，并已合入各发布线（跟踪于
[#2308](https://github.com/antgroup/vsag/issues/2308)）。内置的“上游 + 阿里云 OSS 镜像”兜底在
每条线上依然保留；若你不想镜像某个依赖，也仍可使用[复用系统库](#备选方案复用系统库)。
