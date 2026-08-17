# 离线 / 内网环境构建

VSAG 在 CMake **配置 / 构建阶段**会下载一批第三方库（通过 `ExternalProject_Add`
与 `FetchContent`）。在没有外网访问、或网络较慢 / 受限的机器上，这些下载可能失败或
超时。本文介绍如何把每个依赖指向**本地路径**或**内网镜像**（内网 HTTP 服务、对象
存储、Artifactory 等），从而在完全离线的环境中完成编译。

## 第三方下载的解析顺序

对每一个需要下载的依赖，VSAG 会构造一个候选 URL *列表*，由 CMake **按顺序**依次尝试，
命中第一个成功的即停止。以 `antlr4` 为代表
（[`extern/antlr4/antlr4.cmake`](https://github.com/antgroup/vsag/blob/1.0/extern/antlr4/antlr4.cmake)）：

```cmake
set (antlr4_urls
    https://github.com/antlr/antlr4/archive/refs/tags/4.13.2.tar.gz
)
vsag_resolve_thirdparty_override (ANTLR4 4.13.2 antlr4_urls)

ExternalProject_Add (antlr4
    URL ${antlr4_urls}
    URL_HASH MD5=3b75610fc8a827119258cba09a068be5
    ...)
```

因此解析顺序为：

1. 当前分支确切依赖标识符对应的**固定版本变量（pinned variable）**
   `VSAG_THIRDPARTY_<LIB>_<PIN_SUFFIX>`。
2. 未带版本的 `VSAG_THIRDPARTY_<LIB>`，作为已弃用的兼容兜底。如果两者都设置，固定版本变量优先。
3. 权威的**上游** URL（一个或多个，来自 GitHub / 项目发布页）。

VSAG 不提供项目控制的压缩包镜像。如需使用本地或内网镜像，请设置固定版本变量（推荐），
或临时使用已弃用的旧变量。

只会读取当前分支确切 pin 对应的固定版本变量。其他版本的变量可以一直保留在环境中，它们会被
忽略，因此同一个 CI 镜像或 shell 配置可以安全地支持多个 VSAG 分支。

## 固定版本变量命名

后缀由确切的固定标识符生成：

- 数字版本会去掉一个开头的 `v`/`V`，以及紧挨数字之前的依赖名装饰；ASCII 字母转为大写，
  其他字符的连续段替换为一个下划线。因此 `10.2.1` 和 `v10.2.1` 得到 `10_2_1`，
  `hdf5_1.14.4` 得到 `1_14_4`，`yaml-cpp-0.9.0` 得到 `0_9_0`。
- 完整提交哈希得到 `COMMIT_<12_HEX>`，例如
  `VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8`。
- 其他标签或 ref 得到 `TAG_<NORMALIZED>_H<12_HEX>`。十六进制部分是对大小写敏感的确切
  标识符计算 SHA-256 后的大写前缀，所以规范化后相同的不同拼写仍可区分。
- 同一依赖的两个 pin 若在 12 个十六进制字符处冲突，两边会确定性地逐字符扩展。如果完整哈希
  或摘要仍无法区分，配置会失败，而不会接受有歧义的名称。

## 开始前的关键事项

- **取值可以是本地路径，也可以是 URL。** 支持绝对文件路径
  （`/data/deps/fmt-10.2.1.tar.gz`）、`file://` URL，或任意 `http(s)://` URL——
  包括内网 HTTP 服务或对象存储。
- **依然会校验压缩包哈希。** 每个依赖都声明了 `URL_HASH`（MD5 或 SHA256）。你镜像 /
  本地的压缩包必须与**上游压缩包逐字节一致**，否则 CMake 会因哈希不匹配而中止。最稳妥的
  做法是把上游原始文件下载一次，原封不动地重新托管。
- **覆盖项在配置阶段读取。** 如果你在上一次配置之后修改了变量，请重新执行 CMake 配置或
  先运行 `make clean`，新值才会生效。
- **请使用非空值，否则就不要设置。** CMake 把“已 export 但为空”的变量视为*已定义*，因此
  `export VSAG_THIRDPARTY_FMT_10_2_1=` 会把一个空项 prepend 到 URL 列表里，导致下载失败。若要停用
  某个覆盖项，请 `unset` 它，而不要把它设为空字符串。
- **每个依赖相互独立。** 没有单一的全局镜像变量；每个需要的依赖各自设置一个固定版本变量。
  你只需为本次构建实际拉取的依赖设置变量（见
  [我需要哪些依赖？](#我需要哪些依赖)）。
- **日志中的确认信息。** CMake 会报告依赖、pin、选中的来源类别与变量，以及兜底 / 弃用行为。
  由于 URL 可能包含凭据，日志不会打印变量值。

## 环境变量

| `1.0` 上的固定版本变量 | 库 | 需镜像的上游压缩包 | 何时被拉取 |
| --- | --- | --- | --- |
| `VSAG_THIRDPARTY_JSON_3_11_3` | nlohmann/json 3.11.3 | `github.com/nlohmann/json/.../v3.11.3.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_ANTLR4_4_13_2` | ANTLR4 runtime 4.13.2 | `github.com/antlr/antlr4/.../4.13.2.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_BOOST_1_67_0` | Boost 1.67.0（头文件） | `archives.boost.io/.../boost_1_67_0.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_OPENBLAS_0_3_24` | OpenBLAS 0.3.24 | `github.com/OpenMathLib/OpenBLAS/.../OpenBLAS-0.3.24.tar.gz` | 默认 BLAS 后端（未使用系统库 / MKL 时） |
| `VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8` | pytorch/cpuinfo | `github.com/pytorch/cpuinfo/archive/ca678952...tar.gz` | 始终 |
| `VSAG_THIRDPARTY_FMT_10_2_1` | fmt 10.2.1 | `github.com/fmtlib/fmt/.../10.2.1.tar.gz` | 始终（除非使用系统 fmt） |
| `VSAG_THIRDPARTY_THREAD_POOL_COMMIT_3507796E172D` | log4cplus/ThreadPool | `github.com/log4cplus/ThreadPool/archive/3507796e...tar.gz` | 始终 |
| `VSAG_THIRDPARTY_TSL_1_4_0` | Tessil/robin-map 1.4.0 | `github.com/Tessil/robin-map/.../v1.4.0.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_ROARINGBITMAP_3_0_1` | CRoaring 3.0.1 | `github.com/RoaringBitmap/CRoaring/.../v3.0.1.tar.gz` | 始终 |
| `VSAG_THIRDPARTY_CATCH2_3_7_1` | Catch2 3.7.1 | `github.com/catchorg/Catch2/.../v3.7.1.tar.gz` | `ENABLE_TESTS=ON` |
| `VSAG_THIRDPARTY_HDF5_1_14_4` | HDF5 1.14.4 | `github.com/HDFGroup/hdf5/.../hdf5_1.14.4.tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_ARGPARSE_3_1` | p-ranav/argparse 3.1 | `github.com/p-ranav/argparse/.../v3.1.tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_YAML_CPP_0_9_0` | yaml-cpp 0.9.0 | `github.com/jbeder/yaml-cpp/.../yaml-cpp-0.9.0.tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_TABULATE_COMMIT_3A58301067BB` | p-ranav/tabulate | `github.com/p-ranav/tabulate/archive/3a583010...tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_HTTPLIB_0_35_0` | cpp-httplib 0.35.0 | `github.com/yhirose/cpp-httplib/.../v0.35.0.tar.gz` | `ENABLE_TOOLS=ON`（且 C++11 ABI） |
| `VSAG_THIRDPARTY_PYBIND11_2_11_1` | pybind11 2.11.1 | `github.com/pybind/pybind11/.../v2.11.1.tar.gz` | Python 绑定（`pyvsag` / `ENABLE_PYBINDS=ON`） |

> 每个依赖确切的上游 URL **以及**期望的 `URL_HASH`，其唯一权威来源是对应的
> [`extern/<lib>/<lib>.cmake`](https://github.com/antgroup/vsag/tree/1.0/extern)
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
  （见 [`cmake/VSAGThirdParty.cmake`](https://github.com/antgroup/vsag/blob/1.0/cmake/VSAGThirdParty.cmake)）。
- **+ Python wheel**（`make pyvsag`）：另加 `PYBIND11`。

## 示例

### A. 内网 HTTP 服务或对象存储（推荐）

将上游压缩包原封不动地重新托管到内网地址，再让每个变量指向它。用一个基础 URL 的 shell
变量可以让配置更简洁：

```bash
# 内网镜像，逐字节提供上游压缩包
export VSAG_MIRROR=https://mirror.corp.example.com/vsag-thirdparty

export VSAG_THIRDPARTY_JSON_3_11_3=$VSAG_MIRROR/v3.11.3.tar.gz
export VSAG_THIRDPARTY_ANTLR4_4_13_2=$VSAG_MIRROR/antlr4-4.13.2.tar.gz
export VSAG_THIRDPARTY_BOOST_1_67_0=$VSAG_MIRROR/boost_1_67_0.tar.gz
export VSAG_THIRDPARTY_OPENBLAS_0_3_24=$VSAG_MIRROR/OpenBLAS-0.3.24.tar.gz
export VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8=$VSAG_MIRROR/cpuinfo-ca678952.tar.gz
export VSAG_THIRDPARTY_FMT_10_2_1=$VSAG_MIRROR/fmt-10.2.1.tar.gz
export VSAG_THIRDPARTY_THREAD_POOL_COMMIT_3507796E172D=$VSAG_MIRROR/thread_pool-3507796e.tar.gz
export VSAG_THIRDPARTY_TSL_1_4_0=$VSAG_MIRROR/robin-map-1.4.0.tar.gz
export VSAG_THIRDPARTY_ROARINGBITMAP_3_0_1=$VSAG_MIRROR/CRoaring-3.0.1.tar.gz

make release
```

对象存储端点的用法完全相同：使用其网络可达的对象 URL 即可。

### B. 预先下载的本地文件（完全离线）

在完全没有网络的机器上，先把压缩包拷贝到本机（例如 `/data/vsag-deps`），再让变量指向本地
文件：

```bash
export VSAG_THIRDPARTY_JSON_3_11_3=/data/vsag-deps/v3.11.3.tar.gz
export VSAG_THIRDPARTY_ANTLR4_4_13_2=/data/vsag-deps/antlr4-4.13.2.tar.gz
export VSAG_THIRDPARTY_BOOST_1_67_0=/data/vsag-deps/boost_1_67_0.tar.gz
export VSAG_THIRDPARTY_OPENBLAS_0_3_24=/data/vsag-deps/OpenBLAS-0.3.24.tar.gz
export VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8=/data/vsag-deps/cpuinfo-ca678952.tar.gz
export VSAG_THIRDPARTY_FMT_10_2_1=/data/vsag-deps/fmt-10.2.1.tar.gz
export VSAG_THIRDPARTY_THREAD_POOL_COMMIT_3507796E172D=/data/vsag-deps/thread_pool-3507796e.tar.gz
export VSAG_THIRDPARTY_TSL_1_4_0=/data/vsag-deps/robin-map-1.4.0.tar.gz
export VSAG_THIRDPARTY_ROARINGBITMAP_3_0_1=/data/vsag-deps/CRoaring-3.0.1.tar.gz

make release
```

使用 `file://` URL（`export VSAG_THIRDPARTY_FMT_10_2_1=file:///data/vsag-deps/fmt-10.2.1.tar.gz`）
同样有效。

### C. 只覆盖单个依赖

如果只有某一个下载不稳定，只覆盖它即可，其余继续使用默认地址：

```bash
export VSAG_THIRDPARTY_OPENBLAS_0_3_24=https://mirror.example.com/OpenBLAS-0.3.24.tar.gz
make release
```

### D. 同时保留多个分支的 pin

这些变量可以共存。后续各发布分支独立回移后，切换分支时会自动选择该分支自己的 pin，
无需修改环境：

```bash
export VSAG_THIRDPARTY_OPENBLAS_0_3_23=/data/vsag-deps/OpenBLAS-0.3.23.tar.gz
export VSAG_THIRDPARTY_OPENBLAS_0_3_24=/data/vsag-deps/OpenBLAS-0.3.24.tar.gz
export VSAG_THIRDPARTY_YAML_CPP_0_8_0=/data/vsag-deps/yaml-cpp-0.8.0.tar.gz
export VSAG_THIRDPARTY_YAML_CPP_0_9_0=/data/vsag-deps/yaml-cpp-0.9.0.tar.gz

git switch 1.0    # 选择 OpenBLAS 0.3.24 与 yaml-cpp 0.9.0
make release
```

## 备选方案：复用系统库

对于主机上已安装的依赖，你可以直接跳过下载，而不必镜像。设置
`VSAG_USE_SYSTEM_DEPS=ON`（或按依赖设置 `VSAG_USE_SYSTEM_<DEP>=ON`）。当前支持系统复用的
依赖列表见
[`DEVELOPMENT.md`](https://github.com/antgroup/vsag/blob/1.0/DEVELOPMENT.md#system-third-party-dependencies)。

## 常见问题

- **哈希不匹配 / 出现 “HASH mismatch” 错误** —— 你镜像或本地的压缩包与上游文件不是逐字节
  一致。请重新下载确切的上游压缩包并原样托管，或在 `extern/<lib>/<lib>.cmake` 中核对期望的
  `URL_HASH`。
- **覆盖项似乎未生效** —— 确认变量是在运行 `make` / `cmake` 的同一个 shell 中 `export` 的，
  然后重新执行配置（或 `make clean`），因为取值是在 CMake 配置阶段读取的。确认配置输出中的
  `Third-party override` 诊断包含预期的 pin 与变量。
- **仍然在访问网络** —— 多半是漏掉了本次构建会拉取的某个依赖。请对照
  [我需要哪些依赖？](#我需要哪些依赖) 与你启用的选项（`ENABLE_TESTS`、`ENABLE_TOOLS`、
  Python 绑定）逐项核对。

## 版本可用性

固定版本变量已在 `main` 与 `1.0` 上实现；`0.18`、`0.17` 与 `0.16` 的回移需要独立完成，
因为每个分支拥有自己的 pin。已弃用的未带版本覆盖仍保持兼容。它最初由
[#1606](https://github.com/antgroup/vsag/pull/1606) 在 `main` 引入，并通过
[#2308](https://github.com/antgroup/vsag/issues/2308) 回移到发布线。权威上游 URL 兜底与
[复用系统库](#备选方案复用系统库)的行为不变。
