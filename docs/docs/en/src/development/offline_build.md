# Offline / Air-gapped Builds

VSAG downloads a set of third-party libraries at CMake **configure / build time**
(via `ExternalProject_Add` and `FetchContent`). On a machine without internet
access, or behind a slow / restricted network, those downloads can fail or time
out. This page explains how to point each dependency at a **local path** or an
**internal mirror** (internal HTTP server, OSS bucket, Artifactory, etc.) so the
build can complete fully offline.

## How third-party downloads are resolved

For every downloaded dependency, VSAG builds a *list* of candidate URLs and lets
CMake try them **in order**, stopping at the first one that succeeds. Using
`antlr4` as the representative example
([`extern/antlr4/antlr4.cmake`](https://github.com/antgroup/vsag/blob/main/extern/antlr4/antlr4.cmake)):

```cmake
set (antlr4_urls
    https://github.com/antlr/antlr4/archive/refs/tags/4.13.2.tar.gz   # 1. upstream
    https://vsagcache.oss-rg-china-mainland.aliyuncs.com/antlr4/v4.13.2.tar.gz  # 2. project mirror
)
if (DEFINED ENV{VSAG_THIRDPARTY_ANTLR4})
    message (STATUS "Using local path for antlr4: $ENV{VSAG_THIRDPARTY_ANTLR4}")
    list (PREPEND antlr4_urls "$ENV{VSAG_THIRDPARTY_ANTLR4}")   # 0. your override (tried first)
endif ()

ExternalProject_Add (antlr4
    URL ${antlr4_urls}
    URL_HASH MD5=3b75610fc8a827119258cba09a068be5
    ...)
```

The resolution order is therefore:

1. **`VSAG_THIRDPARTY_<LIB>`** — your override, if the environment variable is
   set to a **non-empty** value. Tried **first**.
2. The **upstream** URL (GitHub / project release page).
3. The project-maintained **Aliyun OSS mirror**
   (`vsagcache.oss-rg-china-mainland.aliyuncs.com`). This fallback is always
   present and helps in mainland-China / poor-network environments, but it is
   **not** user-configurable — for a fully internal mirror, use the environment
   variable.

> **Availability:** the `VSAG_THIRDPARTY_*` override is available on `main` and on
> the `0.15`, `0.16`, `0.17`, and `0.18` release lines — see
> [Version availability](#version-availability).

## Key facts before you start

- **The value may be a local path or a URL.** Accepted forms include an
  absolute filesystem path (`/data/deps/fmt-10.2.1.tar.gz`), a `file://` URL, or
  any `http(s)://` URL — including an internal HTTP server or an OSS / S3 bucket.
- **The archive hash is still verified.** Each dependency declares a
  `URL_HASH` (MD5 or SHA256). Your mirrored / local archive must be **byte
  identical to the upstream archive**, otherwise CMake aborts with a hash
  mismatch. The simplest safe approach is to download the exact upstream file
  once and re-host it unchanged.
- **Overrides are read at configure time.** If you change a variable after a
  previous configure, re-run CMake configure or run `make clean` first so the
  new value takes effect.
- **Use a non-empty value, or leave it unset.** CMake treats a variable that is
  exported but empty as *defined*, so `export VSAG_THIRDPARTY_FMT=` would prepend
  an empty entry to the URL list and break the download. To disable an override,
  `unset` it instead of setting it to an empty string.
- **Each dependency is independent.** There is no single global mirror variable;
  set one `VSAG_THIRDPARTY_<LIB>` per dependency you need. You only need to set
  variables for the dependencies your build actually pulls in (see
  [Which dependencies do I need?](#which-dependencies-do-i-need)).
- **Confirmation in the log.** When an override is picked up, CMake prints
  `-- Using local path for <lib>: <your value>`.

## Environment variables

| Environment variable | Library | Upstream archive to mirror | Pulled in when |
| --- | --- | --- | --- |
| `VSAG_THIRDPARTY_JSON` | nlohmann/json 3.11.3 | `github.com/nlohmann/json/.../v3.11.3.tar.gz` | always |
| `VSAG_THIRDPARTY_ANTLR4` | ANTLR4 runtime 4.13.2 | `github.com/antlr/antlr4/.../4.13.2.tar.gz` | always |
| `VSAG_THIRDPARTY_BOOST` | Boost 1.67.0 (headers) | `archives.boost.io/.../boost_1_67_0.tar.gz` | always |
| `VSAG_THIRDPARTY_OPENBLAS` | OpenBLAS 0.3.23 | `github.com/OpenMathLib/OpenBLAS/.../OpenBLAS-0.3.23.tar.gz` | default BLAS backend (when not using system / MKL) |
| `VSAG_THIRDPARTY_CPUINFO` | pytorch/cpuinfo | `github.com/pytorch/cpuinfo/archive/ca678952...tar.gz` | always |
| `VSAG_THIRDPARTY_FMT` | fmt 10.2.1 | `github.com/fmtlib/fmt/.../10.2.1.tar.gz` | always (unless system fmt) |
| `VSAG_THIRDPARTY_THREAD_POOL` | log4cplus/ThreadPool | `github.com/log4cplus/ThreadPool/archive/3507796e...tar.gz` | always |
| `VSAG_THIRDPARTY_TSL` | Tessil/robin-map 1.4.0 | `github.com/Tessil/robin-map/.../v1.4.0.tar.gz` | always |
| `VSAG_THIRDPARTY_ROARINGBITMAP` | CRoaring 3.0.1 | `github.com/RoaringBitmap/CRoaring/.../v3.0.1.tar.gz` | always |
| `VSAG_THIRDPARTY_CATCH2` | Catch2 3.7.1 | `github.com/catchorg/Catch2/.../v3.7.1.tar.gz` | `ENABLE_TESTS=ON` |
| `VSAG_THIRDPARTY_HDF5` | HDF5 1.14.4 | `github.com/HDFGroup/hdf5/.../hdf5_1.14.4.tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_ARGPARSE` | p-ranav/argparse 3.1 | `github.com/p-ranav/argparse/.../v3.1.tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_YAML_CPP` | yaml-cpp 0.9.0 | `github.com/jbeder/yaml-cpp/.../yaml-cpp-0.9.0.tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_TABULATE` | p-ranav/tabulate | `github.com/p-ranav/tabulate/archive/3a583010...tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_HTTPLIB` | cpp-httplib 0.35.0 | `github.com/yhirose/cpp-httplib/.../v0.35.0.tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_PYBIND11` | pybind11 2.11.1 | `github.com/pybind/pybind11/.../v2.11.1.tar.gz` | Python bindings (`pyvsag` / `ENABLE_PYBINDS=ON`) |

> The exact upstream URL **and** the expected `URL_HASH` for each dependency are
> the single source of truth in the corresponding
> [`extern/<lib>/<lib>.cmake`](https://github.com/antgroup/vsag/tree/main/extern)
> file. Check that file when mirroring, especially after a version bump.

Not listed here (no download, so no override needed): **Intel MKL** (located on
the host with `find_path`) and **DiskANN** (vendored in-tree under
`extern/diskann/`).

## Which dependencies do I need?

You only have to mirror what your specific build actually downloads:

- **Core library** (`make debug` / `make release`): `JSON`, `ANTLR4`, `BOOST`,
  `OPENBLAS`, `CPUINFO`, `FMT`, `THREAD_POOL`, `TSL`, `ROARINGBITMAP`.
  Two of these are conditional: `OPENBLAS` is **not** downloaded when BLAS comes
  from Intel MKL (x86_64 with `ENABLE_INTEL_MKL=ON`) or from a system OpenBLAS,
  and `FMT` is skipped when a system `fmt` is found.
- **+ Tests** (`make test`, `ENABLE_TESTS=ON`): also `CATCH2`.
- **+ Tools** (`ENABLE_TOOLS=ON` **and** `ENABLE_CXX11_ABI=ON`): also `HDF5`,
  `ARGPARSE`, `YAML_CPP`, `TABULATE`, `HTTPLIB` — downloaded only when *both*
  options are enabled (see
  [`cmake/VSAGThirdParty.cmake`](https://github.com/antgroup/vsag/blob/main/cmake/VSAGThirdParty.cmake)).
- **+ Python wheel** (`make pyvsag`): also `PYBIND11`.

## Examples

### A. Internal HTTP server or OSS bucket (recommended)

Re-host the upstream archives unchanged on an internal endpoint, then point each
variable at it. A base-URL shell variable keeps this compact:

```bash
# Internal mirror that serves the upstream archives byte-for-byte
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

An OSS / S3 bucket works identically — just use its public (or
network-reachable) object URL, for example
`https://my-bucket.oss-cn-hangzhou.aliyuncs.com/vsag/OpenBLAS-0.3.23.tar.gz`.

### B. Pre-downloaded local files (fully air-gapped)

On a machine that has *no* network at all, copy the archives onto the box first
(e.g. to `/data/vsag-deps`) and point the variables at the local files:

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

A `file://` URL (`export VSAG_THIRDPARTY_FMT=file:///data/vsag-deps/fmt-10.2.1.tar.gz`)
is equally valid.

### C. Override a single dependency

If only one download is unreliable, override just that one and let the rest use
the defaults:

```bash
export VSAG_THIRDPARTY_OPENBLAS=https://mirror.corp.example.com/OpenBLAS-0.3.23.tar.gz
make release
```

## Alternative: reuse system libraries

For dependencies that are already installed on the host, you can skip the
download entirely instead of mirroring it. Set `VSAG_USE_SYSTEM_DEPS=ON` (or the
per-dependency `VSAG_USE_SYSTEM_<DEP>=ON`). See
[`DEVELOPMENT.md`](https://github.com/antgroup/vsag/blob/main/DEVELOPMENT.md#system-third-party-dependencies)
for the list of dependencies that currently support system reuse.

## Troubleshooting

- **Hash mismatch / "HASH mismatch" error** — your mirrored or local archive is
  not byte-identical to the upstream file. Re-download the exact upstream
  archive and re-host it unchanged, or confirm the expected `URL_HASH` in
  `extern/<lib>/<lib>.cmake`.
- **Override seems ignored** — make sure the variable was `export`ed in the same
  shell that runs `make` / `cmake`, then re-run configure (or `make clean`),
  because the value is read at CMake configure time. Confirm the
  `-- Using local path for <lib>: <your value>` line appears in the configure output.
- **Still hitting the network** — you probably missed a dependency that your
  build pulls in. Cross-check the list in
  [Which dependencies do I need?](#which-dependencies-do-i-need) against your
  enabled options (`ENABLE_TESTS`, `ENABLE_TOOLS`, Python bindings).

## Version availability

The per-dependency `VSAG_THIRDPARTY_*` override is available on the `main`
development line and on the `0.15`, `0.16`, `0.17`, and `0.18` release lines, so
local-path and internal-mirror overrides behave the same way across all of them.
It was introduced on `main` by
[#1606](https://github.com/antgroup/vsag/pull/1606) and backported to the release
lines (tracked in [#2308](https://github.com/antgroup/vsag/issues/2308)). The
built-in upstream + Aliyun OSS mirror fallback remains present on every line, and
[system-library reuse](#alternative-reuse-system-libraries) is still available when
you would rather not mirror a dependency at all.
