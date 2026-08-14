# Offline / Air-gapped Builds

VSAG downloads a set of third-party libraries at CMake **configure / build time**
(via `ExternalProject_Add` and `FetchContent`). On a machine without internet
access, or behind a slow / restricted network, those downloads can fail or time
out. This page explains how to point each dependency at a **local path** or an
**internal mirror** (internal HTTP server, object storage, Artifactory, etc.) so the
build can complete fully offline.

## How third-party downloads are resolved

For every downloaded dependency, VSAG builds a *list* of candidate URLs and lets
CMake try them **in order**, stopping at the first one that succeeds. Using
`antlr4` as the representative example
([`extern/antlr4/antlr4.cmake`](https://github.com/antgroup/vsag/blob/1.0/extern/antlr4/antlr4.cmake)):

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

The resolution order is therefore:

1. The **pinned variable**, `VSAG_THIRDPARTY_<LIB>_<PIN_SUFFIX>`, for the exact
   dependency identifier used by the current branch.
2. The unversioned `VSAG_THIRDPARTY_<LIB>` variable, as a deprecated compatibility
   fallback. If both variables are set, the pinned variable wins.
3. The authoritative **upstream** URL or URLs (GitHub / project release page).

VSAG does not provide a project-controlled archive mirror. For a local or internal
mirror, set the pinned variable (preferred) or the deprecated legacy variable.

Only the pinned variable for the branch's exact pin is read. Pinned variables for
other versions can remain exported and are ignored. This lets one CI image or shell
profile safely support several VSAG branches at once.

## Pinned-variable names

The suffix is derived from the exact pinned identifier:

- Numeric releases strip one leading `v`/`V` and a dependency-name decoration
  immediately before the number. ASCII letters are uppercased and runs of other
  characters become one underscore. Thus `10.2.1` and `v10.2.1` become `10_2_1`,
  `hdf5_1.14.4` becomes `1_14_4`, and `yaml-cpp-0.9.0` becomes `0_9_0`.
- A full commit hash becomes `COMMIT_<12_HEX>`, for example
  `VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8`.
- Other tags and refs become `TAG_<NORMALIZED>_H<12_HEX>`. The hexadecimal part is
  the uppercase prefix of SHA-256 over the exact, case-sensitive identifier, so
  spellings that normalize alike remain distinguishable.
- If two pins for one dependency collide at 12 hexadecimal characters, both names
  extend deterministically one character at a time. Configuration fails instead of
  accepting an ambiguous name if the full hash or digest cannot distinguish them.

## Key facts before you start

- **The value may be a local path or a URL.** Accepted forms include an
  absolute filesystem path (`/data/deps/fmt-10.2.1.tar.gz`), a `file://` URL, or
  any `http(s)://` URL — including an internal HTTP server or object storage.
- **The archive hash is still verified.** Each dependency declares a
  `URL_HASH` (MD5 or SHA256). Your mirrored / local archive must be **byte
  identical to the upstream archive**, otherwise CMake aborts with a hash
  mismatch. The simplest safe approach is to download the exact upstream file
  once and re-host it unchanged.
- **Overrides are read at configure time.** If you change a variable after a
  previous configure, re-run CMake configure or run `make clean` first so the
  new value takes effect.
- **Use a non-empty value, or leave it unset.** CMake treats a variable that is
  exported but empty as *defined*, so `export VSAG_THIRDPARTY_FMT_10_2_1=` would prepend
  an empty entry to the URL list and break the download. To disable an override,
  `unset` it instead of setting it to an empty string.
- **Each dependency is independent.** There is no single global mirror variable;
  set one pinned variable per dependency you need. You only need to set
  variables for the dependencies your build actually pulls in (see
  [Which dependencies do I need?](#which-dependencies-do-i-need)).
- **Confirmation in the log.** CMake reports the dependency, pin, selected source
  category and variable, plus fallback/deprecation behavior. It intentionally does
  not print the value because URLs can contain credentials.

## Environment variables

| Pinned variable on `1.0` | Library | Upstream archive to mirror | Pulled in when |
| --- | --- | --- | --- |
| `VSAG_THIRDPARTY_JSON_3_11_3` | nlohmann/json 3.11.3 | `github.com/nlohmann/json/.../v3.11.3.tar.gz` | always |
| `VSAG_THIRDPARTY_ANTLR4_4_13_2` | ANTLR4 runtime 4.13.2 | `github.com/antlr/antlr4/.../4.13.2.tar.gz` | always |
| `VSAG_THIRDPARTY_BOOST_1_67_0` | Boost 1.67.0 (headers) | `archives.boost.io/.../boost_1_67_0.tar.gz` | always |
| `VSAG_THIRDPARTY_OPENBLAS_0_3_23` | OpenBLAS 0.3.23 | `github.com/OpenMathLib/OpenBLAS/.../OpenBLAS-0.3.23.tar.gz` | default BLAS backend (when not using system / MKL) |
| `VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8` | pytorch/cpuinfo | `github.com/pytorch/cpuinfo/archive/ca678952...tar.gz` | always |
| `VSAG_THIRDPARTY_FMT_10_2_1` | fmt 10.2.1 | `github.com/fmtlib/fmt/.../10.2.1.tar.gz` | always (unless system fmt) |
| `VSAG_THIRDPARTY_THREAD_POOL_COMMIT_3507796E172D` | log4cplus/ThreadPool | `github.com/log4cplus/ThreadPool/archive/3507796e...tar.gz` | always |
| `VSAG_THIRDPARTY_TSL_1_4_0` | Tessil/robin-map 1.4.0 | `github.com/Tessil/robin-map/.../v1.4.0.tar.gz` | always |
| `VSAG_THIRDPARTY_ROARINGBITMAP_3_0_1` | CRoaring 3.0.1 | `github.com/RoaringBitmap/CRoaring/.../v3.0.1.tar.gz` | always |
| `VSAG_THIRDPARTY_CATCH2_3_7_1` | Catch2 3.7.1 | `github.com/catchorg/Catch2/.../v3.7.1.tar.gz` | `ENABLE_TESTS=ON` |
| `VSAG_THIRDPARTY_HDF5_1_14_4` | HDF5 1.14.4 | `github.com/HDFGroup/hdf5/.../hdf5_1.14.4.tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_ARGPARSE_3_1` | p-ranav/argparse 3.1 | `github.com/p-ranav/argparse/.../v3.1.tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_YAML_CPP_0_9_0` | yaml-cpp 0.9.0 | `github.com/jbeder/yaml-cpp/.../yaml-cpp-0.9.0.tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_TABULATE_COMMIT_3A58301067BB` | p-ranav/tabulate | `github.com/p-ranav/tabulate/archive/3a583010...tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_HTTPLIB_0_35_0` | cpp-httplib 0.35.0 | `github.com/yhirose/cpp-httplib/.../v0.35.0.tar.gz` | `ENABLE_TOOLS=ON` (+ C++11 ABI) |
| `VSAG_THIRDPARTY_PYBIND11_2_11_1` | pybind11 2.11.1 | `github.com/pybind/pybind11/.../v2.11.1.tar.gz` | Python bindings (`pyvsag` / `ENABLE_PYBINDS=ON`) |

> The exact upstream URL **and** the expected `URL_HASH` for each dependency are
> the single source of truth in the corresponding
> [`extern/<lib>/<lib>.cmake`](https://github.com/antgroup/vsag/tree/1.0/extern)
> file. Check that file when mirroring, especially after a version bump.

Not listed here (no download, so no override needed): **Intel MKL**, which is located on the host
with `find_path`, and **DiskANN**, which is vendored in-tree under `extern/diskann/`.

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
  [`cmake/VSAGThirdParty.cmake`](https://github.com/antgroup/vsag/blob/1.0/cmake/VSAGThirdParty.cmake)).
- **+ Python wheel** (`make pyvsag`): also `PYBIND11`.

## Examples

### A. Internal HTTP server or object storage (recommended)

Re-host the upstream archives unchanged on an internal endpoint, then point each
variable at it. A base-URL shell variable keeps this compact:

```bash
# Internal mirror that serves the upstream archives byte-for-byte
export VSAG_MIRROR=https://mirror.corp.example.com/vsag-thirdparty

export VSAG_THIRDPARTY_JSON_3_11_3=$VSAG_MIRROR/v3.11.3.tar.gz
export VSAG_THIRDPARTY_ANTLR4_4_13_2=$VSAG_MIRROR/antlr4-4.13.2.tar.gz
export VSAG_THIRDPARTY_BOOST_1_67_0=$VSAG_MIRROR/boost_1_67_0.tar.gz
export VSAG_THIRDPARTY_OPENBLAS_0_3_23=$VSAG_MIRROR/OpenBLAS-0.3.23.tar.gz
export VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8=$VSAG_MIRROR/cpuinfo-ca678952.tar.gz
export VSAG_THIRDPARTY_FMT_10_2_1=$VSAG_MIRROR/fmt-10.2.1.tar.gz
export VSAG_THIRDPARTY_THREAD_POOL_COMMIT_3507796E172D=$VSAG_MIRROR/thread_pool-3507796e.tar.gz
export VSAG_THIRDPARTY_TSL_1_4_0=$VSAG_MIRROR/robin-map-1.4.0.tar.gz
export VSAG_THIRDPARTY_ROARINGBITMAP_3_0_1=$VSAG_MIRROR/CRoaring-3.0.1.tar.gz

make release
```

An object-storage endpoint works identically: use its network-reachable object URL.

### B. Pre-downloaded local files (fully air-gapped)

On a machine that has *no* network at all, copy the archives onto the box first
(e.g. to `/data/vsag-deps`) and point the variables at the local files:

```bash
export VSAG_THIRDPARTY_JSON_3_11_3=/data/vsag-deps/v3.11.3.tar.gz
export VSAG_THIRDPARTY_ANTLR4_4_13_2=/data/vsag-deps/antlr4-4.13.2.tar.gz
export VSAG_THIRDPARTY_BOOST_1_67_0=/data/vsag-deps/boost_1_67_0.tar.gz
export VSAG_THIRDPARTY_OPENBLAS_0_3_23=/data/vsag-deps/OpenBLAS-0.3.23.tar.gz
export VSAG_THIRDPARTY_CPUINFO_COMMIT_CA678952A9A8=/data/vsag-deps/cpuinfo-ca678952.tar.gz
export VSAG_THIRDPARTY_FMT_10_2_1=/data/vsag-deps/fmt-10.2.1.tar.gz
export VSAG_THIRDPARTY_THREAD_POOL_COMMIT_3507796E172D=/data/vsag-deps/thread_pool-3507796e.tar.gz
export VSAG_THIRDPARTY_TSL_1_4_0=/data/vsag-deps/robin-map-1.4.0.tar.gz
export VSAG_THIRDPARTY_ROARINGBITMAP_3_0_1=/data/vsag-deps/CRoaring-3.0.1.tar.gz

make release
```

A `file://` URL (`export VSAG_THIRDPARTY_FMT_10_2_1=file:///data/vsag-deps/fmt-10.2.1.tar.gz`)
is equally valid.

### C. Override a single dependency

If only one download is unreliable, override just that one and let the rest use
the defaults:

```bash
export VSAG_THIRDPARTY_OPENBLAS_0_3_23=https://mirror.example.com/OpenBLAS-0.3.23.tar.gz
make release
```

### D. Keep several branch pins exported

These variables can coexist. After later backports land, switching branches will
select only that branch's pin without editing the environment:

```bash
export VSAG_THIRDPARTY_OPENBLAS_0_3_23=/data/vsag-deps/OpenBLAS-0.3.23.tar.gz
export VSAG_THIRDPARTY_OPENBLAS_0_3_24=/data/vsag-deps/OpenBLAS-0.3.24.tar.gz
export VSAG_THIRDPARTY_YAML_CPP_0_8_0=/data/vsag-deps/yaml-cpp-0.8.0.tar.gz
export VSAG_THIRDPARTY_YAML_CPP_0_9_0=/data/vsag-deps/yaml-cpp-0.9.0.tar.gz

git switch 1.0    # selects OpenBLAS 0.3.23 and yaml-cpp 0.9.0
make release
```

## Alternative: reuse system libraries

For dependencies that are already installed on the host, you can skip the
download entirely instead of mirroring it. Set `VSAG_USE_SYSTEM_DEPS=ON` (or the
per-dependency `VSAG_USE_SYSTEM_<DEP>=ON`). See
[`DEVELOPMENT.md`](https://github.com/antgroup/vsag/blob/1.0/DEVELOPMENT.md#system-third-party-dependencies)
for the list of dependencies that currently support system reuse.

## Troubleshooting

- **Hash mismatch / "HASH mismatch" error** — your mirrored or local archive is
  not byte-identical to the upstream file. Re-download the exact upstream
  archive and re-host it unchanged, or confirm the expected `URL_HASH` in
  `extern/<lib>/<lib>.cmake`.
- **Override seems ignored** — make sure the variable was `export`ed in the same
  shell that runs `make` / `cmake`, then re-run configure (or `make clean`),
  because the value is read at CMake configure time. Confirm that the
  `Third-party override` diagnostic names the expected pin and variable.
- **Still hitting the network** — you probably missed a dependency that your
  build pulls in. Cross-check the list in
  [Which dependencies do I need?](#which-dependencies-do-i-need) against your
  enabled options (`ENABLE_TESTS`, `ENABLE_TOOLS`, Python bindings).

## Version availability

Pinned variables are implemented on `main` and `1.0`; backports to `0.18`, `0.17`,
and `0.16` are independent changes because each branch owns its pins. The deprecated
unversioned overrides remain compatible. They were introduced on `main`
by [#1606](https://github.com/antgroup/vsag/pull/1606) and backported to release
lines through [#2308](https://github.com/antgroup/vsag/issues/2308). Authoritative upstream URL
fallbacks and [system-library reuse](#alternative-reuse-system-libraries) are unchanged.
