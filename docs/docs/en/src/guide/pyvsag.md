# pyvsag

`pyvsag` is the official Python binding for VSAG. It is a pure-Python layer that calls the
[C API](c_api.md) (`libvsag_c`) through [cffi](https://cffi.readthedocs.io/), so one
`py3-none` platform wheel per architecture serves every CPython 3.10+ release. Data crosses
the boundary only as pointer, shape and dtype, hence it does not depend on the NumPy ABI. Sources
live under [`python/`](https://github.com/antgroup/vsag/tree/main/python).

## Installation

```bash
pip install pyvsag
```

Requirements:

- **Platform:** Linux `x86_64` or `aarch64` with glibc 2.28 or later (`manylinux_2_28`; for example
  Ubuntu 20.04+, Debian 10+, RHEL/Alma/Rocky 8+). macOS, Windows and musl-based distributions such as
  Alpine have no wheels on the production PyPI index. macOS arm64 wheels are available through
  the TestPyPI workflow below, or can be built locally.
- **Python:** CPython 3.10 or later. 3.10 to 3.14 are supported; later releases need no new wheel, because the wheel does not depend on the Python ABI.
- **Dependencies:** `numpy` and `cffi`, installed automatically. The wheel bundles its OpenMP runtime
  and has no libaio, libcurl or OpenBLAS dependency.

### Nightly builds (TestPyPI)

Wheels built from `main` are published to TestPyPI by the `pyvsag TestPyPI` workflow (nightly and
manual runs; uploads can be paused). They carry a `.devN` version, so `--pre` is required, and the
dependencies still come from the real PyPI. The workflow includes Linux x86_64/aarch64 and
macOS Apple Silicon arm64 (`macosx_15_0_arm64`):

```bash
pip install numpy cffi
pip install --pre --no-deps --index-url https://test.pypi.org/simple/ pyvsag
```

The macOS job runs `make pyvsag` and the Mach-O audit, then tests the same SHA-256-verified
wheel in clean environments: CPython 3.10–3.14 with NumPy 2.x, plus NumPy 1.26.4 on
CPython 3.10–3.12. Two Python 3.12 lanes require `torch==2.15.0` and `scikit-learn==1.9.1` and all six OpenMP
import orders, with NumPy 1.26.4 and 2.x and `KMP_DUPLICATE_LIB_OK=FALSE`; missing dependencies
or skipped OpenMP tests fail the job. The minimum supported macOS is 15, matching the CI runner. This raises the previous macOS 14
minimum: macOS 14 is no longer supported, including by the default local source build.

Linux and macOS upload independently; only macOS upload waits for the macOS matrix. After upload, a new macOS job downloads the exact
version from TestPyPI alone (without dependencies or pip cache), verifies the filename and
SHA-256 against the build manifest, audits it and runs the ordinary suite in a clean environment.
The download retries for up to ten minutes to allow index propagation. A same-name macOS file already on
TestPyPI is rejected before upload: rerun only verification in the original upload run, or use a new version.
Dependencies are installed separately from PyPI. Manual runs with `publish` unchecked can validate any branch and retain
build/test artifacts without uploading or running the published-wheel check. Source SHA, wheel
hash, dependency versions and test reports are retained with the workflow artifacts.
The production PyPI release workflow remains Linux-only.

### Build from source

```bash
make pyvsag
pip install wheelhouse/pyvsag-*.whl
```

`make pyvsag` runs `scripts/python/local_build_wheel.sh` from the repository root and writes the
wheel to `wheelhouse/`. On Linux, with Docker it builds inside the `manylinux_2_28` image and runs
part of `tests/python` there. Without Docker it builds a native `linux_<arch>` wheel using
`python -m build`, installs it into the active virtual environment and runs `tests/python`.
Outside an active virtual environment the script creates `.venv/` in the repository.

On macOS Apple Silicon, install Xcode Command Line Tools and Homebrew, then run:

```bash
bash scripts/deps/install_deps_macos.sh
# Activate a Python 3.10+ virtual environment, or put python3 3.10+ on PATH.
make pyvsag
```

macOS always builds natively, even when Docker is running. The build requires CMake 3.20+ and
Perl on `PATH` (`perl -v`; install with `brew install perl` if missing), and
statically links OpenBLAS and LLVM OpenMP 20.1.8, with only the VSAG C API exported to isolate
OpenMP from other packages. The pinned OpenMP source is downloaded and built automatically;
its Apache-2.0-with-LLVM-exception license and attribution are included in macOS wheels only.
The source build needs access to GitHub release downloads, or mirrors configured through
`VSAG_THIRDPARTY_OPENMP_20_1_8` and `VSAG_THIRDPARTY_LLVM_CMAKE_20_1_8` (URLs for the unchanged
`openmp-20.1.8.src.tar.xz` and `cmake-20.1.8.src.tar.xz` archives; SHA256 checks remain enforced).
The OpenMP sub-build uses the same `CMAKE_BUILD_PARALLEL_LEVEL` limit as VSAG.
`delocate` is installed automatically, repairs dynamic dependencies and checks the native architecture.
A Mach-O audit then requires exactly `libvsag_c.dylib`, only system dependencies, C API exports,
license metadata, and a deployment target no newer than the wheel tag, before installation and tests.

The default `MACOSX_DEPLOYMENT_TARGET` is `15.0`, applied to VSAG and its source-built dependencies.
No environment variables are needed for the default build. Set `MACOSX_DEPLOYMENT_TARGET` to
choose another deployment target, or `CMAKE_BUILD_PARALLEL_LEVEL` to limit build concurrency.
The `macosx_15_0_arm64` tag declares macOS 15 as the minimum; CI runs on that OS version. TestPyPI includes macOS arm64; production PyPI releases
remain Linux-only. Install `torch` and `scikit-learn` in the test environment to include the optional OpenMP
coexistence tests.

The macOS wheel build enables `VSAG_OPENMP_STATIC`; ordinary C++ builds keep dynamic OpenMP.
For C++ builds, install `libomp` with Homebrew or pass `-DLIBOMP_PREFIX=/path/to/libomp` to CMake.
For a wheel, an optional `LIBOMP_PREFIX` environment variable selects a custom installation containing
`$LIBOMP_PREFIX/include/omp.h` and `$LIBOMP_PREFIX/lib/libomp.a`; the archive must target the requested macOS version or older.
Newer Homebrew bottles are rejected instead of being silently relabeled. Unset `LIBOMP_PREFIX` to
return to the compatible source build.

To develop against an in-tree build without packaging a wheel:

```bash
make debug VSAG_ENABLE_PYBINDS=ON              # builds build/src/libvsag_c.so
bash scripts/python/prepare_python_build.sh    # generates python/pyvsag/_version.py
export PYVSAG_LIBRARY="$PWD/build/src/libvsag_c.so"
export PYTHONPATH="$PWD/python"
```

On macOS, use `build/src/libvsag_c.dylib` instead of `.so`.

`PYVSAG_LIBRARY` selects the `libvsag_c` to load; without it, `pyvsag` loads the copy packaged next
to the Python files.

## Quick Start

`pyvsag.Index(name, parameters)` accepts the index name and a JSON-encoded parameter string,
matching the C++ `vsag::Factory::CreateIndex` signature:

```python
import json
import numpy as np
import pyvsag

dim = 128
num_elements = 10_000

data = np.random.random((num_elements, dim)).astype(np.float32)
ids = np.arange(num_elements, dtype=np.int64)

index_params = json.dumps({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": dim,
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 300,
    },
})

index = pyvsag.Index("hgraph", index_params)
index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)

query = np.random.random(dim).astype(np.float32)
search_params = json.dumps({"hgraph": {"ef_search": 100}})
result_ids, result_dists = index.knn_search(
    vector=query, k=10, parameters=search_params,
)
for rid, rdist in zip(result_ids, result_dists):
    print(f"{rid}: {rdist}")
```

For the available index types and parameters, see [Create an Index](create_index.md) and
[Index Parameters](../resources/index_parameters.md). Complete programs live in
[`examples/python/`](https://github.com/antgroup/vsag/tree/main/examples/python); start with
`103_index_hgraph.py`.

## Data Types and Input Rules

| Index `dtype` | Vectors passed to `build`, `add` and the search methods |
| --- | --- |
| `float32` | `numpy.float32` |
| `float16` | `numpy.float16` |
| `bfloat16` | `numpy.uint16` array holding the raw bfloat16 bit patterns |
| `sparse` | CSR arrays: `index_pointers` (`uint32`), `indices` (`uint32`), `values` (`float32`) |

`pyvsag` never converts or copies dense vectors, which keeps the call zero-copy; input that does not
match is rejected with `ValueError`:

- dense arrays must have exactly the dtype above, in native byte order, and be C-contiguous (a 2D
  matrix must be row-major, and its row length must equal `dim`);
- ID arrays and sparse `index_pointers` / `indices` must be integer arrays whose values fit the
  target type (`int64` and `uint32`); floating-point IDs and out-of-range values are rejected instead
  of being truncated;
- `ids` for `build` and `add` must be one-dimensional with one entry per vector.

## Searching

| Method | Returns |
| --- | --- |
| `knn_search(vector, k, parameters)` | `(ids, distances)`: arrays of shape `(k,)` (`int64`, `float32`). Slots the search could not fill hold `-1`. |
| `knn_search(index_pointers, indices, values, k, parameters)` | The same for a batch of sparse queries, with shape `(num_queries, k)`. |
| `range_search(point, threshold, parameters)` | `(ids, distances)` of all results within `threshold`; empty arrays if none. |
| `knn_search_with_statistics(...)` / `range_search_with_statistics(...)` | The arrays above plus the search statistics as a JSON string (a list of strings for sparse batches). See [Search Statistics](search_statistics.md). |

Arguments can be passed by position or by keyword (`vector=`, `k=`, `parameters=`, `point=`,
`threshold=`).

## Sparse Vectors

Sparse vectors use the CSR layout: row `i` owns
`indices[index_pointers[i]:index_pointers[i + 1]]` and the matching `values`; `index_pointers` starts
at `0` and is non-decreasing.

```python
index = pyvsag.Index("sindi", json.dumps({
    "dtype": "sparse",
    "dim": 128,
    "metric_type": "ip",
    "index_param": {"use_reorder": True, "doc_prune_ratio": 0.0, "window_size": 50000},
}))
index.build(index_pointers=ptrs, indices=cols, values=vals, ids=ids)
index.add(ptrs, cols, vals, new_ids)
result_ids, result_dists = index.knn_search(
    q_ptrs, q_cols, q_vals, 10, json.dumps({"sindi": {"n_candidate": 100}}),
)
```

`pyvsag.sparse` additionally decodes the sparse HDF5 benchmark layout; see
[Dataset Format](../resources/dataset_format.md).

## Saving & Loading

```python
index.save("index.bin")

new_index = pyvsag.Index("hgraph", index_params)
new_index.load("index.bin")
```

`save` and `load` raise `RuntimeError` when the underlying serialization or deserialization fails,
including when the file cannot be opened.

## Index Operations

| Method | Result | Notes |
| --- | --- | --- |
| `build(...)` | `None` | Dense `build(vectors, ids, num_elements, dim)` or sparse `build(index_pointers, indices, values, ids)`. |
| `add(...)` | `None` | The same two forms as `build`. |
| `remove(ids)` | `int` | Removes a one-dimensional integer ID array and returns the number actually removed. Returns `0` both when no ID is removed and when the operation fails. |
| `get_num_elements()` | `int` | Current number of elements. |
| `get_memory_usage()` | `int` | Index memory usage in bytes. |
| `check_id_exist(id)` | `bool` | Whether an ID currently exists. |
| `get_min_max_id()` | `(min_id, max_id)` | Returns `(-1, -1)` if the operation fails. |
| `calc_distances_by_id(query, ids)` (legacy alias `cal_distance_by_id`) | `numpy.ndarray` | One float32 distance per ID. Missing IDs produce `-1`. |

```python
# Add two dense vectors.
new_vectors = np.random.random((2, dim)).astype(np.float32)
new_ids = np.array([10_001, 10_002], dtype=np.int64)
index.add(new_vectors, new_ids, len(new_ids), dim)

assert index.check_id_exist(10_001)
print(index.get_num_elements(), index.get_memory_usage())
print(index.get_min_max_id())

# Distances from one query to a candidate list.
candidate_ids = np.array([10_001, 10_002], dtype=np.int64)
distances = index.calc_distances_by_id(query, candidate_ids)

# Remove IDs; the return value is the number actually removed.
removed = index.remove(candidate_ids)
```

`add` accepts a flat contiguous array or a row-major two-dimensional matrix. Operation support
remains index-dependent.

`calc_distances_by_id` takes one 1D float32 query whose length equals the index dimension and a 1D
ID array. Queries of the wrong length and empty queries are rejected with `RuntimeError`, not
treated as missing IDs. Other NumPy inputs are converted to contiguous float32 / int64 arrays, so
views with positive or negative strides keep their logical order. It does not support sparse,
multi-vector or multi-query input.

## Error Handling

| Failure | What happens |
| --- | --- |
| `pyvsag.Index(...)` cannot create the index | `RuntimeError` (`error type: UNSUPPORTED_INDEX`, `error type: invalid_parameter` or `error type: unexpectedError`) |
| `build`, `add`, `save`, `load`, `get_num_elements`, `get_memory_usage`, `check_id_exist`, `calc_distances_by_id`, `*_with_statistics` | `RuntimeError` whose message starts with the operation name and carries the C++ error |
| `knn_search` | No exception: ids and distances are filled with `-1` |
| `range_search` | No exception: empty arrays |
| `remove` | No exception: returns `0` |
| `get_min_max_id` | No exception: returns `(-1, -1)` |
| Wrong dtype, shape, layout, byte order or integer range | `ValueError`; a call with the wrong arguments raises `TypeError` |

Use the `*_with_statistics` variants when a failed search must raise.

## Logging

```python
pyvsag.set_logger_off()     # silence the library
pyvsag.set_logger_info()    # important information only
pyvsag.set_logger_debug()   # detailed debug output
```

`pyvsag.__version__` holds the package version.

## Threads and Processes

- The GIL is released during every call into the C++ library, and the library's internal OpenMP
  threads are unaffected by it. Control their count with `OMP_NUM_THREADS`. Concurrent searches
  from several Python threads are supported; other concurrent operations follow the rules of the
  C++ index.
- `pyvsag` can share a process with other OpenMP users such as PyTorch and scikit-learn;
  `tests/python/test_openmp_coexistence.py` runs every import order.
- With `multiprocessing`, use the `spawn` start method; do not share an index across `fork`.
- A shallow `copy.copy(index)` shares the same native index; the native index is released when the
  last reference goes away.

## Relationship with the C++ Library

`pyvsag` wraps the same `vsag::Index` implementation as C++ and shares the underlying index
binaries, so behavior and performance characteristics match. You can build an index in Python and
load it in C++ (and vice versa) as long as parameters match. Most C++ parameters are passed as the
same JSON fields, and new index types, quantizers and metrics ship with the next `pyvsag` release.

The released wheel is built with `ENABLE_LIBAIO=OFF`, so it has no libaio dependency. The
`async_io` I/O type can still be selected, but it falls back to a buffered compatibility backend
(single `buffer_io`-style reads and no fsync after writes) and logs a one-time warning. To get the
real libaio path, build locally with `VSAG_ENABLE_LIBAIO=ON pip install python/`. See
[Disk Index](../resources/disk_index.md).

## Node.js Equivalents

The Node.js `calcDistancesById(query, ids)` and legacy `calDistanceById(query, ids)` return a 1D
`Float32Array`. They require exactly `Float32Array` queries and `BigInt64Array` IDs (other array
kinds throw `TypeError`), validate the actual query length through the C++ Dataset API, and throw
`Error` with the C++ message on operation failure. Missing IDs still produce `-1`; they are not
operation failures.
