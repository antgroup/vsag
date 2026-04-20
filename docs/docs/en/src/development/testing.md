# Running Tests

VSAG uses [Catch2](https://github.com/catchorg/Catch2) for testing, organized in two layers:

- **Unit tests** live next to source files under `src/`.
- **Functional tests** live under `tests/` and cover cross-module, end-to-end behavior. Typical
  files include `test_hnsw.cpp`, `test_hgraph.cpp`, `test_diskann.cpp`, `test_ivf.cpp`,
  `test_pyramid.cpp`, `test_sindi.cpp`, `test_brute_force.cpp`, `test_multi_thread.cpp`,
  `test_memleak.cpp`.

## Run the Full Suite

Build a debug (or dev) configuration first, then:

```bash
make debug      # or: make dev
make test
```

This runs unit + functional tests and collects coverage data. Generate an HTML coverage report
with:

```bash
make cov
# open build-debug/coverage/index.html
```

## Run a Single Binary

```bash
./build-debug/tests/functional_tests "[hgraph]"
./build-debug/tests/functional_tests "[hnsw][concurrent]"
```

Catch2 supports filtering by name, tag, and wildcards — see `--help`.

## Coverage Expectations

Contributions are expected to keep the C++ line coverage over `src/` and `include/` at **90%** or
higher, as measured by the `make test` / CI coverage job.

## Memory & Concurrency

- `test_memleak.cpp`: run under AddressSanitizer / LeakSanitizer to verify construction and
  destruction paths.
- `test_multi_thread.cpp`: concurrent `Build` / `KnnSearch` / `RangeSearch` correctness.

## Python Tests

```bash
make pyvsag PY_VERSION=3.10
cd tests/python && pytest -q
```

## References

- `tests/` directory
- Makefile entries: `test`, `cov`, `asan`
