# Running Tests

VSAG uses [Catch2](https://github.com/catchorg/Catch2) for testing, organized in two layers:

- **Unit tests** live next to source files under `src/`.
- **Functional tests** live under `tests/` and cover cross-module, end-to-end behavior. Typical
  files include `test_hgraph.cpp`, `test_ivf.cpp`, `test_pyramid.cpp`, `test_sindi.cpp`,
  `test_brute_force.cpp`, and `test_memleak.cpp`.

## Run the Full Suite

`make test` configures a Debug build with tests enabled and runs the full unit + functional
suite:

```bash
make test
```

Note: `make test` does not enable coverage instrumentation. To produce a coverage report, use
`make cov` — it configures the build with `ENABLE_COVERAGE=ON`; then run the same non-daily unit
and functional suites as coverage CI with its fixed seed before collecting the trace:

```bash
make cov
VSAG_TEST_SEED=424242 bash scripts/testing/test_parallel_bg.sh
bash scripts/coverage/collect_cpp_coverage.sh
bash scripts/coverage/check_cov.sh
```

The collector writes `coverage/coverage.info` with repository-relative paths and branch data.
It measures maintained production sources under `src/` and public headers under `include/`,
excluding generated `src/version.h` and vendored `include/vsag/expected.hpp`. Code compiled only
on another platform is explicitly outside the Linux x86 report and must be measured by a
platform-specific coverage job rather than treated as covered.

## Run a Single Binary

```bash
./build/tests/functests "[hgraph]"
./build/tests/functests "[hgraph][concurrent]"
```

Catch2 supports filtering by name, tag, and wildcards — see `--help`.

## Coverage Expectations

Contributions are expected to keep the C++ line coverage over `src/` and `include/` at **90%** or
higher, as measured by the `make cov` flow and the CI coverage job.

## Memory & Concurrency

- `test_memleak.cpp`: run under AddressSanitizer / LeakSanitizer to verify construction and
  destruction paths.
- `test_index/test_index_concurrent.cpp`: shared concurrent add and search correctness for
  maintained indexes that advertise the corresponding feature flags.

## Python Tests

```bash
make pyvsag PY_VERSION=3.10
cd tests/python && pytest -q
```

## References

- `tests/` directory
- Makefile entries: `test`, `cov`, `asan`
