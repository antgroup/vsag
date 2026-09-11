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

### Pre-merge SIMD stage

`PR SIMD Coverage` runs for SIMD source/test changes and changes to its coverage infrastructure.
It builds the PR base and GitHub merge candidate sequentially on one Linux x86 runner, using
four build jobs, OpenBLAS, `[simd]~[!benchmark]`, Catch2 seed `424242`, and lexical order.
Each test run has a 15-minute limit; the job has a 90-minute limit. It still builds the aggregate
test executable. Full post-merge/manual unit and functional coverage is unchanged.

The Actions summary and artifact report **only maintained `src/simd/` line coverage**.
Measured added executable lines must reach 80%; scoped line coverage cannot decrease against
the fresh base measurement, using unrounded ratios. New files enter head and patch totals
without an invented per-file baseline. Missing traces, SHA/profile mismatches, unmeasured
changed files, or disappearing non-deleted source records fail. No measured added executable
lines means N/A, not 100%.

This is a bounded first stage, not full-project regression protection. Other changed C++ paths
are listed as unsupported when the job runs; PRs outside its trigger paths do not receive this
measurement. Conditional paths remain limited to this runner. Some tests use `random_device`,
so the Catch2 seed does not control every input. Partial traces are never uploaded to Codecov
or compared with full-suite coverage. The job uses read-only PR permissions, no secrets,
no persistent checkout credentials, and no restored/saved build caches.
Changes to dependency definitions report an unavailable comparison because the two builds
cannot safely share one prepared dependency checkout when its pins differ.

To compare collected traces, put `base.info`, `head.info`, and JSON metadata files
`base.json`/`head.json` in a report directory. Each metadata file contains `sha` and `profile`
(the exact `PROFILE` constant in `scripts/coverage/compare_simd_coverage.py`). Then run:

```bash
python3 scripts/coverage/compare_simd_coverage.py --root . \
  --base "$BASE_SHA" --head "$MERGE_SHA" --reports simd-coverage
```

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
