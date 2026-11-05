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

## Build and Run One Unit-Test Module

For a shorter edit-build-test cycle, build and run one maintained unit-test subsystem:

```bash
make test-module MODULE=datacell
make test-module MODULE=algorithm CASE='[ut][sample_train_data]'
```

The available module names are `simd`, `common`, `algorithm`, `factory`, `attr`, `datacell`,
`layout`, `quantization`, `storage`, `io`, `utils`, and `impl`. Each command builds the shared
production and fixture dependencies plus only the selected module's test sources. The corresponding
CMake target and executable are named `unittests_<module>`:

```bash
cmake -S . -B build -DENABLE_TESTS=ON
cmake --build build --target unittests_datacell
./build/tests/unittests_datacell '[ut][AttributeInvertedInterfaceParameter]'
```

These module targets are a local acceleration path. Continue to run `make test`, whose aggregate
`unittests` executable contains every module, for full validation.

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
excluding the generated `version.h` header in `src/` (created at build time from the tracked
template `src/version.h.in` by `cmake/GenerateVersionHeader.cmake`) and vendored
`include/vsag/expected.hpp`. Code compiled only
on another platform is explicitly outside the Linux x86 report and must be measured by a
platform-specific coverage job rather than treated as covered.

### Pre-merge SIMD stage

`PR SIMD Coverage` runs for pull requests targeting `main` when SIMD sources/tests or their build and coverage infrastructure change.
This rollout is intentionally limited to `main`. Release branches retain their existing PR CI, but do not receive this new SIMD coverage check; extending it to release branches requires a separate rollout.
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

## Report Testcase Budgets and Time

`scripts/testing/test_budget_report.py` reads saved test listings, results and cost records
without running tests. It works without Team CI inventory tooling:

```bash
build/tests/functests '~[daily]' --list-tests --reporter xml > functests-list.xml
build/tests/unittests '~[daily]' --list-tests --reporter xml > unittests-list.xml
python3 scripts/testing/test_budget_report.py \
  --list-tests tests/functests=functests-list.xml \
  --list-tests tests/unittests=unittests-list.xml \
  --source-root "$PWD" --output budget-report.md
```

`--list-tests` consumes Catch2 listing XML, not execution XML. The listing commands enumerate
tests without executing them. All listed cases are retained, including template expansions;
the report applies no extra daily exclusions. Add `--xml tests/functests=report.xml` for an
existing Catch2 execution report, and `--cost tests/functests=cost.log` for budget logs from
the same source revision. List XML hashes identify the input listings, not binary contents.

Catch2's XML reporter captures budget messages in `StdOut`. Import the same saved execution
XML explicitly for both measurements; `--xml` imports time only, while `--cost` decodes XML
escaping and reads only `[test-budget]` lines from `StdOut`:

```bash
python3 scripts/testing/test_budget_report.py \
  --list-tests tests/functests=functests-list.xml \
  --xml tests/functests=report.xml --cost tests/functests=report.xml \
  --source-root "$PWD" --output budget-report.md
```

An XML report without marked budget records leaves costs **unknown**; malformed marked
records fail rather than silently disappearing.

Existing Team CI artifacts can instead supply the frozen inventory and execution results:

```bash
python3 scripts/testing/test_budget_report.py \
  --inventory team-plan-RUN-ATTEMPT.zip \
  --result team-results-0-RUN-ATTEMPT.zip \
  --result team-results-1-RUN-ATTEMPT.zip \
  --result team-results-2-RUN-ATTEMPT.zip \
  --result team-results-3-RUN-ATTEMPT.zip \
  --output budget-report.md
```

The inventory can also be schema 3 inventory/plan JSON, and each result can be a shard
`result.json`. For Catch2 XML use `--xml tests/functests=report.xml` instead of duplicating
those cases through `--result`. Add `--binary tests/functests` for the functional subset,
`--group-by family` for index families, or `--format json` for per-case assignments.

Counts retain binary identity: a case compiled into both unit and functional binaries counts
twice. The ordered, mutually exclusive categories are Tune, recovery, serialization,
mutation/concurrency, search/filter, build/train, data/quantization/SIMD, and Other. They are
name/tag/source heuristics, not assertion coverage measurements.

Optional `--cost tests/functests=cost.jsonl` adds schema 1 complete-case cost records. Each
record has `model_version`, `unit`, `policy`, `case` (`name`, `tags`, `source_file`,
`source_line`), `cap_points`, and `total_points`. Test logs may prefix records with
`[test-budget]`; in that mode unrelated log lines are ignored. Only the top-level cap and
estimated cost count, not nested `runs`. Identical complete plans emitted again by Catch
generator re-entry count once and expose `identical_plan_repetitions`; differing plans for
one case fail. A bridge record with empty tags and `source_line: 0` must match exactly one
inventory case by binary, name, and full source identity. Catch's original absolute or
parent-relative paths are identity strings, never paths the report opens. Exact full-path
matching is preferred; use `--source-root /absolute/checkout` only when an absolute path must
map to a repository-relative identity. Basename matching is forbidden, and frozen inventory
IDs and digests remain unchanged.

Case-count shares, cap shares, estimated-cost shares, and measured case-second shares have
separate denominators. Models, units, and policies are never combined. Missing measurements
remain **unknown**, not zero; a timing-only baseline is valid when cost data is unavailable.
Case-second sums are **neither elapsed CI time nor CPU time**. Failed cases and over-budget
records remain visible; this reporting command does not replace CI's completion checks.

Run the report's local fixture tests with `python3 tests/scripts/test_test_budget_report.py`.

## Budget Policy

`tests/fixtures/core/test_budget.h` models logical work using N, D, Q and declared counts
for data preparation, training, building, searching, reference-oracle calculations,
serialization and mutations. A testcase plans its complete configuration matrix under one
cap. Each candidate retains the required configurations and legal minimum sizes; if no legal
candidate fits, planning fails. Catch generator re-entry shares that complete plan rather
than granting a fresh cap. `CurrentPolicy` takes its seed from Catch's `--rng-seed`, so resource
selection is reproducible for the same testcase and seed.

The first integration covers randomized resource families in HGraph, IVF, BruteForce, Warp
and Dataset tests, not every static testcase. Uninstrumented cases remain unmodeled in reports.
PR HGraph Tune keeps four representative transitions with Q=20 and N≤400; PiPNN uses one
transition with Q=10 and N=256. Daily keeps all ten transitions with Q=100.

Logical points and declared upper bounds are neither elapsed seconds nor a universal runtime
limit. The budget does not replace assertions or timeouts, and exhausting it cannot silently
skip a testcase or turn an incomplete run into a pass.

## References

- `tests/` directory
- Makefile entries: `test`, `cov`, `asan`
