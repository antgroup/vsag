# Daily regression budget and weekly rotation

Tracking: [CI layering plan #3225](https://github.com/antgroup/vsag/issues/3225).
The regression schedule remains disabled. Existing performance schedules are independent.

## Budget contract

The six-hour target is **360 cumulative test-execution minutes**, not workflow
wall time. Parallel test processes, architectures and sanitizers are added.
Compilation, dependency installation, artifact transfer, lint and queueing are
outside this test budget.

| Area | Minutes |
| --- | ---: |
| HGraph | 85 |
| Pyramid | 45 |
| SINDI | 40 |
| Other indexes | 25 |
| Concurrent tests / TSan | 70 |
| SIMD | 15 |
| Unit tests / coverage | 45 |
| Python / examples / historical compatibility | 20 |
| Reserve | 15 |

Budget excess warns; failures, timeouts, skipped selected tests or incomplete
results fail. Existing coverage scope and thresholds are unchanged. The initial
timing history is explicitly unmeasured; this implementation does not claim the
360-minute target is already demonstrated.

## What is selected

* HGraph PR representatives and independent regression cases run daily. Extended
  Daily cases rotate metric/dimension groups; every group's original quantizers,
  tune transitions, workload size and assertions remain intact. Disk IO groups also
  carry their original graph-backend cursor, including offsets caused by skipped
  unsupported RaBitQ entries; splitting must preserve the whole-case IO sequence.
  This first version
  deliberately does not remove repeated TestGeneral assertions without a measured
  replacement contract.
* Pyramid and SINDI finite GENERATE axes are explicitly enumerated. The first
  configuration of every case is core; remaining configurations are assigned to
  seven disjoint slots. SECTION branches remain intact and are recorded in XML.
  Choices in mutually exclusive branches can yield redundant paths: the declared
  product is a conservative superset, not a claim that all axes are independent.
* Unadapted tests are indivisible exhaustive groups. Unknown ownership or unsupported
  generator syntax fails planning instead of silently dropping a test.
* A single SIMD mode rotates across generic/SSE/AVX, with module tests and fixed
  index integration sentinels. This is separate from PR SIMD Coverage.
* TSan still runs concurrent tests. Coverage retains its existing complete
  non-Daily functional support set and all supported unit tests; it is not replaced by a
  seven-day aggregate coverage percentage.
* Linux MKL and macOS repeat daily core validation. Main extended functional
  coverage uses the original OpenBLAS x86 and ARM build lanes. All repeated test
  executions are charged separately.

Capability exclusions are determined before selection, from the matching build's
`CMakeCache.txt` beside the `tests/` binary directory. Only two exact name/source
pairs are excluded: the Kmeans resident-budget test when CUDA is explicitly off,
and the allocator reallocation-failure probe when ASan is explicitly on. The
sealed plan records the build flags, cache digest, full excluded identities and
reasons; the final summary lists these as **not validated**, not passed tests.
Missing build metadata grants no exclusion. CUDA-enabled builds without a suitable
device still fail on unexpected runtime skips; they are not certified by a CPU run.
Any skip within a selected group fails, even if another case in that group passes.
Suppression file paths must be absolute before listing or running isolated children.

No Daily environment means existing local, PR, Branch and release test behavior.
The Examples runner now requires the generated binary manifest and propagates
each example's exit status.

## Reproduction and evidence

The plan records the source SHA, original workflow creation time, Shanghai date,
ISO week, slot, policy/matrix/history hashes, seed, environment and configuration
IDs. A rerun uses the original run date. Manual regression accepts logical_date
in YYYY-MM-DD format for a specific weekday. It does not enable a schedule.

For a development build:

```bash
python3 scripts/testing/daily_plan.py \
  --source-sha "$(git rev-parse HEAD)" \
  --created-at 2026-11-03T00:00:00Z \
  --logical-date 2026-11-03 --output daily-context.json
python3 scripts/testing/daily_run.py \
  --context daily-context.json --binary build/tests/functests \
  --lane local-functests --environment linux-clang20-asan \
  --profile functional --output daily-results/local --workers 2
```

Use --plan-only to inspect all seven assignments without test execution. Use a
fresh output directory per attempt. daily-results artifacts retain plans, logs,
Catch2 XML and terminal records for 30 days. Artifact names are scoped to the
workflow run; a failed-job rerun replaces only its own lane evidence and reuses
successful lanes and the original context. The final Daily summary reconciles
every expected lane; absent artifacts are failures.

A successful weekday is not full weekly coverage. Seven days at different
source SHAs are not proof that one release SHA passed the full matrix. Release
validation must use a fixed SHA and the full supported configuration set.

## Issue-linked regressions

Bug regressions should describe behavior in the test name and link the Issue in
a nearby comment, with the triggering conditions and protected invariant. Use
the Issue to explain why the test exists, not as a substitute for assertions.
Critical known regressions belong in daily core. CI bugs need helper/workflow
regressions; they do not need a C++ test just to obtain an Issue number.
