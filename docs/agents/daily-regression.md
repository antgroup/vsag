# Daily regression budget and weekly rotation

Tracking: [CI layering plan #3225](https://github.com/antgroup/vsag/issues/3225).
The regression schedule remains disabled. Existing performance schedules are independent.

## Budget contract

The operating target is **22:00–08:00 Asia/Shanghai**, or **600 elapsed minutes**
per attempt: 570 minutes for execution, 30 for cleanup/reporting. Manual runs use
the same ten-hour duration starting at the attempt's start, without waiting for 22:00.
The execution cutoff is frozen separately from the original source/date context,
so retrying a failed lane gets a fresh execution window while retaining its original
weekly selection. A local invocation without `window.json` gets a fresh process-local
270-minute execution allowance, matching a single regression job; it does not receive
the whole workflow's 570-minute execution allowance.

Compilation, dependency setup, transfer, queueing and reporting count toward elapsed
workflow time. The Summary obtains actual attempt/job timestamps from GitHub;
individual job times are not summed into workflow elapsed time. Queue delay is not
inferred from dependency waiting. Each regression job is bounded at five hours. The execution cutoff is fixed at
4.5 hours after that job starts and includes setup/inventory time, leaving room
for upload and cleanup. All binaries in a job share this cutoff; starting the
next binary never renews it. The cutoff is also clamped to the attempt window.
Tests that cannot start before the execution cutoff produce explicit failing
`budget_exhausted` records; missing results and timeouts remain failures.

Cumulative test-process minutes and the old per-area reference allocations remain
separate resource diagnostics, with warnings above references. **Their old 360-minute
sum is not the new wall-time deadline.** Linux expansion time reports its measured
ARM share against an aspirational 10% target; this is not CPU quota enforcement.

x86 unit and functional execution each default to four longest-first balanced shards.
`daily_budget_policy.json:x86_shards` is the single source for the matrix, shard
count and expected lane list. Window clocks are validated against the configured
elapsed duration and rendered directly from policy.
Historical successful durations and timeout lower bounds order and balance work.
A timeout or an old unsplit parent's time never certifies an exact new tuple estimate.
The report validates every shard against its full frozen parent selection and requires
all shard indices exactly once. A failed shard does not cancel its siblings.

ARM uses one shared OpenBLAS build for tests and examples. Its former duplicate
build/core lane and artifact-overwriting cleanup are removed. x86's bundled-OpenBLAS
source build and MKL build remain separate because they validate different backends.
Coverage scope and thresholds remain unchanged. Regression scheduling stays disabled.

## What is selected

* HGraph PR representatives and independent regression cases run daily. Extended
  Daily cases rotate metric/dimension groups, further split by quantizer or tune
  transition for supported long-tail helpers. The union preserves every original
  tuple, workload size and assertion. Disk IO groups also
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
* ARM lanes retain one build-preferred core smoke per source file/owner plus explicit
  regression tags and issue-named regressions. Approximately one tenth of remaining
  configurations is eligible per week, split into seven daily slots. Ten weekly
  strata rotate deterministically; this is not full ARM weekly coverage. The Summary
  lists selected and outside-this-week counts. Full supported matrices remain a
  release obligation. Architecture expansion resource target is x86:ARM = 9:1;
  configuration counts are only a proxy until actual timings are calibrated.
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

## Acceptance boundaries

The ten-hour window, shard planner, artifact graph and reporting need a complete
manual Daily replay on this revision before scheduling is enabled. A fast individual
configuration is not proof of whole-night performance. Completed PR CI does not
substitute for macOS/ARM/full Daily acceptance.
Multiline DAILY_GENERATE plans include both opening and closing source-line aliases
for GCC/Clang. Aliases select the same value and collisions fail closed.
