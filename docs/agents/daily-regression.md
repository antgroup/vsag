# Nightly regression budget and weekly rotation

Tracking: [CI layering plan #3225](https://github.com/antgroup/vsag/issues/3225).
Nightly CI runs the complete regression profile on upstream `main` every day at
**21:17 Asia/Shanghai (13:17 UTC, `17 13 * * *`)**. Only that cron in the canonical
repository selects regression; the weekly build-performance cron remains separate.
The former weekday runtime Performance cron is removed; its manual profile remains
available. Future performance regression on dedicated Team CI resources is separate
work, not enabled here.

The workflow filename `daily_test.yml`, helper names, `VSAG_DAILY_*` variables,
`[daily]` test tags and artifact identifiers stay unchanged for compatibility.
The Actions entry is **Nightly CI**, with one final **Nightly Summary**.
The same run updates the README coverage badge after Summary when its Coverage
job succeeds. Publication only runs for the upstream main regression schedule.
GitHub may delay scheduled dispatch: 21:17–07:17 is the intended operating window,
not a guaranteed local-clock finish. The enforced ten-hour allowance starts at the
actual attempt start; a delayed dispatch can therefore finish after 07:17.

## Budget contract

The operating target is **21:17–07:17 Asia/Shanghai**, or **600 elapsed minutes**
per attempt: 570 minutes for execution, 30 for cleanup/reporting. Manual runs use
the same ten-hour duration starting at the attempt's start, without waiting for 21:17.
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
Coverage case names and thresholds remain unchanged; six expensive HGraph functional cases use a smaller, explicitly bounded Coverage dataset as described below.

macOS ASan restores the same compiler cache in Branch CI and Nightly. Only
Branch CI on pushes to the upstream main branch can publish, once per cache
key. A change to the top-level `CMakeLists.txt` or
`.circleci/fresh_ci_cache.commit` rotates the key. Existing timestamped caches
for the same key can seed the first fixed snapshot.

## Python installed-wheel matrix

Nightly builds one manylinux 2.28 `py3-none` wheel per Linux architecture
(x86_64 and aarch64), using the same action and container-mounted compiler cache
as PR, Branch and TestPyPI. It also builds one native macOS 15 arm64
`py3-none` wheel with the shared macOS wheel build and audits its platform tag
and libraries. TestPyPI publication remains independent.
Every environment installs the matching source/architecture/SHA-256-verified wheel.

| Python | NumPy 1.x | NumPy 2.x | Ordinary suite |
| --- | --- | --- | --- |
| 3.10, 3.11, 3.12 | 1.26.4 | Latest compatible 2.x; resolved version recorded | Complete directory, excluding the separate OpenMP module |
| 3.13, 3.14 | Unsupported, not scheduled | Latest compatible 2.x; resolved version recorded | Same complete ordinary suite |

Each architecture also has one mandatory Python 3.12 / NumPy 2 OpenMP lane with
CPU torch and scikit-learn installed. Missing dependencies, fewer than six import
orders, any skipped integration test or a failed process makes the lane fail.
This is eight ordinary environments plus one OpenMP environment per architecture:
**three builds and 27 required test environments**, with no daily rotation of Python.
The macOS environments use clean virtual environments, and the OpenMP lane uses the
pinned macOS torch and scikit-learn packages.

The matrix shares the existing ten-hour execution window. The remaining-window
check bounds dependency installation and testing, and optional x86 C++ supplemental
planning waits for all 27 environments. Mandatory failures do not become budget
skips; incomplete work fails final reconciliation. No new wall-time quota is added.

Per-job artifacts carry JUnit plus JSON counts, source/wheel hashes, installed import
paths and Python/NumPy/cffi versions (plus torch/sklearn for OpenMP). The final
Nightly Summary reconciles every unique expected environment and reports Python
cost separately from the optional x86 supplemental stage. Container compiler-cache
counters feed the shared diagnostics table. Ordinary pytest discovers new test
files automatically. Python executions are reported separately from the existing
four-number Catch/C++ logical-case inventory; they do not inflate that inventory.

## What is selected

* HGraph PR representatives and independent regression cases run daily. Extended
  Daily cases rotate metric/dimension groups, further split by quantizer or tune
  transition for supported long-tail helpers. Except for explicitly bounded concurrent workloads and recorded exact quarantines below, the union preserves every original
  tuple, workload size and assertion. Disk IO groups also
  carry their original graph-backend cursor, including offsets caused by skipped
  unsupported RaBitQ entries; splitting must preserve the whole-case IO sequence.
  This first version
  deliberately does not remove repeated TestGeneral assertions without a measured
  replacement contract.
  Nightly Reader IO, Duplicate, and Multi-Query Knn Search keep their original
  metric/dimension groups for weekly selection and exact quarantines. After
  selection, each supported quantizer in a group executes independently with
  the 30-minute limit. The test shapes, assertions, and job deadlines apply
  as before. The Summary counts each selected execution unit and still
  counts each Catch2 test name once.
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

## Functional ownership check

The Team and Community Linux ASan builds in PR CI list the compiled
`functests` inventory, including Daily-only cases, and validate ownership
with the Nightly resolver. Keep the tags or explicit owners in
`scripts/testing/daily_matrix.json` current when adding functional tests.
The check lists registrations without running test bodies or building a
Nightly plan.

## Parameter-adapter preflight

Before building, the Daily plan job runs:

```bash
python3 scripts/testing/daily_parameters.py
```

The first command validates every test declaration in the adapted Pyramid and
SINDI sources, including inactive preprocessor branches. New finite axes in those
files must use `DAILY_GENERATE` with the original choices. Without Daily selection
variables this macro retains the ordinary Cartesian product; it must not reduce
workload sizes or relax assertions. Errors identify the source file, case name and
location. Heavy regression jobs depend on this preflight through the plan job.

For adapter/helper changes, also run
`python3 -m unittest tests.scripts.test_daily_parameters tests.scripts.test_daily_plan tests.scripts.test_daily_extra tests.scripts.test_daily_workflow`.

The source preflight is deliberately conservative and does not certify platform
execution. Each compiled binary still supplies its authoritative test inventory,
and the runtime planner validates its parameter choices before sharding. Validate
new adapters with both the ordinary case and all explicit configurations, then
check the union of core and seven-day rotation before manual Daily acceptance.

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

## Case counts in the workflow Summary

The final Nightly Summary starts with four numbers: **passed / failed / selected this run /
global discovered**. They count unique active Catch2 cases, identified by repository-relative
source file and exact case name. Repeated configurations, shards and platform executions
are not additional cases; assertion totals are not testcase totals.

A case passes only when all its selected executions pass. Any failed or timed-out
single-case execution makes the case fail. Case-insensitive Catch2 groups use individual
XML results: a normal nonzero exit with an explicit member failure preserves successful
siblings' results, while the group and run still fail. A nonzero exit with only successful
members, a signal, or invalid/missing evidence cannot establish a passing sibling.
An interrupted group without attributable results remains pending rather than making
every member fail. Selected cases without complete terminal results remain
**pending/not executed**; consequently passed + failed can be smaller than selected.
Unexpected runtime skips and budget exhaustion still fail completeness and never count
as successful coverage. Non-Catch command groups (examples, compatibility and Python)
remain separately listed because their command status cannot establish a testcase count.

Global inventory is collected from the same source's compiled test binaries **before**
capability exclusions, quarantines, architecture sampling and weekday selection. It is
the discovered active C++ inventory, not every possible parameter tuple or unsupported
platform feature. New cases appear automatically when their binaries are discovered.
Hidden cases and commands without named-case inventories are outside these four counts.
Legacy artifacts without this inventory expose only a lower bound, explicitly marked.

Missing execution records, shards or lanes retain the available validated counts with a
prominent **Incomplete evidence** warning and a failing result. Missing lanes can add
cases or change apparent passes; partial counts are not final acceptance. Corrupt plans,
mixed contexts and duplicate results remain rejected; if no trustworthy summary can be
formed the four fields show **unknown**, never fabricated zeroes. The JSON artifact
contains the same `case_counts` values for local analysis.

When a required job fails or is cancelled and both x86 supplemental jobs are
skipped, Nightly Summary reports the supplemental stage as skipped. It retains
the required job failure. When the planner is cancelled, Summary records the
cancellation and skips its quota plan lookup. If there is no supplemental
execution evidence and the dependent jobs are skipped or cancelled, Summary
also omits those unstarted shards from reconciliation. Missing required results,
missing evidence from an attempted supplemental stage, and unexplained
supplemental skips still fail reconciliation.

A per-case timeout in the base x86 OpenBLAS functional rotation is advisory
when captured evidence contains no assertion or sanitizer failure. The
configuration still stops at its existing deadline and counts as failed. A
single-case timeout counts as a case failure. For grouped cases, the timed-out
execution leaves each case without an attributable result pending. Summary
lists these timeouts separately and reports
`passed_with_warnings` when there are no blocking errors. Supplemental planning
may continue from complete base artifacts and still excludes every selection
from the base plan, including timed-out configurations. Core timeouts,
execution-window deadlines, failures in other lanes, and missing artifacts
remain blocking.

macOS full examples execute in an independent step after successful artifact download,
even when unit tests fail. Cancellation still stops further work. Their outcome remains
part of the final expected-lane completeness check.

## Issue-linked regressions

Bug regressions should describe behavior in the test name and link the Issue in
a nearby comment, with the triggering conditions and protected invariant. Use
the Issue to explain why the test exists, not as a substitute for assertions.
Critical known regressions belong in daily core. CI bugs need helper/workflow
regressions; they do not need a C++ test just to obtain an Issue number.

## Acceptance boundaries

The complete manual replay [43714378490](https://github.com/antgroup/vsag/actions/runs/43714378490)
on #3293 head `af70a508` passed in 7h48m, including all four supplemental x86 shards.
Scheduling is enabled by the separately reviewed activation change; that replay
validates its source and selected rotation, not all seven days or later main commits.
Its 86.29% scoped line coverage remains below the 90% target. A fast individual
configuration is not proof of whole-night performance. Completed PR CI does not
substitute for macOS/ARM/full Daily acceptance.
Multiline DAILY_GENERATE plans include both opening and closing source-line aliases
for GCC/Clang. Aliases select the same value and collisions fail closed.

## HGraph long-tail diagnostics

Daily configuration groups split Build, Reverse Edges, ODescent Build, Serialize File, Multi-Query Range, With Extra Info, Compressed Graph Build and concurrent Add/Search/Remove by quantizer. Disk groups select one legal quantizer/backend tuple while retaining its original graph-IO cursor. The seven-day union preserves the effective combinations and assertions. Extra Info retains its original dataset sizes; splitting only isolates quantizers. The 1,800-second per-configuration timeout remains unchanged; logical cost estimates are not measured runtime guarantees.

Daily Reverse Edges enables reverse edges in the build parameters, checks recall, and serializes that single built graph. A separately loaded index checks recall again. Support Get Raw Vector retains its metric/dimension groups for weekly selection, then runs each of its three fixed quantizers with a separate 30-minute limit.

Daily Tune restores a verified snapshot of the original built index for the separate disable-future-tuning check, avoiding a second identical build. Ordinary PR/Branch invocations retain the original setup. All Tune, restored-index and general-contract assertions remain enabled.

Each executed configuration writes `phases.jsonl` beside `report.xml` and `result.json`. Phase events include operation, start/end, timestamp and relevant quantizer/backend details; completed phases include elapsed seconds. The sidecar is flushed independently of Catch2 capture, so a timeout can retain progress even when XML is incomplete. `result.json` records `active_phases`, `last_phase_event` and any malformed-tail diagnostic. The last entered operation narrows investigation but does not by itself prove a deadlock.

### Bounded concurrency and exact exceptions

When `VSAG_DAILY_ENUMERATE` or `VSAG_DAILY_HGRAPH` selects the planned Daily path,
HGraph Concurrent Add and Concurrent Add/Search/Remove use at most 512 rows and
a raw dimension cap of 256 before the existing cost cap. RaBitQ keeps its existing
960-dimensional recall floor after quantizer normalization; the cap must not
weaken that accuracy precondition. The cost model charges the effective dimension.
PQ still has at least 256 initial
training rows. Query counts, metrics, quantizers, five worker threads, operations
and assertions are unchanged. Ordinary PR/Branch and direct full/release runs
without these environment variables keep their original workloads. The seven-day
inventory describes this bounded workload; large-shape stress remains a release
validation obligation, not an implied Daily pass.

`daily_matrix.json` may quarantine an exact extension configuration key with its
case name, reason and tracking issue. Wildcards, ambiguous identity and core
exclusions are rejected. A dedicated Summary section lists quarantines as
**not validated**, including when their weekly slot is not today's slot.
Explicit full-coverage planning ignores quarantines.

The L2 and cosine HGraph Duplicate groups at 1536 dimensions and 600 rows were
restored after replaying their original seed and supported quantizers. The
ordinary and filtered KNN reference now includes the appended labels and counts only the
available slots for exact distance ties. The original recall thresholds are
preserved. [Issue #3504](https://github.com/antgroup/vsag/issues/3504).

Weekly seeds can change the effective parameter inventory. If the testcase still has
rotation configurations but an exact quarantined key is absent, the rule is reported
separately as **inactive / not in this inventory**; no current configuration is excluded
by that rule and no pass is credited. The rule remains configured and applies again
if that exact key returns in a later week. Missing/renamed testcases, malformed rules,
identity mismatches and rules targeting core configurations still fail planning.
This does not broaden an exclusion to other configurations of the same testcase.
Remove an entry after a root-cause fix and a successful targeted replay.
Unexpected test skips, failures or missing planned results still fail.

Scheduling uses exact configuration/environment observations when available;
case-level observations are fallback ordering hints for previously unseen tuples.
A slow sibling must not give every tuple the same estimated cost. Timeouts can
supply an ordering lower bound but never a successful runtime estimate. The latest
751 imported observations come from run 42989404023 and retain source/run/job
provenance; inherited history is not a measurement of this revision. Re-measure
changed concurrency shapes and newly split Extra Info groups in a full manual
Daily before accepting the ten-hour target.

### SIMD build memory diagnostics

Nightly SIMD keeps the full ASan build, `unittests_simd`, and the fixed functional
sentinels. The full build explicitly uses `COMPILE_JOBS=3 DATACELL_COMPILE_JOBS=1`;
the subsequent SIMD-module target also uses three jobs. Both commands run through
`monitor_build.py`, which streams memory/swap, cgroup OOM counters and compiler
process memory into the log and saves separate `asan/` and `simd/` diagnostics.
The matrix-index/run/attempt-specific diagnostic artifact is retained for seven
days and upload is attempted even on failure; upload failure cannot mask the
build outcome. Abrupt runner termination can still prevent upload, so retain the
streamed samples. Exit 137 alone is not proof of an OOM kill.

### Coverage-specific workload bound

`VSAG_DAILY_COVERAGE=1` caps the original PR variants of HGraph Duplicate Build,
Export Model, Search Over Time, Mark Remove, Reverse Edges, and Estimate Memory And Get Memory Usage at
**256 rows after fixed-workload normalization and before cost-budget selection**.
Coverage's `-O0` and atomic counters amplify the repeated distance/pruning work;
these functional contracts do not require large-data stress. The bound preserves
the selected metrics, quantizers, raw dimensions, RaBitQ's 960-dimensional floor,
query count, construction thread count, assertions and the coverage threshold.
The memory case still independently builds the estimate and serialization/load
paths and retains both memory-accounting assertions. Its fixed 1,000-row default
must not overwrite the Coverage cap or disagree with the recorded budget.
256 retains the PQ training minimum. This is an explicit dataset-size reduction,
not a claim that the original large workloads became faster.

Other Coverage cases, concurrent tests, ordinary PR/Branch/ASan/TSan and full-size
release invocation retain their workload rules. The runtime flag is already scoped
to the Daily Coverage lane; empty or other flag values do not activate this bound.
Native plan regressions verify the positive and negative scopes. A complete manual
Daily remains necessary to validate hosted duration and the aggregate coverage result.
