# Community and team PR CI

`pr-ci.yml` routes PR checks to reusable workflows:

- `pr-ci-community.yml` runs GitHub-hosted checks. When team Linux is selected, macOS, Python, formatting and coverage-configuration checks still run here.
- `pr-ci-team.yml` builds Linux x86 ASan binaries once with `COMPILE_JOBS=24`, runs complete formatting/helper checks in a parallel GitHub-hosted job using the same image and tested commit, then runs four test-shard jobs and an independent index-compatibility job in parallel, followed by a GitHub-hosted summary job. Changed-source lint, unit/tool tests, functional tests, examples and index compatibility remain required when relevant files change. Diagnostics are uploaded as artifacts.

The multi-job team workflow is under validation. Its storage and runner prerequisites must be deployed and verified before enabling it. The configured parallelism does not establish a particular CI completion time.

## Routing

The router runs without checking out PR code. It resolves the canonical repository's `main` commit and reads `.github/ci-routing.json` and `.github/maintainers.json` at that exact commit. Personal forks are supported; eligibility depends on the PR author, not the actor who starts or reruns CI. The maintainer policy was supplied in [PR #3146](https://github.com/antgroup/vsag/pull/3146); the routing configuration is supplied separately in [PR #3150](https://github.com/antgroup/vsag/pull/3150).

The live PR must be open and its head/base commits must match the event. Missing or malformed policy, API errors, stale events, an invalid image, or a disabled switch select community CI. Labels do not affect routing. Manual workflow dispatches also select community CI.

Routing is not an authorization boundary against modified workflow files. Keep **Require approval for all outside collaborators** enabled and review workflow changes before approving execution. Approval automation is outside this change. Do not use `pull_request_target` to build PR code.

## Configuration

The source of truth is `.github/ci-routing.json` on canonical `antgroup/vsag` main. Repository variables are not used: they are unavailable to our personal-fork PR workflows. Changes in a PR or its fork cannot change the selected configuration.

```json
{
  "team_enabled": false,
  "team_image": "ghcr.io/antgroup/vsag@sha256:ccfe5a122031782debf8c24d41e648029f79a98d0bc5a0a7217bcb1a05ca4e17"
}
```

Only the JSON boolean `true` enables team routing; strings such as `"true"` are rejected. Both fields are required and unknown fields are rejected. The image must use the canonical repository and a digest of 64 lowercase hex characters. Missing, malformed or inaccessible configuration selects community. The configuration prerequisite enables team routing; changes take effect only after reaching main.

The team runner must support non-root container jobs and provide the build dependencies. The image prerequisite is supplied separately in [PR #3147](https://github.com/antgroup/vsag/pull/3147). Use the digest of the published x86 OpenBLAS image, not the base image's. The CI image must include clang-tidy 20 and clang-format 20 exactly, GNU time, ccache and GitHub CLI 2.75.0. The CLI version and archive SHA-256 are pinned in `docker/Dockerfile.x86.base`; update both together with the workflow version check. No GitHub token is used during image construction. The team compatibility check uses a pre-staged release snapshot and does not receive `GH_TOKEN`.

The image repository restriction is intentional; changing registries requires a reviewed router change. Enable routing only after the selected image and runner have passed the complete team workflow. Infrastructure provisioning and operational settings are maintained separately.

The token-path prerequisite check detects the standard ServiceAccount token mount only; it does not prove that credentials are absent at other paths. Runner administrators must enforce credential isolation independently of this smoke check.

## Team execution contract

The caller serializes the complete team reusable-workflow invocation with `concurrency.queue: max` and `cancel-in-progress: false`. Different PRs wait for the shared team execution capacity; four test jobs and the independent compatibility job within the admitted run may execute concurrently when execution capacity permits. The outer per-PR concurrency rule still cancels an older run when that PR is updated. Runner cleanup must finish before shared storage can be reused; GitHub concurrency alone does not establish this condition.

For relevant C++ and CI changes, the workflow follows this sequence:

1. **Build and hosted checks:** run the complete formatting and helper-script checks in a separate GitHub-hosted job while Build restores an optional private compiler cache, compiles ASan binaries once with 24 compiler jobs and precompiled headers disabled, and lints changed sources with clang-tidy 20. CMake receives the absolute build-directory path; the checkout's `build` symlink is only a convenience for existing tools.
2. **Inventory and plan:** enumerate the unit, functional and three tool-test binaries using Catch2's XML listing. Preserve the existing non-daily test scope, verify that filtering does not drop tool tests, and assign test IDs using each case's name, tags and source location. Cases matched by the same case-insensitive name filter in one binary stay in an indivisible scheduling group, with each case counted separately. Use available duration history to balance work and a fixed seed for reproducibility. Also enumerate every numbered example executable and bind its relative path, size and SHA-256 to the same inventory.
3. **Test:** each shard runs up to four test processes in batches targeting 25 seconds and normally capped at 32 cases; a case-insensitive name group is never split to meet the batch size. Each process has separate working directories, temporary directories and logs. Examples are assigned across all four shards using the existing Catch2 load estimates and their recorded durations, longest first. Historical example weights use a namespaced executable-relative path identity so they survive ordinary binary rebuilds; the size and SHA-256 bound to the current inventory still authenticate execution. Only examples without a matching history entry use the conservative 300-second fallback. These values are scheduling weights, not runtime limits: long examples and Catch2 batches start first in the shared worker queue. Each shard schedules its assigned examples and Catch2 batches in the same four-worker pool, so examples can overlap test batches without increasing the process limit. Every shard first verifies the complete example inventory; each example keeps its own working directory, temporary directory and log, with its original 1,800-second timeout. Those checks remain mandatory, and a failed case cannot be hidden by another shard succeeding. Existing Catch2 skips are retained: exit code 4 is accepted only when a complete report proves that every planned case in that batch was skipped without a failure.
4. **Compatibility:** an independent job starts after Build, alongside the four test shards. It uses the same approved inline source-verification bootstrap and runtime setup through YAML aliases, a private workspace, and read-only artifacts and compatibility inputs. It verifies the complete required release manifest before executing all maintained compatibility checks, then uploads a separate small result bound to the full run context and separate diagnostic logs.
5. **Summary:** a GitHub-hosted job requires the formatting/helper and independent compatibility jobs to succeed and verifies the plan, all four result manifests, the independent compatibility result and the additional check outcomes. The compatibility result must match the repository, run, attempt, PR head, tested SHA and image, and prove both artifact initialization and compatibility execution succeeded; missing, stale, failed, skipped or cancelled results fail the gate. Missing, duplicate, unexpected or unsuccessful cases or examples fail the result. Example results must also match their assigned shard and planned executable hashes, cover the global example inventory exactly once, and report exit code zero. If the existing change filter selects no relevant files, the summary explicitly reports the intentional skip.

The workflow uses `--batch-target-seconds 25 --batch-size 32` to pack known short groups from one binary while keeping slow, unknown and oversized indivisible groups alone. All identity and XML checks remain mandatory. Disabling precompiled headers can change source locations reported by Catch2, so duration hints must match the freshly generated inventory's exact IDs. Old path identities are not normalized by the PR workflow; unmatched hints are ignored and unmatched groups run alone.

### Local artifacts and cache

Large build outputs stay on shared local storage. The build job writes a fresh artifact directory, including the complete regular-file Git tree of the tested commit; test and compatibility jobs consume the completed directory read-only. The source export excludes `.git`, credentials in Git configuration, untracked downloads and logs. Symlinks, submodules and other non-regular tracked modes are rejected. Consumer jobs do not fetch the repository again: an inline bootstrap approved with the workflow validates the source manifest, exact file inventory, modes and hashes before copying files into each empty private workspace. The copy preserves the canonical checkout path used by the build and does not write to shared artifacts. Each publication is bound to the repository ID, workflow run ID, attempt, PR head SHA, tested SHA and image digest. Every consumer verifies the execution identity and complete manifests before running exported scripts or binaries. The manifest includes the source-file hashes and is only a consistency check; the deployment authenticates the admitted run and the approved workflow bootstrap. GitHub artifacts carry the small plan, results, merged summary and diagnostics, rather than transferring the compiled build tree between jobs.

Compiler-cache input is optional. A trusted publisher supplies a read-only snapshot with an image/compiler/architecture/sanitizer/build-options/path-scheme namespace. The build-options namespace includes `CMAKE_DISABLE_PRECOMPILE_HEADERS=ON`, so a previous PCH-enabled snapshot cannot match. The build copies a matching snapshot into its own writable cache; an absent or rejected snapshot starts a cold build. PR jobs never publish back into the trusted snapshot. A publisher may include a separate, read-only duration-history file whose size and hash are bound by the cache manifest. History stays outside compiler-cache entries and is not copied into the private ccache. Valid hints improve shard balance; missing or rejected hints fall back to scheduling the complete inventory. They never authenticate results or change test selection. The test inventory, plan and result format uses schema 3 for per-shard example assignments. The independent trusted history envelope remains schema 1; its hints match exact Catch2 IDs or namespaced example-path IDs. Catch2 fallback estimates use only current Catch2 history, and stale IDs are ignored. Successful merged results include both measured case and example durations for a trusted publisher to review and refresh; PR jobs cannot update the shared snapshot themselves. Old plan/result formats are rejected rather than mixed across attempts.

Compatibility inputs have a different contract: the complete release snapshot and its manifest are required. The verifier checks release identity, exact file inventory, sizes and digests before the compatibility tool runs. Missing or invalid inputs fail the check; they do not silently skip compatibility testing. Publisher authentication and protection of these read-only inputs belong to the runner deployment, not to checksums supplied by PR code.

### Deployment prerequisites and reruns

The deployment must provide capacity for four test consumers plus the independent compatibility consumer to overlap, with adequate resources, run-scoped artifact storage, a writable build mount followed by read-only consumer mounts, and private writable work and temporary directories. It must authenticate each mount's association with the admitted run, stop all artifact writers before allowing consumers, prevent access to other runs and trusted publishers, and clean up after success, failure and cancellation. Checkout credentials must not persist in the worktree. Workflow consistency checks do not replace these deployment controls.

Use **Re-run all jobs** after a failed or cancelled team run. A new attempt must rebuild and republish its own artifacts and plan. Re-running only failed jobs can leave successful build jobs in an older attempt; consumers then reject that stale identity or missing artifact directory. Do not reuse another attempt's files or let individual shards rebuild independently. Verify cancellation, cleanup and a full rerun against the deployed runner before enabling this workflow for PRs.

## Required checks and rollback

**PR CI Gate** requires community checks and, when selected, team checks to succeed. Reusable-workflow check names gain `Community`/`Team` prefixes. Review existing required-check names and migrate only after the gate's first successful run, without leaving a gap in branch protection.

To disable team routing globally, set `team_enabled` to `false` in canonical main's `.github/ci-routing.json`, preserving the image. Submit the change with a subject such as `[skip ci] ci: disable team routing`; preserve the prefix in the final merge commit if push CI should also be skipped. To enable routing, use the JSON boolean `true` after runner/image validation. No repository variable or label controls this route.

The change takes effect only after it reaches main and the routing job runs again. Cancel affected old runs and **rerun all jobs**; rerunning only failed jobs can reuse old route outputs. Updating the config does not reroute existing jobs or automatically rerun open PRs. The router fetches the current main SHA at execution time, including on full reruns. A rerun of an older workflow revision that still reads repository variables cannot use this new mechanism; first update that PR to the file-based router.

`[skip ci]` skips applicable push/pull-request workflows; it does not bypass branch protection. Required checks can remain Pending and block a configuration PR from merging. Follow the repository's authorized merge process, or omit the skip instruction and run the required checks. See [GitHub's skip-workflow documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/skip-workflow-runs).

## Validation

```bash
node --test tests/scripts/ci_runner_route.test.cjs
python3 -m unittest tests/scripts/test_dependency_workflows.py \
  tests/scripts/test_build_benchmark_workflow.py \
  tests/scripts/test_team_test_shards.py \
  tests/scripts/test_team_mixed.py \
  tests/scripts/test_team_workflow.py \
  tests/scripts/test_team_cache.py \
  tests/scripts/test_team_source.py \
  tests/scripts/test_compatibility_inputs.py \
  tests/scripts/test_compatibility_cache.py \
  tests/scripts/coverage_diagnostics_test.py
node .github/scripts/check-docs.mjs
```

Routing tests execute the actual inline router with mocked API responses and cover policy source pinning, forks, author identity, fallback, stale events and failure propagation. Shard tests cover inventory completeness, exact filtering, merge failures, timeout and cancellation; workflow tests cover the multi-job contract and build-directory handling. The CMake/Ninja regression needs both tools to exercise the real build path. These checks do not replace a complete team workflow run, warm/cold cache validation, or deployment lifecycle tests. Measure completion time only after those checks pass; no performance result is implied by this configuration.

## Advisory Team timing summaries

Team build, test shards and compatibility jobs retain diagnostic artifacts.
The final hosted Team Summary job publishes one consolidated GitHub Job Summary.
It compares completed job and step
intervals from the **current run attempt** against
`scripts/ci/team_timing_baseline.json`, frozen from successful PR #3143,
run 40171699713, attempt 2. The baseline was measured with a warm compiler cache.
It is descriptive evidence, not a guarantee that all PRs or cold builds match it.

The 15-minute advisory budget is divided into build (7 minutes), the parallel
validation envelope (7 minutes), and handoff/checks tail (1 minute). The parallel
envelope is earliest consumer job start to latest consumer job completion;
it includes four shards and compatibility without adding their overlapping times.
The total runs from build job start to the last required execution job completion.
The same baseline scope is **690 seconds**, distinct from the previous **753.991
seconds** that included routing and final resource cleanup. Initial admission queue,
this reporting job, final resource cleanup and Community jobs are excluded;
internal scheduling gaps remain included and are not called measured queue time.

Individual job and command sub-budgets are diagnostic and must not be added to the
15-minute total. Missing, skipped or partial results are reported as unavailable,
not as zero-cost success. Failed jobs retain available timings and their failure
conclusion. Budget overruns only produce warnings: all existing test and artifact
identity gates remain authoritative. API access is read-only and confined to the
hosted reporting job; inability to retrieve timing data does not fail the PR.

The native Jobs API supplies complete job/step timing; existing shard result JSON
and ccache statistics supply local diagnostics. Collapsed sections list the slowest
cases/examples, which overlap across workers and therefore cannot be summed into
wall-clock time. Raw diagnostic artifacts retain their existing three-day lifetime.
No background timing collector, infrastructure mutation or automatic baseline
refresh is introduced. Workflow-version authorization remains a separate deployment
operation; this change does not bypass it.

Validate the reporter offline with `python3 -m unittest tests/scripts/test_team_timing.py`.

Only the final Team Summary job publishes Markdown; execution jobs retain their diagnostic artifacts. The stage budget table stays expanded, with job/step timings, compiler cache and shard details under closed `<details>` sections in a fixed order. Budget usage is green at or below 90%, yellow above 90% through 100%, and red above 100%, using colored icons and text rather than custom CSS. Red remains advisory and never changes the PR gate.

## Community failure propagation

Community and standalone Python jobs use the pytest process exit status, including
pipeline failures, as their result. Structured JUnit/JSON evidence is uploaded even when pytest fails;
matching words such as `passed` in a log never overrides a nonzero exit status.
Community changed-source lint passes the generated pattern through a quoted
environment variable, and manual dispatch defaults its comparison branch to `main`.

Community and standalone compatibility jobs use
`scripts/ci/download_compatibility_inputs.py` to download into a new job-local
directory. The helper reads all release asset pages, checks the exact selected
`.index`/`.bin`/`.json` inventory and file sizes, verifies GitHub-provided SHA-256
digests when present, and rejects release inventory changes during download.
Older assets without an API digest still receive inventory and size checks;
this is not a cryptographic integrity guarantee for those assets. Download
failures are fatal. The compatibility tool then validates and tests the input
set; Team's separately verified read-only snapshot remains unchanged.

Regression coverage is included in
`python3 -m unittest tests/scripts/test_dependency_workflows.py`.

## Preparing the lint base

Team and Community prepare the lint comparison immediately after checkout and
workspace prerequisites, before compiling. `scripts/linters/prepare_lint_base.py`
fetches the exact PR event base SHA, verifies it is that commit, and computes the
changed-source filter before publishing step outputs. Lint consumes the saved
filter after compilation without another network request. A failed fetch or diff
never becomes a successful empty filter. Manual dispatch resolves canonical
checkout origin's `main` once and then pins that SHA; PR events never fall back
to a moving branch when their base SHA is absent.

Recognized transient transport errors and process timeouts get at most four
attempts, each bounded to 60 seconds, with 2/5/10-second backoff plus up to one
second of jitter. Authentication, missing refs and unrecognized failures stop
immediately. Diagnostics identify the operation, attempt, duration and error
class without printing potentially credential-bearing transport output. No
compiler work is started until preparation succeeds. Lint file-selection policy
and build configuration are unchanged by this network hardening.

Regression checks: `python3 -m unittest discover -s scripts/linters -p 'test_*.py'`.

## GCC 13 / LLVM 20 image rollout

The maintained base images use Ubuntu 26.04; Ubuntu 24.04 is a separate development/runtime compatibility target. Rebuild and publish both architecture base images and their BLAS variants from the updated Dockerfiles before activating the new checks. Existing image digests retain the previous tools; update the configured team image digest to the newly published x86 OpenBLAS image. Team jobs cannot install tools as root and deliberately fail prerequisite checks for stale images. Hosted jobs use Ubuntu 24.04 with the shared toolchain installer. Image publication and team routing activation require the existing deployment workflow; a Dockerfile change alone does not update deployed images.


The image upgrade can land on main ahead of the C++20 migration. Copy the following files together into the image-only PR; the base Dockerfiles need the new installer in their build context:

- `docker/Dockerfile.x86.base`
- `docker/Dockerfile.aarch64.base`
- `scripts/deps/install_toolchain_ubuntu.sh`
- `.github/workflows/update-ci-image.yml`
- the `performance` profile in `.github/workflows/daily_test.yml` (use a virtual environment with the CI image Python)

These changes build independently of the CMake, source, formatter configuration and release-package changes in the toolchain PR. The existing x86 MKL/OpenBLAS and aarch64 OpenBLAS Dockerfiles inherit the upgraded bases without edits. The manylinux `Dockerfile.dist_*` changes belong to the toolchain/release migration and are not required for this image-only PR.

The image workflow watches both the Dockerfiles and the shared installer. PR events build the complete image chains without pushing; merging the image PR triggers the existing push workflow to publish the tags. After publication, obtain the `linux/amd64` manifest digest from the `ghcr.io/antgroup/vsag:ci-main-openblas` index, verify GCC/G++ 13 and clang-format/clang-tidy 20 in that image, and commit that exact digest to `.github/ci-routing.json` on canonical main. Preserve the existing routing switch unless intentionally changing routing policy. The toolchain PR then needs a new run or **Re-run all jobs** to read the new main configuration and use the new image; rerunning only failed jobs can retain the old image selection.

## Main-target PR check scope

A newer push cancels the previous PR run, including the Community reusable-workflow
caller (`!cancelled()`). Final Gate reporting still uses `always()` to retain failed
or incomplete results; manual shared checks still run when route selection is
intentionally skipped.

PRs targeting `main` keep the Linux ASan tests, examples, compatibility, changed-source
lint, formatting, routing checks and `PR CI Gate`. Team and Community remain alternative
Linux execution paths, not different correctness requirements.

- Python wheel build/test (`build-python-x86` and `test-python-x86`) runs on main-target PRs
  that change Python/bindings, the C ABI, or CMake/packaging files (the `python` and `cmake`
  path filters). It builds the `py3-none` wheel with the shared Linux cibuildwheel action and runs complete ordinary pytest against the installed artifact. Main PRs test one primary environment (Python 3.12, NumPy 2); the Python 3.10-3.14 /
  NumPy 1.x-2.x matrix and the OpenMP coexistence test stay on version branches and manual runs, and
  belong in Nightly. The wheel build tolerates the intentionally skipped Community format job on
  Team-routed C++ changes, but still stops on failed or cancelled prerequisites. Main pushes restore the same primary Python environment through Branch CI; explicit manual Python checks remain available.
  Python selection also covers native dependency/build definitions, `scripts/python/`,
  `scripts/deps/`, the Python CI/cache helpers, Python helper regressions and the
  PR/Branch/Nightly/TestPyPI workflows that call the shared Python actions. The shared
  `daily_measure.py` test wrapper and its `daily_plan.py`/`team_test_shards.py` imports
  remain included because installed-wheel tests execute them even outside Nightly. Unrelated
  test-planning, coverage, lint and maintenance helpers no longer trigger a wheel
  build merely because they live under `scripts/` or `.github/workflows/`. Selected
  wheels still receive the same build and complete ordinary pytest checks.
- `.github/workflows/pyvsag-testpypi.yml` builds the pyvsag `py3-none` wheels for x86_64
  and aarch64 from `main` every night (03:17 Beijing time, cron `17 19 * * *`) and on manual
  dispatch, and can upload them to TestPyPI with the `TEST_PYPI_API_TOKEN` secret. Uploading is
  on by default: manual runs can set `publish: false` to only build, and the repository variable
  `PYVSAG_NIGHTLY_PUBLISH=false` pauses the nightly upload. Its shared build action restores
  the compiler cache described below; only Nightly CI publishes weekly snapshots. A scheduled
  TestPyPI run skips when the last successful scheduled run already built the same main commit
  (manual runs do not skip). Every shared build uses `VSAG_USE_SYSTEM_DEPS=OFF` and checks the
  repaired wheel with `scripts/python/check_wheel_libs.py`, rejecting bundled or linked libgfortran,
  libquadmath, libaio and liburing before artifacts are uploaded. `make pyvsag` retains the same
  check for its Docker path when system dependencies are unset or OFF; explicit local AUTO builds
  may legitimately depend on system libraries. TestPyPI publication remains independent of PR CI
  and never uploads to PyPI or a GitHub Release.
- macOS ASan builds and TSan no longer run for main-target PRs. Their main-push and
  manual workflows remain unchanged. Platform-specific and race regressions can thus
  be discovered after merge rather than before it.
- Community formatting is skipped only when Team is selected and its identical C++
  change filter selects the full-repository formatting check. Python-only changes
  retain Community formatting. The final gate still requires the selected formatting
  check. Version/manual Python builds retain their existing formatting dependency.
- CircleCI retains a short classification step, then halts redundant tests for a
  confirmed open main-target PR whose head matches the running commit. No credentials
  are required. Missing metadata, stale commits, ambiguous target branches or API
  failures conservatively retain the old build. Thus a lightweight CircleCI job may
  remain visible; this is not disabling the integration or claiming tests executed.

Version-branch PRs and manual PR-CI dispatch retain the previous scope. Main-push,
nightly/daily and release workflows are not rescheduled or changed here. No required
checks or runner deployment settings change. Regression coverage lives in
`python3 -m unittest tests/scripts/test_dependency_workflows.py`.

## Main integration and daily regression

Tracking plan: [CI layers #3225](https://github.com/antgroup/vsag/issues/3225).
Update that issue's original body before changing a layer and after verification;
do not post progress comments. PR policy above remains unchanged.

`main-ci.yml` handles main pushes independently: no branch-wide cancellation and
no migration to private runners. The classifier compares the entire immutable
push `before..after` range (including deletions), not just the last commit.
Documentation-only changes skip heavy checks. Unknown paths, absent history or
failed diff operations conservatively select heavy checks and full lint.

| Layer | Checks |
| --- | --- |
| Each main push | ARM64/macOS ASan builds plus fixed platform smoke inventory; Python X86 wheel plus Python 3.12 / NumPy 2 ordinary tests; existing TSan concurrent tests; changed implementation lint, expanded to full lint for headers/build/dependencies/unknown changes |
| Nightly scheduled/manual | Existing daily/SIMD/bundled-OpenBLAS/concurrent tests plus full X86/ARM64/macOS ASan suites, all numbered examples (including macOS), both Python architectures, full lint, Coverage and historical compatibility |
| Version branches and standalone manual workflows | Existing full scope; reusable workflow inputs default to full behavior |

Format and path-selected documentation checks use `main-ci.yml`; image publishing remains separate.
CircleCI's duplicate main build is removed; its version branch filters and PR
classifier remain. Full X86 ASan still covers the MKL image in daily, and the
existing daily path retains bundled OpenBLAS coverage. CircleCI's generic debug
build wrapper is not another required duplicate daily job.

The smoke inventory is `scripts/ci/platform_smoke.json`: named memory/file IO and
serialization unit cases, factory validation/BruteForce functional cases, an assertion-bearing Pyramid build/add/search/serialization round trip, and
HGraph, BruteForce, IVF and streaming-persistence examples. The Pyramid example
is excluded from smoke because its search failure path returns exit code zero;
full daily example coverage remains unchanged. Each selected Catch2
case must produce a nonempty successful JUnit report; missing binaries, no tests,
skipped cases, failures and timeouts fail the job. Smoke cases use seed 424242 and
300-second per-process bounds. Full macOS examples use a 1,800-second bound.
No assertions or algorithm implementations are changed. Platform runtime success
must be demonstrated by platform CI; helper tests alone do not establish it.

`Main CI Summary` runs even after upstream failures and requires every selected
reusable workflow to succeed; unexpected skips/cancellations/missing results fail.
Lint runs the changed `src/*.cpp` implementation paths (excluding `_test.cpp`),
with full lint for affected headers or shared build configuration. The release
build used to generate lint inputs is retained in this scope-only change.

Daily checkouts use the invocation's `github.sha`, never moving `main` midway.
Extended-daily artifact/log names are distinct from full-regression names, and
full daily ASan artifacts remain available until retention expiry. Daily jobs
retain independent run-specific concurrency; the schedule stays commented out.
Until daily is separately enabled, full regressions removed from main require an
explicit manual Daily Build & Test run. This PR does not claim daily acceptance
or new main runtime targets. Failures remain visible; restore/fix and rerun the
same relevant revision before using daily evidence for release readiness.

Validate with `python3 -m unittest tests/scripts/test_main_ci.py` and the existing
workflow helper suites. No production runner configuration or trusted digest
change is part of this migration.

## Consolidated checks and event isolation

Code checks are grouped by `pr-ci.yml` (PR CI) and `main-ci.yml` (Branch CI).
PR CI has exactly six top-level jobs: Select CI route, CI Routing Tests, Select
checks, Community, Team and PR CI Gate. Docs, SIMD Coverage and Lite live inside
Community, which denotes hosted resources and still runs for Team-routed PRs.
They execute once and do not extend the internal Team concurrency lock.
Metadata validation lives in `pr-issue-link.yml` (PR Metadata), and manual entry
points live in `manual-checks.yml` (Manual Checks). Body/title and label edits
only trigger PR Metadata, never an empty PR CI run. PR CI listens only to
opened, reopened and synchronize events. After changing the target branch,
push a new commit (an empty commit is sufficient) to validate the new base;
do not rely on a successful Gate from the previous target branch.
Job names are short and fixed; only Lite includes its matrix mode.

| Check | Automatic selection | Manual entry |
| --- | --- | --- |
| Format | Push to main or `0.*`, except documentation-only changes | Manual Checks: format |
| Docs | PR opened/reopened/synchronize against any branch, or main push; original path filters | Manual Checks: docs |
| Issue Link | PR opened/edited/reopened/synchronize/labeled/unlabeled targeting main or `0.*` | None |
| SIMD Coverage | PR opened/reopened/synchronize targeting main; original path filters | None |
| Lite | PR opened/reopened/synchronize targeting main or `0.*`; original path filters | Manual Checks: lite |
| Core PR CI | PR opened/reopened/synchronize targeting main or `0.*` | Manual Checks: full-pr |
| Main integration | Non-deleted main pushes | Manual Checks: full-branch |

Docs, Format and Lite command steps are shared composite actions under
`.github/actions/check-{docs,format,lite}/`, reused by automatic and manual jobs.
These actions do not add workflow entries. Matrix modes remain visible per job.

Full manual checks call the existing PR/Branch workflows, with an explicitly
manual caller job prefix. Manual-only docs/format/Lite do not call the core
workflows, so they cannot publish skipped automatic gate/build checks. Code
workflows no longer expose selective workflow_dispatch inputs. The manual wrapper
keeps read-only permissions; full-pr routes to Community, as before.

The required names `PR CI Gate` and `Format Check` remain unchanged.
Gate requires successful selection and Community execution, plus routing tests
and Team when applicable. Community's Shared Checks Result verifies that each
selected Docs/SIMD/Lite job (including the complete Lite matrix) succeeded;
only unselected checks may be skipped. Failure, cancellation or unexpected skips
fail the result and therefore Gate. On other PR target branches, only the
original docs scope runs through Community with core checks disabled; Gate
checks that intended route explicitly. Metadata remains independently enforced
by Mergify; it does not rerun code checks. Other names
include `Community / Docs`, `Branch Docs`, `Community / SIMD Coverage`,
`Issue Link` and `Community / Lite <mode>`.

SIMD Coverage still compares clean base/head coverage with the same test filter,
seed and thresholds. It builds only `unittests_simd` and its dependencies using
`scripts/coverage/simd_coverage.mk`, with a CI-only CMake overlay that also includes
the Local SQ reference tests tagged `[simd]` outside `src/simd/` and preserves
the aggregate executable's normal library initialization/ISA diagnostics. The overlay applies
to both revisions, including bases that predate this helper. Ordinary `make cov`
and local module builds are unchanged. Adding `[simd]` tests outside the module
requires updating the coverage inventory guard and verifying aggregate/module
inventory and SIMD line-coverage equivalence.

The dependency preflight rejects changes to shared dependency definitions and
`prepare-fetchcontent` before reusing prepared sources for base and head. It
excludes the bundled OpenBLAS recipe because both builds require system OpenBLAS
from the same container (`VSAG_USE_SYSTEM_OPENBLAS=ON`). This exception does not
skip either build or coverage comparison; changes to other dependency definitions
still fail the preflight, including when combined with OpenBLAS changes. The
existing exception for the removed, unreferenced pybind11 recipe is unchanged.

Branch Docs is distinct from PR Docs. The 2026-11-02 ruleset audit found main
requires DCO, Mergify Merge Protections and PR CI Gate; version rulesets include
Format Check. Mergify checks issue links directly from labels/body. Recheck
protections before merge if they change.

Issue Link uses a PR/head lock before reading live body/labels, and rejects a
closed PR or changed head. PR code concurrency is set before path selection;
heavy downstream jobs have no cancellation groups. Branch commits run
independently. Branch deletions skip selection before checkout. A failed path
selector causes the required Format Check to fail explicitly, rather than skip.

The selector preserves original path scopes using the complete Git diff,
including deletions and renames. PR comparisons use merge-base; push comparisons
use before/after, and new branches use the oldest pushed commit's parent.
Unavailable ranges fail visibly. Original check commands, images, matrices,
timeouts and artifacts remain unchanged. Existing full version workflows remain
unchanged. The retired label workflow is deleted; its script/manifest remain.

This stage leaves 23 workflow YAML files versus the original 27: four old code
entries and Label configuration are removed, Issue Link stays separate, and one
manual entry is added. It prioritizes reliable event isolation over removing the
last two entries. Team/Community remain reusable workflows.

Changing pr-ci.yml changes the trusted Team caller digest. Review and authorize
its final digest via the private operations process before Team admission. This
PR does not change production admission, runner configuration or protections.

Validation: the checks-policy, main-policy, routing, Team and SIMD helper suites,
Lite collector fixtures, documentation checks and actionlint. The SIMD fixtures
require Linux timeout. Local tests do not establish actual PR/branch runtime
success.

### Check identity on metadata and out-of-scope events

PR CI does not subscribe to edited: a metadata-only run of the same workflow
can obscure the full run in the PR merge widget. Target-branch changes require
a subsequent push; this is a maintainer procedure, not an automatically enforced
retarget guard. Re-running an old run is not a substitute for validating the new
base. A one-element naming matrix produces PR CI Gate for maintained targets
and Docs Gate for other targets. The gate shell verifies its context matches
the selected core scope. Metadata never publishes a successful code Gate.

Branch formatting/docs use the same always-instantiated context pattern. Deleted
branches and successful out-of-scope selection publish Format Check Not Required
or Branch Docs Not Required, never an automatic validation name. Selected checks
retain Format Check / Branch Docs; selection failures fail the applicable check.
The no-op identity jobs use hosted runners briefly and run no checkout/check steps.

Manual PR grants actions: read at that calling job only, alongside contents/read
and pull-requests/read, because nested reusable workflows cannot increase their
caller's token permissions even when the Team branch will be skipped.

## Workflow entry layout

The Actions entries are PR CI, Branch CI, Nightly CI, Release, Maintenance, CI Images,
Manual Checks, and PR Metadata. Team CI and Community CI remain reusable groups
under PR CI; their standalone Actions pages do not own independent run history.
Common hosted execution steps live under `.github/actions/`; entry workflows own
job runners, dependencies, selection and permissions. No workflow generator is needed.

- PR CI retains the required `PR CI Gate` name. PR Metadata alone handles body/label
  edits. Retargeting a PR still requires pushing a new commit before code validation.
- Branch CI retains a separate run for each main commit. Main uses platform smoke,
  TSan, selected lint and Python x86 wheel/install/ordinary tests; version branches retain their full standalone
  scopes, including Coverage and compatibility. SIMD path checks are also here.
- Manual Checks offers full PR/branch profiles and individual ASan, TSan, Python,
  lint, Coverage, compatibility, SIMD, toolchain and performance checks. The actual
  selected jobs appear inside that manual run, not a separately dispatched run.
- Nightly CI runs complete regression on upstream main at 22:00 Asia/Shanghai daily
  (`0 14 * * *` UTC), including supplemental x86 work and the final Nightly Summary.
  The workflow filename remains `daily_test.yml`; manual/reusable regression is
  still available. The weekday runtime-performance schedule is removed; runtime
  performance remains manual. The weekly build-performance schedule is unchanged
  and never selects regression. No compiler-cache or package-publishing schedule changes.
- Maintenance preserves docs-only push filtering for docs synchronization and
  main-push contributor updates. It never executes on PR events.
- Release consolidates release creation, Wheels and old-version indexes. The
  existing Release workflow is disabled in repository settings and must stay so
  until publication is separately authorized. Manual toolchain checks remain
  available through Manual Checks without enabling publication.

All new shared actions must be reviewed together with their callers. Moving a
command into a composite action does not change the contributor-side validation
requirements. For toolchain checks of historical commits, load the shared action
from the workflow's immutable source before checking out the requested target.

After merge, verify the intended runs before disabling retired workflow entries.
First check references on maintained version branches as well: GitHub workflow
state is repository-wide, so disabling an old entry while a version branch still
uses it would suppress that branch's checks. Keep historical run records unless
separate deletion is explicitly requested. A smaller workflow directory does not
by itself remove historical entries from the Actions sidebar.

### Performance profile source and concurrency

The Nightly CI `performance` profile runs only in the canonical repository's main
context (manual, or a reusable call from a main push; no scheduled execution). Like
`build-performance`, it accepts `inputs.ref` only as `main` or a full commit SHA
reachable from fetched upstream main. Both profiles share the same inline ref
validator through a YAML anchor; validation precedes dependency installation and
execution, and the actual commit is recorded in the job summary. Notification
credentials are scoped to the runtime profile's report step.

Each runtime-performance run/attempt has its own concurrency group with
`cancel-in-progress: false`; separate invocations do not cancel each other.
Runs can execute concurrently if GitHub-hosted capacity permits. This does not
change the Team pool. A future nightly performance service on dedicated Team CI
resources is outside the current activation change.

See [Nightly regression budgets and seven-day rotation](daily-regression.md) for scheduled and manual regression, reproduction, and acceptance boundaries.

### Shared Linux Python build and restored Branch checks

PR, Branch, Nightly and TestPyPI build jobs call `.github/actions/python-wheel-build`.
It requires native Linux Docker, pins cibuildwheel v3.3.1, and uses the maintained
`python/pyproject.toml` manylinux 2.28 / repair configuration. The dependency policy
is `VSAG_USE_SYSTEM_DEPS=OFF`; x86 OpenBLAS uses `GENERIC`. CI never silently falls
back to a host wheel. Source preparation and the `vsag_c` Release build remain shared.
The in-container repaired-wheel gate deliberately runs the complete ordinary `tests/python`
directory, excluding only the separately required OpenMP integration module. Its report is
`build-tests.json`; a failure blocks artifact upload and all consumers, including publication.
Dedicated environment lanes additionally check compatibility on their respective host runtimes.
After repair, `check_wheel_libs.py` rejects forbidden bundled/NEEDED libraries before upload.

The compressed ccache is capped at 3 GB per architecture. Restore keys include architecture
and build configuration, and immutable save keys additionally use the UTC ISO week (not a
run ID or source SHA). Only successful canonical-main **Nightly CI** scheduled/manual builds
save, and only on an exact-key miss; PR, Branch and TestPyPI restore only. Thus unchanged
configuration writes at most two snapshots a week. Older snapshots remain subject to GitHub
cache eviction; this cap is not a repository-wide storage guarantee. New namespaces start cold.
Changes to hashed build inputs (including `python/setup.py`, `extern/**/*.cmake` and the
shared build action) create a new namespace: without a compatible entry, PR builds stay cold
until that configuration reaches main and a successful Nightly CI build populates it.
Fork PRs can read base-branch caches subject to GitHub's cache scope rules; they do not write
trusted main entries. Restoring an entry does not prove compilation hits.

Container counters are best-effort diagnostics: machine-readable counters are preferred, with
portable `ccache -s` text retained separately for older versions. Unavailable or unknown counters
remain unmeasured and never fall back to an unrelated host cache or fail the build. No PR
artifacts or cache writes are promoted to publication.

Each caller retains its immutable source selection: PR validation SHA, actual
Branch push SHA, frozen Nightly SHA, or TestPyPI's independently resolved main SHA.
A separate manifest binds that SHA, architecture, filename and SHA-256 to exactly
one wheel. Test jobs verify it before installation and reject source-tree imports.
The publication job selects only wheel artifacts; provenance is separate. TestPyPI's
source resolver, triggers, credentials and publishing job are unchanged.

Branch main pushes run x86 wheel construction plus Python 3.12 / NumPy 2 full
ordinary tests. Pure documentation skips heavy work; unknown changes/missing diffs
conservatively select Python. Branch Summary rejects missing, failed, cancelled or
unexpectedly skipped selected Python jobs. Version/manual routing keeps its scope.
No OpenMP dependency requirement is added to the main PR or Branch primary lane.

See [the Nightly Linux matrix](daily-regression.md#linux-python-matrix) for mandatory
Python/NumPy compatibility and OpenMP environments. Native macOS Python integration
is deferred to the separately owned packaging adaptation; Linux results do not
validate macOS wheels.

## Branch ARM build diagnostics

The Branch CI Aarch64 ASan build keeps its existing three-way compilation and
examples. `scripts/ci/monitor_build.py` wraps that build without changing its
exit code and streams `[build-resource]` JSON samples every 15 seconds. Samples
include available host memory, accessible cgroup memory limits/OOM counters,
CPU accounting, and build-process RSS and compiler source operands. This is
useful because Ninja prints completed commands: the last printed source is not
necessarily the source still compiling. Arbitrary command lines and environment
variables are not captured.

The best-effort `arm-build-diagnostics-<job>-<run>-<attempt>` artifact retains
`samples.jsonl` and `summary.json` for seven days. Diagnostic upload failure does
not change the build result. Cancellation is forwarded to the build process
group, and a stuck group is killed after a ten-second grace period. If the runner
is abruptly terminated, the upload/final sample cannot be guaranteed; consult
the already streamed samples and GitHub's runner annotations. Exit 137 or a
shutdown signal alone does not establish OOM. Compare counter deltas and memory
pressure; unavailable cgroup evidence is unknown, not proof of no OOM.

### Hosted formatting environment

Branch, Community and Manual Format checks run on GitHub-hosted Ubuntu runners
inside `vsaglib/vsag:ci-main-base`. Their shared `check-format` action verifies
preinstalled clang-format major version 20 and runs the existing full-repository
format check; it does not install packages at job runtime. A missing or wrong
formatter fails the check. Container Git trust is limited to `GITHUB_WORKSPACE`.
The Branch `Format Check Not Required` job uses no container and retains its
existing no-op check identity. Team keeps its reviewed digest image and existing
format/helper job. Image pulls still require registry connectivity. Image-size
optimization is tracked separately in [#3327](https://github.com/antgroup/vsag/issues/3327).

## Per-case and example timing CSVs

Team Summary exports `team-test-timings-<run_id>-<run_attempt>` (30-day retention)
with two UTF-8 CSV files. Download it from the workflow run's Artifacts section.
This is post-processing of existing Catch2 XML-derived results and example process
measurements; it does not run tests again, add per-case subprocesses, change
sharding, or publish trusted timing history. The existing JSON manifests and
complete-result gate remain authoritative for CI correctness.

- `test-timings.csv`: one row per planned Catch2 case or numbered example, including
  failed, skipped, timed-out, cancelled and missing results. Tool-binary cases are
  included. It covers the Team plan's non-daily scope, not every Daily configuration.
- `test-duration-estimates.csv`: only successful measured cases/examples from
  successful batches, grouped by repository, performance identity and environment.
  Per-run estimates usually have one sample. Multi-run estimates use the most
  recent 20 samples by default, with median, nearest-rank p90, minimum, maximum and
  last-seen time. Identical records deduplicate; conflicting duplicates are rejected.

The raw schema is versioned (`schema_version=1`). Key fields:

| Fields | Meaning |
| --- | --- |
| `record_id`, `repository_id`, `run_id`, `run_attempt`, `plan_digest` | Exact observation provenance; reruns are separate samples. |
| `head_sha`, `commit_sha` | PR head and actual tested commit (usually the synthetic merge commit). |
| `case_id`, `execution_id` | Performance hint ID and strict sealed execution identity. The hint hashes kind, binary, exact name, tags and source file, excluding source line; examples use their binary path. It survives line shifts, but source-file/path or name changes intentionally create a new hint. Strict source-line and binary-hash verification is unchanged. |
| `kind`, `binary`, `case_name`, `tags`, `source_file`, `source_line` | Human-readable case or example identity. |
| `platform`, `build_profile`, `image` | Execution environment; the image digest distinguishes toolchains. Never pool different sanitizer/platform/image configurations. |
| `workers`, `batch_size`, `batch_target_seconds`, `omp_num_threads`, `openblas_num_threads` | Concurrency and batching settings, also separated in estimates. |
| `shard`, `seed`, `observed_at` | Assignment, random seed and shard completion time in UTC (not a per-case start/finish timestamp). |
| `duration_seconds`, `measurement` | Catch2 top-level case duration (`catch2_case`) or example process wall duration (`process_wall`), in seconds. |
| `status`, `batch_status`, `batch_seconds` | Case/example outcome and enclosing batch outcome/wall duration. Batch duration repeats for its cases: do not sum it per row. |

Missing/unusable per-case XML yields an empty duration, never an invented zero or
an apportionment of batch time. Examples cancelled before launch or failing to
launch also have empty durations and `measurement=unavailable`; a real exit code,
including cancellation/timeout signals, preserves the measured process wall time.
A missing shard yields `missing` rows; a batch that
reported success without usable case data yields `unavailable`. Any case with
reported skips is excluded from estimates. If the job is killed before publishing
its result, its individual timings cannot be recovered from these small artifacts.
A plan must exist to enumerate missing observations. CSV failures do not turn failed
CI into success or relax the gate.

To regenerate reports locally from downloaded plan/results:

```bash
python3 scripts/testing/team_test_timings.py export \
  --plan plan/plan.json --results-dir results \
  --platform linux-amd64 --build-profile asan \
  --omp-num-threads 2 --openblas-num-threads 2 --output-dir timings
```

To aggregate downloaded raw CSVs from multiple runs:

```bash
python3 scripts/testing/team_test_timings.py estimate \
  --input run-a/test-timings.csv --input run-b/test-timings.csv \
  --window 20 --output test-duration-estimates.csv
```

Excel, Numbers, pandas and DuckDB can read these CSVs. Empty duration cells mean
unknown, while a numeric zero is a real measurement. Text cells beginning with
spreadsheet formula characters (or an apostrophe) receive a protective leading
apostrophe; the helper reverses this escaping when reading CSVs. The estimator
is a bounded sample window, not an elapsed-day window. Random parameter values are
not collected yet, so these data support duration estimation but cannot by
themselves explain or fit dimension/data-size cost coefficients.

These reports are observations, **not authorization or trusted shared history**.
They do not automatically replace the planner's existing ID-to-seconds JSON or
update the production cache publisher. Promotion and scheduling changes require
separate validation; a digest verifies consistency, not who produced the data.


### Nightly regression reporting and spare-capacity expansion

Nightly regression publishes one final **Nightly Summary**. Individual jobs upload
small diagnostic records instead of ccache Summary panels; PR, Branch and manual
checks keep their existing panels. The final report distinguishes scoped C++ line
coverage (90% target) from test/collection/upload success. Missing coverage is
reported as unmeasured. A green Coverage job alone does not certify the target.
The README Nightly badge is a budget/report link, not a status claim: the same entry
also contains performance-only profiles, whose green runs must not stand in for a
complete regression.

Named-case counts deduplicate by source file and exact case name. Configuration
execution counts retain repeated parameter/platform executions and deduplicate
sibling shards. Neither number counts assertions; examples, Python and compatibility
remain separately reported command groups.

After mandatory regression checks succeed, **X86 Extra Coverage** reuses the
original OpenBLAS binaries in four shards, with two test workers per shard. A
frozen proposal contains only other days' non-quarantined functional configurations.
It uses at most four additional hours and stops testing by nine hours from the
current attempt start, leaving an hour within the ten-hour envelope. Queueing and
artifact setup consume that allowance. Workers select additional configurations
while their measured same-case estimate (with margin; conservative fallback for
unknown cases) fits the remaining quota. They never delay completion just to fill
time. Original core/rotation selection and ARM scope are unchanged.

The final report separates supplemental wall time, runner-minutes, test-process
minutes, started/passed/failed configurations and budget-deferred candidates.
Deferred candidates are **not validated**, never passed; an executed failure or
missing shard evidence still fails the run. Runner/process minutes are not actual
CPU-core consumption. The initial cutoff is an operating target, not a guarantee
against GitHub capacity delays. No separate Weekly full rerun is introduced: the
seven-day planned union still needs execution evidence, and release-only scope
remains a separate obligation. Scheduled regression uses the same checked execution
and reporting path as manual regression. The intended 22:00–08:00 window may shift
when GitHub delays dispatch; the ten-hour allowance is measured from attempt start.

Daily diagnostic identity (source SHA, run ID, attempt, job and job status) is required. Missing identity fails with an explicit diagnostic; optional compiler-cache measurements may be unavailable without failing collection. Full functional inventories must match every configured quarantine exactly; stale or core-targeted rules fail planning instead of silently disappearing.
