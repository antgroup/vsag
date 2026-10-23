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

The team runner must support non-root container jobs and provide the build dependencies. The image prerequisite is supplied separately in [PR #3147](https://github.com/antgroup/vsag/pull/3147). Use the digest of the published x86 OpenBLAS image, not the base image's. The CI image must include clang-tidy 15 and clang-format 15 exactly, GNU time, ccache and GitHub CLI 2.75.0. The CLI version and archive SHA-256 are pinned in `docker/Dockerfile.x86.base`; update both together with the workflow version check. No GitHub token is used during image construction. The team compatibility check uses a pre-staged release snapshot and does not receive `GH_TOKEN`.

The image repository restriction is intentional; changing registries requires a reviewed router change. Enable routing only after the selected image and runner have passed the complete team workflow. Infrastructure provisioning and operational settings are maintained separately.

The token-path prerequisite check detects the standard ServiceAccount token mount only; it does not prove that credentials are absent at other paths. Runner administrators must enforce credential isolation independently of this smoke check.

## Team execution contract

The caller serializes the complete team reusable-workflow invocation with `concurrency.queue: max` and `cancel-in-progress: false`. Different PRs wait for the shared team execution capacity; four test jobs and the independent compatibility job within the admitted run may execute concurrently when execution capacity permits. The outer per-PR concurrency rule still cancels an older run when that PR is updated. Runner cleanup must finish before shared storage can be reused; GitHub concurrency alone does not establish this condition.

For relevant C++ and CI changes, the workflow follows this sequence:

1. **Build and hosted checks:** run the complete formatting and helper-script checks in a separate GitHub-hosted job while Build restores an optional private compiler cache, compiles ASan binaries once with 24 compiler jobs and precompiled headers disabled, and lints changed sources with clang-tidy 15. CMake receives the absolute build-directory path; the checkout's `build` symlink is only a convenience for existing tools.
2. **Inventory and plan:** enumerate the unit, functional and three tool-test binaries using Catch2's XML listing. Preserve the existing non-daily test scope, verify that filtering does not drop tool tests, and assign test IDs using each case's name, tags and source location. Cases matched by the same case-insensitive name filter in one binary stay in an indivisible scheduling group, with each case counted separately. Use available duration history to balance work and a fixed seed for reproducibility. Also enumerate every numbered example executable and bind its relative path, size and SHA-256 to the same inventory.
3. **Test:** each shard runs up to four test processes in batches targeting 25 seconds and normally capped at 32 cases; a case-insensitive name group is never split to meet the batch size. Each process has separate working directories, temporary directories and logs. Examples are assigned across all four shards using the existing Catch2 load estimates and an equal, conservative 300-second hint per example, without changing Catch2 membership or batches. This is a scheduling weight, not a runtime limit: potentially long examples start ahead of shorter Catch2 batches, while Catch2 batches with larger estimates retain priority. Each shard schedules its assigned examples and Catch2 batches in the same four-worker pool, so examples can overlap test batches without increasing the process limit. Every shard first verifies the complete example inventory; each example keeps its own working directory, temporary directory and log, with its original 1,800-second timeout. Those checks remain mandatory, and a failed case cannot be hidden by another shard succeeding. Existing Catch2 skips are retained: exit code 4 is accepted only when a complete report proves that every planned case in that batch was skipped without a failure.
4. **Compatibility:** an independent job starts after Build, alongside the four test shards. It uses the same approved inline source-verification bootstrap and runtime setup through YAML aliases, a private workspace, and read-only artifacts and compatibility inputs. It verifies the complete required release manifest before executing all maintained compatibility checks, then uploads a separate small result bound to the full run context and separate diagnostic logs.
5. **Summary:** a GitHub-hosted job requires the formatting/helper and independent compatibility jobs to succeed and verifies the plan, all four result manifests, the independent compatibility result and the additional check outcomes. The compatibility result must match the repository, run, attempt, PR head, tested SHA and image, and prove both artifact initialization and compatibility execution succeeded; missing, stale, failed, skipped or cancelled results fail the gate. Missing, duplicate, unexpected or unsuccessful cases or examples fail the result. Example results must also match their assigned shard and planned executable hashes, cover the global example inventory exactly once, and report exit code zero. If the existing change filter selects no relevant files, the summary explicitly reports the intentional skip.

The workflow uses `--batch-target-seconds 25 --batch-size 32` to pack known short groups from one binary while keeping slow, unknown and oversized indivisible groups alone. All identity and XML checks remain mandatory. Disabling precompiled headers can change source locations reported by Catch2, so duration hints must match the freshly generated inventory's exact IDs. Old path identities are not normalized by the PR workflow; unmatched hints are ignored and unmatched groups run alone.

### Local artifacts and cache

Large build outputs stay on shared local storage. The build job writes a fresh artifact directory, including the complete regular-file Git tree of the tested commit; test and compatibility jobs consume the completed directory read-only. The source export excludes `.git`, credentials in Git configuration, untracked downloads and logs. Symlinks, submodules and other non-regular tracked modes are rejected. Consumer jobs do not fetch the repository again: an inline bootstrap approved with the workflow validates the source manifest, exact file inventory, modes and hashes before copying files into each empty private workspace. The copy preserves the canonical checkout path used by the build and does not write to shared artifacts. Each publication is bound to the repository ID, workflow run ID, attempt, PR head SHA, tested SHA and image digest. Every consumer verifies the execution identity and complete manifests before running exported scripts or binaries. The manifest includes the source-file hashes and is only a consistency check; the deployment authenticates the admitted run and the approved workflow bootstrap. GitHub artifacts carry the small plan, results, merged summary and diagnostics, rather than transferring the compiled build tree between jobs.

Compiler-cache input is optional. A trusted publisher supplies a read-only snapshot with an image/compiler/architecture/sanitizer/build-options/path-scheme namespace. The build-options namespace includes `CMAKE_DISABLE_PRECOMPILE_HEADERS=ON`, so a previous PCH-enabled snapshot cannot match. The build copies a matching snapshot into its own writable cache; an absent or rejected snapshot starts a cold build. PR jobs never publish back into the trusted snapshot. A publisher may include a separate, read-only duration-history file whose size and hash are bound by the cache manifest. History stays outside compiler-cache entries and is not copied into the private ccache. Valid hints improve shard balance; missing or rejected hints fall back to scheduling the complete inventory. They never authenticate results or change test selection. The test inventory, plan and result format uses schema 3 for per-shard example assignments. The independent trusted history envelope remains schema 1; its hints still match exact Catch2 IDs. Old plan/result formats are rejected rather than mixed across attempts.

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
