## Change Type

- [ ] Bug fix
- [ ] New feature
- [ ] Improvement/Refactor
- [ ] Documentation
- [ ] CI/Build/Infra

## Linked Issue

<!--
REQUIRED for `kind/bug` and `kind/feature` PRs. Use a GitHub-recognized
auto-closing keyword so the linked issue is closed when this PR merges:

  Fixes: #<number>
  Closes: #<number>
  Resolves: #<number>

Cross-repo references (`owner/repo#<number>`) and full issue URLs are
also accepted. `kind/improvement` and `kind/documentation` PRs may leave
this blank (delete the line below or keep it empty).
-->

- Fixes: <!-- #<number> -->

## What Changed

- <!-- Briefly describe the key changes -->

## Test Evidence

<!--
Follow docs/agents/contribution-workflow.md#required-validation-before-opening-or-updating-a-pr.
Record the tested revision, platform/toolchain, exact commands and outcomes.
Mark checks that do not apply as N/A with a reason; report unvalidated checks
explicitly. Pure documentation changes do not require C++ builds.
-->

- [ ] `make fmt`
- [ ] `make lint`
- [ ] `make test`
- [ ] `make cov`, run tests, and collect coverage
- [ ] Other (describe below)

Test details:

```text
<!-- Paste commands and key output here -->
```

## Compatibility Impact

- API/ABI compatibility: <!-- none / describe -->
- Behavior changes: <!-- none / describe -->

## Performance and Concurrency Impact

- Performance impact: <!-- none / improved / regressed / unknown -->
- Concurrency/thread-safety impact: <!-- none / describe -->

## Documentation Impact

- [ ] No docs update needed
- [ ] Updated docs:
  - [ ] `README.md`
  - [ ] `DEVELOPMENT.md`
  - [ ] `CONTRIBUTING.md`
  - [ ] Website docs under `docs/docs/{en,zh}/src/`
  - [ ] Other: <!-- path -->

## Risk and Rollback

- Risk level: <!-- low / medium / high -->
- Rollback plan: <!-- how to revert safely -->

## Checklist

- [ ] I have linked the relevant issue (required for `kind/bug` and `kind/feature`; see "Linked Issue" above)
- [ ] I have added/updated tests for new behavior or bug fixes
- [ ] I have considered API compatibility impact
- [ ] I have updated docs if behavior/workflow changed
- [ ] My commit messages follow project conventions (Conventional Commits, DCO sign-off, `Assisted-by` when applicable, optional `[skip ci]` prefix)
