<!-- agent-hints
canonical: docs/agents/contribution-workflow.md
purpose: Branching, fork-first PR pushes, commits, DCO, Assisted-by, PR labels, doc sync rules
key-facts:
  - Prefer pushing PR branches to a contributor fork, not antgroup/vsag
  - Use Conventional Commits with DCO sign-off; AI agents must NOT self-sign
  - AI agents: do NOT use `git commit -s` (it appends Signed-off-by last, with a blank line before it). Write both trailers manually in the commit body, deriving the human Signed-off-by from `git config user.name`/`user.email`
  - Trailer order: Signed-off-by first, Assisted-by immediately after, no blank line between them
  - Every PR needs both a kind/* and a version/* label
  - PRs labeled kind/bug or kind/feature must link an issue with Fixes:/Closes:/Resolves: #N (or owner/repo#N / issue URL)
related:
  - ../../CONTRIBUTING.md
  - ../../.github/pull_request_template.md
last-reviewed: 2026-08-13
-->

# Contribution Workflow

## Branching

- Before modifying code, create and switch to a working branch from an
  up-to-date `main`.
- When preparing a pull request, inspect the configured remotes and prefer the
  contributor's fork as the push target. Treat `antgroup/vsag` as the upstream
  repository, not the default destination for working branches.
- Push a working branch directly to `antgroup/vsag` only when the task
  explicitly requires it or the contributor has authorized it.
- Keep changes scoped and reviewable.

## Required validation before opening or updating a PR

Reducing main-target PR CI does **not** waive contributor-side testing. Before
opening a PR or pushing a substantive update, agents must run the applicable
checks moved out of PR CI in a local development environment (an authorized
remote development machine/container also counts). Do not use production CI
runner pods as a development environment. Run against the final proposed source;
repeat affected checks after subsequent changes.

| Removed/reduced PR check | Required pre-submission validation |
| --- | --- |
| Python wheel build and tests | For Python/bindings, C++ library/public API, dependencies or packaging/build changes: build a fresh wheel with `make pyvsag`, install that wheel in a clean Python 3.10+ environment, and run `python -m pytest tests/python -v --tb=short`. Use the dependencies/environment from the maintained Python workflow; an already installed wheel is not validation of the changed source. |
| TSan | For C++ library, tests or build/dependency changes: run `make tsan`, then the maintained TSan workflow's `Run TSAN tests` command, including `OPENBLAS_NUM_THREADS=1`, its suppression file and `./build/tests/functests "[concurrent]~[daily]"`. Use a separate checkout/build from ASan. A successful build alone is insufficient. |
| macOS ASan | For C++ or shared build/dependency changes: on macOS arm64 run the maintained workflow's dependency setup and `CMAKE_GENERATOR=Ninja CMAKE_EXPORT_COMPILE_COMMANDS=ON make asan VSAG_ENABLE_EXAMPLES=ON COMPILE_JOBS=3`. Linux compilation does not validate macOS. |
| CircleCI Linux build/tests | For C++ library/tests or build/dependency changes: run `make test_parallel` with the dependency environment and build options of `.circleci/continue_config.yml`'s `pull-request-check`. Do not reproduce disposable CI swap-file or host-cleanup steps on a shared development host. |
| Duplicate Community format | Run `./scripts/format/check_format.sh` before submission when code formatting is applicable, even when Team will check it again. |

Pure documentation changes do not require C++ builds. CI/helper-only changes
require the affected helper regression tests and workflow validation; if they
also change compiler, dependency, test execution or packaging behavior, run the
corresponding checks above. Explain applicability in the PR; do not silently
classify risky changes as unrelated.

Record the tested commit/source revision, platform/toolchain, exact commands and
outcomes in the PR validation section. Keep full logs locally; do not expose
private infrastructure paths or credentials. Missing macOS/Python/toolchain
support is **not** a pass: explicitly report the unvalidated checks and obtain
maintainer confirmation before submitting without them. Likewise report real
failures and obtain confirmation for any exception; do not mask failures or
substitute future main/daily checks for the required pre-submission validation.

The main-target Python PR suspension is temporary; these contributor-side
requirements apply during it. Restore automated Python PR checks after the
Python rework is ready. See [CI routing](ci-routing.md#main-target-pr-check-scope)
and [build/test guidance](build-and-test.md).

## Commit messages

- Follow Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`,
  `refactor:`, etc.
- Commits in this project are expected to carry a DCO `Signed-off-by:`
  trailer.
- **Only humans can legally certify the DCO; AI coding agents must not add
  their own `Signed-off-by` trailer.** The human submitter is responsible for
  reviewing AI-generated changes, ensuring license compliance, and taking
  full responsibility for the contribution.
- **AI agents must NOT invoke `git commit -s`.** `-s` appends a
  `Signed-off-by:` trailer at the very end of the message and inserts a
  blank line before it if other trailers are present, which separates the
  human sign-off from the `Assisted-by:` line and reverses the required
  order. Instead, write both trailers directly in the commit message body.
  Derive the human identity from the active git configuration
  (`git config user.name` and `git config user.email`) and format it as
  `Signed-off-by: <Name> <email>`.
- For changes produced with AI assistance, attribute the agent with an
  `Assisted-by:` trailer in the form `Assisted-by: AgentName:ModelVersion`
  (see the [Linux kernel AI Coding Assistants policy](https://docs.kernel.org/process/coding-assistants.html)).
  Determine `AgentName` and `ModelVersion` at execution time from the active
  agent name and model name (for example
  `Assisted-by: OpenCode:claude-opus-4.7`).
- **Trailer order and spacing**: a single blank line separates the subject
  (and optional body) from the trailer block. Inside the trailer block,
  place the human `Signed-off-by:` first and the `Assisted-by:` line
  immediately after, with **no blank line between them**. Example:

  ```
  docs: reorganize agent guidance

  Signed-off-by: Jane Doe <jane@example.com>
  Assisted-by: OpenCode:claude-opus-4.7
  ```
- When using skip-CI commit messages, follow the repository convention and
  place `[skip ci]` at the **beginning** of the subject line.

## Pull request labels

Every pull request **must** have two labels before it can be merged:

- A `kind/*` label for the type of change: `kind/bug` (bug fix),
  `kind/feature` (new feature), `kind/improvement` (refactor, chore, or minor
  improvement), or `kind/documentation` (documentation change).
- A `version/*` label for the target version (e.g. `version/1.0`,
  `version/0.18`).

Mergify enforces these via check runs. Always add both labels when creating
a PR.

## Linking an issue

PRs labeled `kind/bug` or `kind/feature` **must** reference an existing
issue in the PR description using a GitHub-recognized auto-closing keyword
(`close[sd]?`, `fix(e[sd])?`, `resolve[sd]?`, case-insensitive, optionally
with a colon), followed by `#N`, `owner/repo#N`, or a full issue URL.
Examples:

```text
Fixes: #1234
Closes #1234
Resolves: antgroup/vsag#999
Closes: https://github.com/antgroup/vsag/issues/100
```

`kind/improvement` and `kind/documentation` PRs are exempt — link an issue
only if it helps reviewers. The placeholder line in
`.github/pull_request_template.md` lives inside an HTML comment and does
**not** satisfy the rule on its own; you must replace it with a real
reference.

Enforcement happens in two places:

1. The `PR Issue Link Check` GitHub Action runs on every PR open / edit /
   sync targeting `main` or `0.*` branches. Repo administrators are
   expected to add it to the **required status checks** on those
   branches so a failed check actually blocks merge.
2. A Mergify `merge_protections` rule (`Require linked issue for
   feature/bug PRs`) blocks merge as a defense-in-depth fallback.

## Documentation expectations

Update relevant docs when behavior changes:

- `README.md` for user-facing features or examples.
- `DEVELOPMENT.md` for build or environment changes.
- `CONTRIBUTING.md` for workflow or contribution policy changes.
- Website docs under `docs/docs/{en,zh}/src/` together with any related
  in-repo READMEs (`tools/eval/README.md`, `tools/eval/README_zh.md`, etc.).
  Keep English and Chinese versions in sync.
- Run `node .github/scripts/check-docs.mjs` after documentation or referenced source-path changes.
