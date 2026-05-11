# VSAG Issue Writing Guide

This guide is read by both humans and AI coding agents. Follow it whenever you
file an issue against [`antgroup/vsag`](https://github.com/antgroup/vsag).

## Pick the right template

| Situation                                            | Template            |
| ---------------------------------------------------- | ------------------- |
| Something is broken or behaves unexpectedly          | `bug.yml`           |
| You want a new capability or new public API          | `feature.yml`       |
| Refactor / perf / chore / test-only change           | `improvement.yml`   |
| Wrong, missing, or unclear documentation             | `documentation.yml` |
| Usage question (consider Discussions first)          | `question.yml`      |

## Title format

```
[<prefix>](<area>): <short imperative summary>
```

* `prefix`: `bug` / `feat` / `improve` / `docs` / `q`
* `area`:   `hgraph` | `ivf` | `sindi` | `pyvsag` | `eval_performance` | `build` | `docs` | `other`
* Summary: imperative mood, lowercase, no trailing period.

Good:

```
[bug](hgraph): segfault when build_thread_count exceeds physical cores
[feat](ivf): expose nprobe as runtime parameter
[docs](eval_performance): clarify dataset format for sparse vectors
```

Bad:

```
help me!
HGraph crashes
new feature
```

## Required quality bar

* **One issue, one topic.** Split unrelated problems into separate issues.
* **Cite your sources.** When the issue is grounded in a doc page or a code
  region, include the link. Use `path:line` for repo references and full URLs
  for `vsag.io/docs` pages.
* **Bug reports must be reproducible.** Provide the smallest C++ or Python
  snippet that triggers the problem. Mark unknowns with `// TODO`.
* **Bug reports must include environment.** Run
  `bash scripts/check_environment.sh` and paste the output. Add anything the
  script does not capture.
* **Performance claims need numbers.** Dataset, parameters, before/after
  latency, recall and memory. Reference `tools/eval/` or `benchs/` whenever
  possible.
* **Feature requests need motivation.** Explain who benefits and why this
  cannot be solved with existing knobs.

## Labels

The issue templates pre-apply a `kind/*` label that mirrors the PR
[`kind/*`](../../AGENTS.md#pull-request-labels) policy, so triage and release
notes stay consistent. The one exception is `question.yml`, which applies the
plain `question` label — usage questions do not normally produce a PR and so
do not need a `kind/*` value.

Maintainers may additionally add during triage:

* `area/<name>` — mirrors the `area` field.
* `version/<x.y>` — target release line.

## Drafting with an AI agent

You can let Claude Code, OpenCode or Codex draft the issue body for you and
submit it via `gh`. Two entry points:

* Inside an agent session: `/create-issue`
* From a shell: `./tools/issue-helper/new-issue.sh --help`

The agent will:

1. Read the source you point at (a repo path, a `vsag.io/docs` URL, or pasted
   text).
2. Run `bash scripts/check_environment.sh` for bug reports.
3. Run `gh issue list --search ...` and append matches to a `## Related issues`
   block to help reviewers spot duplicates.
4. Render the chosen template with strict field validation.
5. Append a footer of the form
   `_Drafted with AI assistant: <AgentName>:<ModelVersion>_` for transparency.
   This is informational only — issues do not require DCO sign-off.
6. Show a dry-run preview and only call `gh issue create` after you confirm.

## Anti-patterns

* Pasting screenshots of code instead of code text.
* Filing usage questions as bugs.
* Combining "fix the docs and also add a new index type" in one issue.
* Linking to private gists or internal URLs that maintainers cannot open.
* Using `Signed-off-by:` in issue bodies — DCO applies to commits, not issues.
