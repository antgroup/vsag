# tools/issue-helper

A thin wrapper that lets you file a well-formed VSAG issue from the shell with
the help of a locally installed coding agent (Claude Code, OpenCode, or Codex)
and the GitHub CLI (`gh`).

It does **not** call any LLM directly. It only:

1. Collects the context you point at (URLs, files, optional environment dump)
   into a single Markdown bundle.
2. Picks an agent (env var > `--agent` > autodetect: `opencode` → `claude` →
   `codex`).
3. Hands the bundle plus the shared prompt
   (`.github/agent-prompts/create-issue.md`) to that agent, which then drives
   `gh issue create` after a dry-run preview that you must confirm.

## Prerequisites

* `gh` authenticated against `github.com` (`gh auth login`).
* At least one of: `opencode`, `claude`, or `codex` on `PATH`.
* `curl` (used to attach docs URLs to the bundle; the agent can also fetch
  URLs itself).

## Usage

```bash
tools/issue-helper/new-issue.sh \
  --url https://vsag.io/docs/en/indexes/hgraph \
  --intent "make build_thread_count adjustable at runtime" \
  --kind feature
```

```bash
tools/issue-helper/new-issue.sh \
  --file src/algorithm/hgraph/hgraph_builder.cpp \
  --intent "segfault when build_thread_count exceeds physical cores" \
  --kind bug                # implies --include-env
```

```bash
tools/issue-helper/new-issue.sh \
  --url https://vsag.io/docs/en/indexes/ivf \
  --intent "default value of nlist in docs disagrees with code" \
  --kind docs
```

### Flags

| Flag             | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `--url URL`      | Doc page to attach. Repeatable.                               |
| `--file PATH`    | Repo file to attach. Repeatable.                              |
| `--intent TEXT`  | One-sentence description.                                     |
| `--kind KIND`    | `bug` / `feature` / `improvement` / `docs` / `question`.      |
| `--agent AGENT`  | Force `opencode` / `claude` / `codex`.                        |
| `--include-env`  | Attach `bash scripts/check_environment.sh` output. (Implied with `--kind=bug`.) |

Set `VSAG_ISSUE_AGENT=opencode` (or another) to make the choice sticky across
invocations.

## What you should see

1. The script prints the path of the context bundle it built.
2. The agent enters its normal interactive UI, loads the shared prompt, and
   walks through the workflow described in
   [`.github/ISSUE_TEMPLATE/ISSUE_GUIDE.md`](../../.github/ISSUE_TEMPLATE/ISSUE_GUIDE.md).
3. You confirm the dry-run preview; the agent calls
   `gh issue create --repo antgroup/vsag ...` and prints the resulting URL.

If `gh issue create` fails (e.g. network), the agent falls back to
`gh issue create --web` so you can finish in a browser with the body
pre-filled.

## Inside an agent (no shell wrapper)

You can skip this script and just run `/create-issue` inside Claude Code or
OpenCode. For Codex, see
[`.codex/prompts/create-issue.md`](../../.codex/prompts/create-issue.md).
