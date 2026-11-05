<!-- agent-hints
canonical: docs/agents/docs-map.md
purpose: Map of VSAG markdown docs and where to look for what
key-facts:
  - Canonical user-facing docs live in docs/docs/{en,zh}/src/
  - docs/README.md links local docs/docs sources to published vsag.io pages
  - Tool-level READMEs are lightweight pointers to docs/docs and vsag.io
related:
  - https://vsag.io/docs
  - ../../README.md
  - ../README.md
last-reviewed: 2026-06-07
-->

# Documentation Map

This page is a navigation aid for agents. Before making non-trivial changes,
locate the right document below first.

## Repository structure (high level)

- `include/`: public headers (canonical for public APIs)
- `src/`: core implementation and unit tests
- `tests/`: functional tests
- `examples/cpp/`, `examples/python/`, `examples/typescript/`: example programs
- `python/`: `pyvsag` packaging and Python-side code
- `python_bindings/`: pybind11 bindings
- `docs/`: design and user-facing documentation (see below)
- `tools/`: utility and analysis tools

## Where docs live

### Agent-facing rules

- [`AGENTS.md`](../../AGENTS.md) — index + hard constraints (entry point).
- [`docs/agents/`](.) — detailed agent operating docs (this directory).
- [`.github/agent-prompts/`](../../.github/agent-prompts/) — shared agent
  workflows (e.g. `/create-issue`).

### User-facing project docs

- [`README.md`](../../README.md) — landing page, performance highlights,
  multi-language Quickstart.
- [`DEVELOPMENT.md`](../../DEVELOPMENT.md) — developer onboarding.
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — contribution policy.
- [`CODE_OF_CONDUCT.md`](../../CODE_OF_CONDUCT.md).

### Documentation source of truth

The canonical user-facing documentation is `docs/docs/{en,zh}/src/`; the public website at
<https://vsag.io/docs> is generated from that tree. Keep user-facing guide, index, advanced,
resource, and tool documentation there first.

Top-level `docs/*.md` files should not be used for new user-facing documentation. A small number of
remaining files are historical design notes that have not yet been merged into the website source;
when they are promoted, move the canonical content into `docs/docs/{en,zh}/src/` and leave only a
short local pointer if a compatibility link is still useful.

`docs/README.md` is the local documentation index for developers and AI coding agents. It links to
both local `docs/docs/...` sources and the published website pages.

### Tool READMEs

Tool-level READMEs stay next to the executable source as lightweight entry points. For tools that
have user-facing docs, the README should link to both the local `docs/docs/{en,zh}/src/...` source
and the published `https://vsag.io/docs/...` page rather than duplicating the full content.

- [`tools/README.md`](../../tools/README.md)
- [`tools/eval/README.md`](../../tools/eval/README.md) /
  [`tools/eval/README_zh.md`](../../tools/eval/README_zh.md)
- [`tools/analyze_index/README.md`](../../tools/analyze_index/README.md) /
  [`tools/analyze_index/README_zh.md`](../../tools/analyze_index/README_zh.md)
- [`tools/check_compatibility/README.md`](../../tools/check_compatibility/README.md) /
  [`tools/check_compatibility/README_zh.md`](../../tools/check_compatibility/README_zh.md)
- [`tools/issue-helper/README.md`](../../tools/issue-helper/README.md)
- [`benchs/README.md`](../../benchs/README.md)
- [`scripts/csv_extract/README.md`](../../scripts/csv_extract/README.md)

Chinese versions use the `_zh.md` suffix convention here (kept as-is to
preserve published URLs and user expectations).

### Blog

- `docs/blog/{en,zh}/src/` — long-form articles.

## When to update which doc

| Change | Update |
| --- | --- |
| New user-facing feature | `README.md`, relevant page under `docs/docs/{en,zh}/src/`. |
| New build flag or dependency | `DEVELOPMENT.md`, `docs/agents/build-and-test.md`. |
| Workflow / policy change | `CONTRIBUTING.md`, the matching file under `docs/agents/`. |
| New index parameter | `docs/docs/{en,zh}/src/resources/index_parameters.md` and the per-index page. |
| Behavior change in `eval_performance` | `docs/docs/{en,zh}/src/resources/eval.md`; keep `tools/eval/README{,_zh}.md` as link-only entry points. |

## Pending follow-ups

- Historical design notes that still live at top level under `docs/` should be merged into
  `docs/docs/{en,zh}/src/` or replaced with short pointers once their website destination is clear.
