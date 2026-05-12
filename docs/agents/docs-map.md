<!-- agent-hints
canonical: docs/agents/docs-map.md
purpose: Map of VSAG markdown docs and where to look for what
key-facts:
  - Website source mirrors are docs/docs/{en,zh}/src/
  - Top-level docs/*.md may overlap with website docs (being reconciled separately)
  - Tool-level READMEs live alongside the tool under tools/<name>/
related:
  - https://vsag.io/docs
  - ../../README.md
last-reviewed: 2026-05-12
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

### Website source (English & Chinese)

The official documentation site at <https://vsag.io/docs> is rendered from:

- `docs/docs/en/src/` — English source.
- `docs/docs/zh/src/` — Chinese source (kept in a parallel directory so
  Chinese pages share the website URL path structure).

Common sub-areas (mirrored in both languages):

- `guide/` — getting started (installation, knn search, pyvsag, create index).
- `indexes/` — per-index user docs (`hgraph`, `ivf`, `sindi`, `pyramid`).
- `advanced/` — advanced features (filtered search, range search, memory,
  optimizer, serialization, attribute filter, extra info, etc.).
- `development/` — building, testing, code structure, contributing.
- `resources/` — index parameters, best practices, performance numbers,
  dataset format, `eval_performance` guide, release notes, papers.

### Top-level `docs/*.md`

These files (`docs/hgraph.md`, `docs/ivf.md`, `docs/sindi.md`,
`docs/brute_force.md`, `docs/dataset_format.md`, `docs/eval_performance.md`,
`docs/get_memory_usage_en.md`) historically contained design notes that
partially overlap with the website docs above. They are being reconciled in
a separate track; treat the website docs (`docs/docs/{en,zh}/src/`) as the
preferred source when content differs, and consult both if the topic
requires deeper background.

### Tool READMEs

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
| Behavior change in `eval_performance` | `tools/eval/README{,_zh}.md` and `docs/docs/{en,zh}/src/resources/eval.md`. |

## Pending follow-ups

- Long top-level design notes (`docs/hgraph.md`, `docs/sindi.md`, etc.,
  >200 lines) should gain `<!-- agent-hints -->` blocks once the reconciliation
  track decides whether to keep them; do not edit those files preemptively
  here.
