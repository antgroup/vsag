<!-- agent-hints
canonical: docs/agents/README.md
purpose: Index of agent-facing operating docs split out of AGENTS.md
key-facts:
  - AGENTS.md is the entry point; this directory holds the full rules
  - Read the relevant sub-doc before non-trivial changes
  - Sub-docs are the single source of truth for their topic
related:
  - ../../AGENTS.md
last-reviewed: 2026-05-12
-->

# Agent Operating Docs

This directory holds the detailed operating rules referenced from
[`AGENTS.md`](../../AGENTS.md). `AGENTS.md` itself is intentionally short and
acts as an index plus a list of hard constraints; everything below provides
the full context for a given topic.

| Document | Purpose |
| --- | --- |
| [`build-and-test.md`](build-and-test.md) | Build / test commands, development environment, required tool versions. |
| [`coding-standards.md`](coding-standards.md) | C++ style, layout rules, common code patterns, testing expectations. |
| [`contribution-workflow.md`](contribution-workflow.md) | Branching, commit messages, DCO, `Assisted-by`, PR labels, docs sync. |
| [`filing-issues.md`](filing-issues.md) | How agents draft and submit GitHub issues for VSAG. |
| [`docs-map.md`](docs-map.md) | Map of the repository's markdown documentation and how the pieces relate. |

## How to use these docs as an agent

1. Always read [`AGENTS.md`](../../AGENTS.md) first — it lists the
   non-negotiable hard constraints.
2. Open the sub-doc that matches the current task (build issue, code style
   question, PR prep, issue filing, doc navigation).
3. When updating behavior that any of these docs describe, update the
   sub-doc in the same change. `AGENTS.md` only needs an update when a
   hard constraint or top-level pointer changes.
