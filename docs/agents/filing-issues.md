<!-- agent-hints
canonical: .github/agent-prompts/create-issue.md
purpose: Pointer to the canonical /create-issue workflow shared by all agents
key-facts:
  - Canonical workflow lives at .github/agent-prompts/create-issue.md
  - Writing rules live at .github/ISSUE_TEMPLATE/ISSUE_GUIDE.md
  - Issues do NOT carry Signed-off-by; only commits do
related:
  - ../../.github/agent-prompts/create-issue.md
  - ../../.github/ISSUE_TEMPLATE/ISSUE_GUIDE.md
  - ../../tools/issue-helper/README.md
last-reviewed: 2026-05-12
-->

# Filing Issues With An Agent

Agents are expected to help users file high-quality issues. The full
step-by-step workflow is **not** duplicated here — it lives in a single
canonical file shared across Claude Code, OpenCode, and Codex:

> **Canonical workflow:** [`.github/agent-prompts/create-issue.md`](../../.github/agent-prompts/create-issue.md)
>
> **Writing rules / template guide:** [`.github/ISSUE_TEMPLATE/ISSUE_GUIDE.md`](../../.github/ISSUE_TEMPLATE/ISSUE_GUIDE.md)

The per-agent slash-command wrappers point at the canonical file:

- `.claude/commands/create-issue.md`
- `.opencode/command/create-issue.md`
- `.codex/prompts/create-issue.md`

A shell wrapper is provided at [`tools/issue-helper/new-issue.sh`](../../tools/issue-helper/new-issue.sh)
for users who prefer not to drive the agent's slash command directly.

## Quick rules (must hold regardless of agent)

- Map every required field of the chosen YAML template under
  `.github/ISSUE_TEMPLATE/`. Use `// TODO` for genuinely unknown values.
- Cite sources as `path:line` for repo references and full URLs for
  `vsag.io/docs` pages.
- Run
  `gh issue list --repo antgroup/vsag --search ... --state all --limit 5`
  and append the matches under a `## Related issues` section. Do not block
  creation on duplicates; the maintainers will close as needed.
- Append a footer of the form
  `_Drafted with AI assistant: <AgentName>:<ModelVersion>_` for transparency.
  **Do not** add `Signed-off-by:` to issues — DCO applies only to commits.
- Show a dry-run preview and only call `gh issue create` after the user
  confirms. On failure, fall back to `gh issue create --web`.
