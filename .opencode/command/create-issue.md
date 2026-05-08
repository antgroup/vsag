---
description: Draft a VSAG GitHub issue from a code/doc snippet and submit via gh.
agent: build
---

Load and follow the instructions in
@.github/agent-prompts/create-issue.md verbatim. That file is the single source
of truth for the workflow shared by Claude Code, OpenCode and Codex.

When you append the footer, use `OpenCode:<model>` as `<AgentName>:<ModelVersion>`.

User arguments (optional): $ARGUMENTS
